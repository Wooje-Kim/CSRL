import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path

PROTOCOL = 'SD'
MODEL = 'facebook/wav2vec2-base'
SPEAKERS = {'F02': 2, 'F03': 3, 'F04': 1, 'F05': 0, 'M01': 3, 'M04': 3, 'M05': 1, 'M07': 2, 'M08': 0, 'M09': 0, 'M10': 0, 'M11': 1, 'M12': 3, 'M14': 0, 'M16': 2}
LAMBDAS = [0.025, 0.05, 0.10, 0.20, 0.40]
TEMPERATURES = [0.05, 0.10, 0.20, 0.50, 1.00, 1.50]

EARLY_STOPPING_PATIENCE = 3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def save_tensor(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_tensor(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def read_inputs(folder, heldout):
    speakers = set(SPEAKERS) - ({heldout} if heldout else set())
    parts = []
    for name, count in [('train', 124), ('val', 31)]:
        with (folder / f'{name}.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        require(rows, f'Empty split: {folder}/{name}')
        for row in rows:
            row['severity_label'] = int(row['severity_label'])
            require(row['speaker'] in speakers, 'Unexpected or held-out speaker')
            require(row['severity_label'] == SPEAKERS[row['speaker']], 'Invalid severity label')
            require(row['audio_version'] == 'original' and row['microphone'] == 'M5', 'Expected original/M5')
            require(row['is_common'] == '1' and row['is_uncommon'] == '0', 'Expected common training words only')
            require(row['block'] in {'B1', 'B2', 'B3'}, 'Invalid block')
            require(Path(row['path']).is_file(), f"Missing audio: {row['path']}")
        words = {r['word_key'] for r in rows}
        require(len(words) == count, f'Expected {count} words: {name}')
        counts = Counter((r['speaker'], r['block'], r['word_key']) for r in rows)
        expected = {(s, b, w) for s in speakers for b in ['B1', 'B2', 'B3'] for w in words}
        require(set(counts) == expected and all(n == 1 for n in counts.values()), 'Incomplete split')
        for key in ['utterance_id', 'path']:
            require(len({r[key] for r in rows}) == len(rows), f'Duplicate {key}')
        parts.append(rows)
    for key in ['utterance_id', 'path', 'word_key']:
        require(not ({r[key] for r in parts[0]} & {r[key] for r in parts[1]}), f'Train/val overlap: {key}')
    common = {f'D{i}' for i in range(10)} | {f'L{chr(i)}' for i in range(65, 91)} | {f'C{i}' for i in range(1, 20)} | {f'CW{i}' for i in range(1, 101)}
    require({r['word_key'] for rows in parts for r in rows} == common, 'Unexpected word vocabulary')
    return parts


def audio_fingerprint(rows):
    h = hashlib.sha256()
    for r in rows:
        p = Path(r['path'])
        stat = p.stat()
        h.update(f'{p.resolve()}\t{stat.st_size}\t{stat.st_mtime_ns}\n'.encode())
    return h.hexdigest()


def episode_plan(rows, epoch, seed):
    pool = {}
    for i, r in enumerate(rows):
        pool.setdefault(r['word_key'], {}).setdefault(r['severity_label'], {}).setdefault(r['speaker'], []).append(i)
    words = sorted(pool)
    for word in words:
        for label in range(4):
            require(len(pool[word].get(label, {})) >= 2, 'Need two speakers per severity per word')
    rng = np.random.default_rng(seed + epoch * 10007 + 27271627)
    plan = []
    for _ in range(math.ceil(len(rows) / 8)):
        pair = []
        for word in rng.choice(words, size=2, replace=False):
            group = []
            for label in range(4):
                for speaker in rng.choice(sorted(pool[word][label]), size=2, replace=False):
                    candidates = pool[word][label][speaker]
                    group.append(candidates[int(rng.integers(len(candidates)))])
            pair.append(group)
        plan.append(pair)
    return np.asarray(plan, dtype=np.int64)


class Encoder:
    def __init__(self, device, amp):
        from transformers import AutoFeatureExtractor, Wav2Vec2Model
        self.device, self.amp = device, amp
        self.extractor = AutoFeatureExtractor.from_pretrained(MODEL)
        self.model = Wav2Vec2Model.from_pretrained(MODEL).to(device).eval()
        self.model.requires_grad_(False)
        require(self.model.config.feat_extract_norm == 'group', 'Expected group-normalized backbone')

    def forward_waveforms(self, rows):
        self.model.eval()
        import soundfile as sf
        waves = []
        for r in rows:
            wave, sr = sf.read(r['path'], dtype='float32', always_2d=False)
            if wave.ndim == 2:
                wave = wave.mean(axis=1)
            require(sr == 16000 and wave.ndim == 1 and len(wave) >= 400 and np.isfinite(wave).all(), f"Invalid audio: {r['path']}")
            waves.append(wave)
        values = self.extractor(waves, sampling_rate=16000, padding=True, return_attention_mask=False, return_tensors='pt')['input_values'].to(self.device)
        lengths = torch.tensor([len(w) for w in waves], device=self.device)
        with torch.no_grad(), precision(self.device, self.amp):
            h = self.model(input_values=values, attention_mask=None).last_hidden_state
            lengths = self.model._get_feat_extract_output_lengths(lengths).clamp(1, h.shape[1])
            mask = torch.arange(h.shape[1], device=self.device)[None, :] < lengths[:, None]
            result = (h * mask.unsqueeze(-1).to(h.dtype)).sum(1) / lengths.to(h.dtype).unsqueeze(1)
        require(bool(torch.isfinite(result).all()), 'Nonfinite encoded features')
        return result.detach()


def precision(device, amp):
    return torch.autocast(device_type='cuda', dtype=torch.float16) if amp else nullcontext()


def make_head():
    return torch.nn.Sequential(torch.nn.Linear(768, 768), torch.nn.ReLU(), torch.nn.Linear(768, 4))


def contrastive(z, temperature):
    z = F.normalize(z.float(), dim=-1, eps=1e-8)
    sims = z @ z.T
    candidates = [[i ^ 1] + [j for j in range(8) if j // 2 != i // 2] for i in range(8)]
    indexes = torch.tensor(candidates, device=z.device)
    logits = sims.gather(1, indexes) / temperature
    return F.cross_entropy(logits, torch.zeros(8, device=z.device, dtype=torch.long))


def train_epoch(encoder, head, optimizer, scaler, rows, plan, lam, tau, device, amp, progress):
    head.train()
    labels = torch.arange(4, device=device).repeat_interleave(2)
    totals = np.zeros(3, dtype=np.float64)
    skipped = 0
    for update, pair in enumerate(plan, 1):
        optimizer.zero_grad(set_to_none=True)
        for indices in pair:
            with precision(device, amp):
                h = encoder.forward_waveforms([rows[int(i)] for i in indices])
                z = head[:2](h)
                logits = head[2](z)
                ce = F.cross_entropy(logits, labels)
                csrl = contrastive(z, tau)
                loss = ce + lam * csrl
            require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
            scaler.scale(loss / 2).backward()
            totals += np.array([loss.item(), ce.item(), csrl.item()]) / 2
            del h, z, logits, ce, csrl, loss
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=not amp)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped += int(scaler.get_scale() < previous_scale)
        if update % 50 == 0 or update == len(plan):
            print(f'{progress} train_updates={update}/{len(plan)}', flush=True)
    require(skipped < len(plan), 'All optimizer updates were skipped')
    result = dict(zip(['train_loss', 'train_ce', 'train_csrl'], (totals / len(plan)).tolist()))
    result['skipped_updates'] = skipped
    result['successful_updates'] = len(plan) - skipped
    return result


def evaluate(encoder, head, rows, device, amp):
    head.eval()
    labels = torch.tensor([r['severity_label'] for r in rows], dtype=torch.long)
    confusion = torch.zeros(4, 4, dtype=torch.int64)
    ce_sum = 0.0
    with torch.no_grad():
        for i in range(0, len(labels), 8):
            y = labels[i:i + 8].to(device)
            with precision(device, amp):
                logits = head(encoder.forward_waveforms(rows[i:i + 8]))
                ce = F.cross_entropy(logits, y, reduction='sum')
            require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(ce)), 'Nonfinite validation output')
            ce_sum += ce.item()
            pred = logits.argmax(1)
            confusion += torch.bincount((y * 4 + pred).cpu(), minlength=16).reshape(4, 4)
    c = confusion.double()
    f1 = 2 * c.diag() / (c.sum(0) + c.sum(1)).clamp_min(1)
    return {'val_macro_f1': f1.mean().item(), 'val_ce': ce_sum / len(labels), 'val_accuracy': c.diag().sum().item() / len(labels), 'val_confusion': confusion.tolist()}


def ranking(row):
    return (-row['val_macro_f1'], row['val_ce'], row['epoch'], row['lambda'], row['temperature'])


def early_stopping_status(history):
    """Patience is reset only by a strict increase in validation macro-F1."""
    best_macro_f1 = None
    bad_epochs = 0
    for row in history:
        score = float(row['val_macro_f1'])
        require(math.isfinite(score), 'Nonfinite validation macro-F1')
        if best_macro_f1 is None or score > best_macro_f1:
            best_macro_f1, bad_epochs = score, 0
        else:
            bad_epochs += 1
    return dict(monitor='val_macro_f1', patience=EARLY_STOPPING_PATIENCE,
                best_macro_f1=best_macro_f1, bad_epochs=bad_epochs,
                stopped_early=bad_epochs >= EARLY_STOPPING_PATIENCE)


def cpu_state(head):
    return {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}


def move_optimizer(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and key != 'step':
                state[key] = value.to(device)


@contextmanager
def lock(folder):
    import fcntl
    with (folder / 'run.lock').open('w') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f'Another process is using {folder}') from e
        yield


def run_fold(args, fold, heldout, device, amp, encoder_holder):
    source = args.splits_root / PROTOCOL / fold if heldout else args.splits_root / PROTOCOL
    out = args.output_dir / fold if heldout else args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with lock(out):
        train, val = read_inputs(source, heldout)
        config = {'format': 2, 'input_mode': 'direct_waveform', 'protocol': PROTOCOL, 'fold': fold, 'heldout': heldout, 'seed': args.seed, 'max_epochs': args.max_epochs, 'early_stopping': {'monitor': 'val_macro_f1', 'mode': 'max', 'patience': EARLY_STOPPING_PATIENCE, 'min_delta': 0.0}, 'lambdas': args.lambdas, 'temperatures': args.temperatures, 'model': MODEL, 'amp': amp, 'device_type': device.type, 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu', 'torch': str(torch.__version__), 'transformers': importlib.metadata.version('transformers'), 'numpy': np.__version__, 'script_sha256': digest(__file__), 'train_sha256': digest(source / 'train.csv'), 'val_sha256': digest(source / 'val.csv'), 'audio_stat_sha256': audio_fingerprint(train + val), 'updates_per_epoch': math.ceil(len(train) / 8), 'padding_scope': 'episode8', 'lr': 0.0005, 'betas': [0.9, 0.98], 'selection': 'macro_f1_desc,ce_asc,epoch_asc,lambda_asc,temperature_asc'}
        config_path = out / 'config.json'
        if config_path.exists():
            require(json.loads(config_path.read_text()) == config, f'Run configuration changed; use a new output directory: {out}')
        else:
            atomic_json(config, config_path)
        torch.manual_seed(args.seed)
        initial = cpu_state(make_head())
        combos = [(lam, tau, out / f'lambda_{lam:g}_temperature_{tau:g}') for lam in args.lambdas for tau in args.temperatures]
        for _, _, folder in combos:
            folder.mkdir(exist_ok=True)
        def get_encoder():
            if not encoder_holder:
                encoder_holder.append(Encoder(device, amp))
            return encoder_holder[0]
        for epoch in range(1, args.max_epochs + 1):
            pending = []
            unfinished = False
            for lam, tau, folder in combos:
                path = folder / 'last.pt'
                state = load_tensor(path) if path.exists() else None
                done = state['epoch'] if state else 0
                require(0 <= done <= args.max_epochs, 'Invalid checkpoint epoch')
                if state:
                    require([r['epoch'] for r in state['history']] == list(range(1, done + 1)), 'Invalid checkpoint history')
                    status = early_stopping_status(state['history'])
                    # Skip completed combinations before checking an epoch gap.
                    if status['stopped_early'] or done == args.max_epochs:
                        continue
                unfinished = True
                require(done >= epoch - 1, 'Checkpoint epoch gap')
                if done == epoch - 1:
                    pending.append((lam, tau, folder))
            if not pending:
                if not unfinished:
                    break
                continue
            plan = episode_plan(train, epoch, args.seed)
            plan_hash = hashlib.sha256(plan.tobytes()).hexdigest()
            encoder = get_encoder()
            for lam, tau, folder in pending:
                torch.manual_seed(args.seed + epoch)
                head = make_head().to(device)
                optimizer = torch.optim.Adam(head.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0)
                scaler = torch.amp.GradScaler('cuda', enabled=amp)
                path = folder / 'last.pt'
                state = load_tensor(path) if path.exists() else None
                head.load_state_dict(state['model'] if state else initial)
                if state:
                    optimizer.load_state_dict(state['optimizer'])
                    move_optimizer(optimizer, device)
                    scaler.load_state_dict(state['scaler'])
                history = state['history'] if state else []
                result = {'lambda': lam, 'temperature': tau, 'epoch': epoch, 'plan_sha256': plan_hash}
                progress = f'{fold} epoch={epoch}/{args.max_epochs} lambda={lam:g} temperature={tau:g}'
                result.update(train_epoch(encoder, head, optimizer, scaler, train, plan, lam, tau, device, amp, progress))
                print(f'{progress} validating', flush=True)
                result.update(evaluate(encoder, head, val, device, amp))
                history.append(result)
                best = state['best'] if state else None
                if best is None or ranking(result) < ranking(best):
                    best, best_model = result.copy(), cpu_state(head)
                else:
                    best_model = state['best_model']
                status = early_stopping_status(history)
                completed = status['stopped_early'] or epoch == args.max_epochs
                stop_reason = 'early_stopping' if status['stopped_early'] else ('max_epochs' if completed else 'running')
                save_tensor({'epoch': epoch, 'model': cpu_state(head), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'history': history, 'best': best, 'best_model': best_model, 'early_stopping': status, 'completed': completed, 'stop_reason': stop_reason}, path)
                atomic_json(history, folder / 'history.json')
                print(f"{fold} epoch={epoch}/{args.max_epochs} lambda={lam:g} temperature={tau:g} val_macro_f1={result['val_macro_f1']:.6f} val_ce={result['val_ce']:.6f}", flush=True)
                if status['stopped_early']:
                    print(f"{fold} lambda={lam:g} temperature={tau:g} early_stop_epoch={epoch} best_macro_f1={status['best_macro_f1']:.6f}", flush=True)
                del head, optimizer, scaler, state
        results = []
        winner = None
        for lam, tau, folder in combos:
            state = load_tensor(folder / 'last.pt')
            status = early_stopping_status(state['history'])
            require([r['epoch'] for r in state['history']] == list(range(1, state['epoch'] + 1)), 'Invalid checkpoint history')
            require(state['epoch'] == args.max_epochs or status['stopped_early'], 'Incomplete search')
            atomic_json(state['history'], folder / 'history.json')
            summary = dict(state['best'], trained_epochs=state['epoch'],
                           stopped_early=status['stopped_early'],
                           stop_reason='early_stopping' if status['stopped_early'] else 'max_epochs')
            results.append(summary)
            if winner is None or ranking(summary) < ranking(winner['best']):
                winner = {'best': summary, 'model': state['best_model']}
        results.sort(key=ranking)
        atomic_json(results, out / 'leaderboard.json')
        selected = dict(winner['best'], protocol=PROTOCOL, fold=fold, heldout=heldout, complete=True)
        save_tensor({'config': config, 'selection': selected, 'head_state_dict': winner['model'], 'head_architecture': 'Sequential(Linear(768,768),ReLU(),Linear(768,4))'}, out / 'best.pt')
        atomic_json(selected, out / 'selected.json')
        print(json.dumps(selected, ensure_ascii=False), flush=True)
        return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits-root', type=Path, default=Path('/home/wooje/바탕화면/code/1_preprocessing/splits_tuning'))
    parser.add_argument('--output-dir', type=Path, default=Path('/home/wooje/바탕화면/code/2_calibration/results_direct_waveform') / PROTOCOL)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--lambdas', type=float, nargs='+', default=LAMBDAS)
    parser.add_argument('--temperatures', type=float, nargs='+', default=TEMPERATURES)
    parser.add_argument('--fp32', action='store_true')
    if PROTOCOL == 'LOSO':
        parser.add_argument('--folds', nargs='+', choices=sorted(SPEAKERS), default=sorted(SPEAKERS))
    args = parser.parse_args()
    require(1 <= args.max_epochs <= 50, 'max-epochs must be between 1 and 50')
    for values in [args.lambdas, args.temperatures]:
        require(len(values) == len(set(values)) and all(math.isfinite(x) and x > 0 for x in values), 'Grid values must be unique finite positive numbers')
    global torch, np, F
    import numpy as np
    import torch
    import torch.nn.functional as F
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    device = torch.device(args.device)
    require(device.type in {'cuda', 'cpu'}, 'Use cuda or cpu')
    require(device.type != 'cuda' or torch.cuda.is_available(), 'CUDA is unavailable; specify --device cpu if needed')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    amp = device.type == 'cuda' and not args.fp32
    args.splits_root = args.splits_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    folds = [(f'fold_{s}', s) for s in args.folds] if PROTOCOL == 'LOSO' else [('SD', None)]
    encoder_holder = []
    for fold, heldout in folds:
        run_fold(args, fold, heldout, device, amp, encoder_holder)


if __name__ == '__main__':
    main()
