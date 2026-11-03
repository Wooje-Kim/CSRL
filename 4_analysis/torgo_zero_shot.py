from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
import random
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2FeatureExtractor
SAMPLE_RATE = 16000
WAV2VEC_DIM = HEAD_DIM = 768
NUM_CLASSES = 4
LABEL_NAMES = ['very_low', 'low', 'medium', 'high']
BASE_SEED = 42
TEST_BATCH = 4
NUM_WORKERS = 0
USE_AMP = True
EXPECTED_TORGO_ROWS = 2108
EXPECTED_TORGO_LABELS = {0, 1, 2}
EXPECTED_SPEAKER_INFO = {
    'F03': ('very_low', 0, 401), 'F04': ('very_low', 0, 176),
    'M03': ('very_low', 0, 306), 'F01': ('low', 1, 94),
    'M05': ('low', 1, 352), 'M01': ('medium', 2, 279),
    'M02': ('medium', 2, 293), 'M04': ('medium', 2, 207),
}
TRAIN_ROOT = Path('/home/wooje/바탕화면/code/3_training/results')
MANIFEST = Path('/home/wooje/다운로드/TORGO_raw/zero_shot_manifest_wav_headMic/torgo_zero_shot_manifest.csv')
METRIC_NAMES = ('accuracy','balanced_accuracy_gt_classes','macro_f1_gt_classes',
    'macro_f1_output_space','mae_source_label_order','qwk_output_space',
    'negative_log_likelihood','out_of_target_prediction_rate')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def amp_context(device: torch.device):
    if USE_AMP and device.type == "cuda":
        return torch.autocast("cuda")
    return nullcontext()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

class CompletelyFrozenWav2Vec2(nn.Module):
    """Entire pretrained Wav2Vec2 backbone is frozen and permanently in eval mode."""

    def __init__(self, asset):
        super().__init__()
        config = Wav2Vec2Config.from_dict(asset['encoder_config'])
        if asset.get('attention_implementation'):
            config._attn_implementation = asset['attention_implementation']
        self.wav2vec2 = Wav2Vec2Model(config)
        self.load_state_dict(asset['encoder_state_dict'], strict=True)
        self.requires_grad_(False).eval()

    def validate_frozen(self) -> None:
        trainable = [
            name
            for name, parameter in self.wav2vec2.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            raise RuntimeError(
                "Wav2Vec2 is not completely frozen. "
                f"First trainable parameters: {trainable[:10]}"
            )

    def train(self, mode: bool = True):
        super().train(False)
        self.wav2vec2.eval()
        return self

    def eval(self):
        return self.train(False)

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.wav2vec2.parameters() if p.requires_grad))

    @torch.no_grad()
    def forward(
        self,
        input_values: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.wav2vec2(
            input_values=input_values,
            attention_mask=None,
            return_dict=True,
        ).last_hidden_state

        output_lengths = (
            self.wav2vec2._get_feat_extract_output_lengths(input_lengths)
            .to(hidden.device, dtype=torch.long)
            .clamp(min=1, max=hidden.size(1))
        )

        frame_ids = torch.arange(
            hidden.size(1),
            device=hidden.device,
        )[None, :]

        valid = frame_ids < output_lengths[:, None]
        valid_float = valid.unsqueeze(-1).to(hidden.dtype)

        pooled = (hidden * valid_float).sum(dim=1)
        pooled = pooled / output_lengths.to(hidden.dtype).unsqueeze(1)

        if pooled.shape[-1] != WAV2VEC_DIM:
            raise RuntimeError(
                f"Wav2Vec2 pooled dim={pooled.shape[-1]}, "
                f"expected={WAV2VEC_DIM}"
            )

        return pooled

class SeverityHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(WAV2VEC_DIM, HEAD_DIM),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(HEAD_DIM, NUM_CLASSES)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.head(h)
        logits = self.classifier(z)
        return {"z": z, "logits": logits}

class AudioCollator:
    def __init__(self, extractor: AutoFeatureExtractor):
        self.extractor = extractor

    def __call__(
        self,
        batch: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        waveforms = [item["waveform"] for item in batch]

        lengths = torch.tensor(
            [len(w) for w in waveforms],
            dtype=torch.long,
        )

        encoded = self.extractor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_attention_mask=False,
            return_tensors="pt",
        )

        metadata = [
            {k: v for k, v in item.items() if k != "waveform"}
            for item in batch
        ]

        return {
            "input_values": encoded["input_values"],
            "input_lengths": lengths,
            "metadata": metadata,
        }

def save_json(obj: Mapping[str, Any], path: Path) -> None:
    """정의되지 않은 지표는 JSON null로 기록하며 계산값은 변경하지 않는다."""
    def clean(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(dict(obj)), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")

def load_audio(path: str) -> np.ndarray:
    waveform, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=False,
    )
    if int(sample_rate) != SAMPLE_RATE:
        raise RuntimeError(
            f"{path}: sample_rate={sample_rate}, expected={SAMPLE_RATE}"
        )
    # Fixed manifest is mono. Do not alter the target waveform.
    if waveform.ndim != 1:
        raise RuntimeError(
            f"{path}: expected mono waveform, got shape={waveform.shape}"
        )
    if waveform.size == 0:
        raise RuntimeError(f"{path}: empty waveform")
    if not np.isfinite(waveform).all():
        raise RuntimeError(f"{path}: NaN/Inf waveform")
    return waveform.astype(np.float32, copy=False)

class TORGODataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.reset_index(drop=True).copy()
    def __len__(self) -> int:
        return len(self.df)
    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[int(index)]
        return {
            "index": int(index),
            "utterance_id": str(row["utterance_id"]),
            "path": str(row["path"]),
            "speaker": str(row["speaker"]),
            "severity_label": int(row["severity_label"]),
            "severity_name": str(row["severity_name"]),
            "word_key": str(row["word_key"]),
            "word_text": str(row["word_text"]),
            "waveform": load_audio(str(row["path"])),
        }

def read_manifest(path, audio_root=None, label_map=None, label_map_reference=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'TORGO manifest not found: {path}. Supply the existing --manifest file.')
    if path.stat().st_size == 0:
        raise ValueError(f'TORGO manifest is empty: {path}. No file was overwritten. Restore the existing valid manifest.')
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f'TORGO manifest has no CSV columns: {path}. No inference was started.') from exc
    required = ['utterance_id', 'path', 'speaker', 'severity_name', 'severity_label',
                'word_text', 'word_key', 'audio_subdir', 'sample_rate', 'num_channels']
    if any(c not in df for c in required):
        raise ValueError(f'Manifest missing columns: {set(required)-set(df)}')
    if df[required].isna().any().any():
        raise ValueError('Manifest has missing values in required columns')
    for c in set(required)-{'severity_label', 'sample_rate', 'num_channels'}:
        df[c] = df[c].astype(str).str.strip()
        if df[c].eq('').any(): raise ValueError(f'Empty manifest field: {c}')
    df['speaker'] = df['speaker'].str.upper()
    df['severity_name'] = df['severity_name'].str.lower()
    for c in ('severity_label', 'sample_rate', 'num_channels'):
        v = pd.to_numeric(df[c], errors='raise')
        if not np.isfinite(v).all() or not (v == np.floor(v)).all():
            raise ValueError(f'Noninteger manifest values: {c}')
        df[c] = v.astype(int)
    if len(df) != EXPECTED_TORGO_ROWS or set(df.speaker) != set(EXPECTED_SPEAKER_INFO):
        raise ValueError('Expected original fixed 2108 utterances / 8 TORGO speakers')
    if df.utterance_id.duplicated().any() or df.path.duplicated().any():
        raise ValueError('Duplicate utterance or path')
    if not df.sample_rate.eq(SAMPLE_RATE).all() or not df.num_channels.eq(1).all():
        raise ValueError('Expected 16 kHz mono manifest')
    if not df.audio_subdir.eq('wav_headMic').all():
        raise ValueError('Expected wav_headMic only')
    for spk, (name, label, count) in EXPECTED_SPEAKER_INFO.items():
        part = df[df.speaker == spk]
        if len(part) != count or set(part.severity_label) != {label} or set(part.severity_name) != {name}:
            raise ValueError(f'{spk}: original manifest count/label differs from supplied protocol')
    df['original_path'] = df.path
    df['original_severity_label'] = df.severity_label
    if audio_root is not None:
        def relocate(row):
            parts = Path(row['path']).parts
            indices = [i for i, p in enumerate(parts) if p == row['speaker']]
            if len(indices) != 1: raise ValueError(f'Cannot relocate path: {row["path"]}')
            return str(audio_root.joinpath(*parts[indices[0]:]))
        df['path'] = df.apply(relocate, axis=1)
    if df.path.duplicated().any(): raise ValueError('Relocated paths overlap')
    missing = [p for p in df.path if not Path(p).is_file()]
    if missing: raise FileNotFoundError(f'{len(missing)} missing audio files; first: {missing[0]}')
    mapping = {s: info[1] for s, info in EXPECTED_SPEAKER_INFO.items()}
    policy = 'legacy_supplied_manifest_mapping_not_clinically_validated'
    if label_map is not None:
        mapping = json.loads(label_map.read_text(encoding='utf-8'))
        if set(mapping) != set(EXPECTED_SPEAKER_INFO): raise ValueError('label-map needs all 8 speaker IDs')
        if any(type(v) is not int or v not in range(4) for v in mapping.values()):
            raise ValueError('label-map values must be integer SOURCE output indices 0..3')
        if not label_map_reference: raise ValueError('--label-map-reference is required with --label-map')
        policy = 'user_supplied_source_class_mapping'
    active = sorted(set(mapping.values()))
    if len(active) != 3: raise ValueError('This fixed target ground-truth mapping requires 3 distinct source indices; model output remains full4')
    df['severity_label'] = df.speaker.map(mapping).astype(int)
    df['severity_name'] = df.severity_label.map(dict(enumerate(LABEL_NAMES)))
    info = dict(policy=policy, speaker_to_source_label=mapping, active_source_classes=active,
                reference=label_map_reference, label_map_sha256=sha256_file(label_map) if label_map else None,
                mapping_selected_on_target_results=False)
    return df.reset_index(drop=True), info

def validate_audio_headers(df: pd.DataFrame) -> None:
    for path in df['path']:
        try:
            info = sf.info(path)
        except Exception as exc:
            raise RuntimeError(f'Cannot read target audio header: {path}') from exc
        if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.frames <= 0:
            raise ValueError(
                f'{path}: expected nonempty 16-kHz mono audio; '
                f'got sr={info.samplerate}, channels={info.channels}, frames={info.frames}'
            )

@torch.no_grad()
def infer_logits(df, extractor, encoder, head, device, description):
    generator = torch.Generator().manual_seed(BASE_SEED)
    loader = DataLoader(TORGODataset(df), batch_size=TEST_BATCH, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=AudioCollator(extractor),
        pin_memory=device.type == 'cuda', worker_init_fn=seed_worker, generator=generator)
    logits, seen = [], []
    encoder.eval(); head.eval()
    for batch in tqdm(loader, desc=description):
        values = batch['input_values'].to(device, non_blocking=True)
        lengths = batch['input_lengths'].to(device, non_blocking=True)
        with amp_context(device):
            output = head(encoder(values, lengths))['logits']
        if output.ndim != 2 or output.shape[1] != 4 or not torch.isfinite(output).all():
            raise ValueError('Invalid four-class logits')
        logits.append(output.float().cpu().numpy())
        seen.extend(m['utterance_id'] for m in batch['metadata'])
    if seen != df.utterance_id.tolist(): raise RuntimeError('Inference row identity/order mismatch')
    return np.concatenate(logits, axis=0)

def decision_probabilities(logits, active):
    """Stable FP64 four-class softmax of saved FP32 logits; SWCS tau is never reused."""
    del active
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 4 or not np.isfinite(logits).all():
        raise ValueError('Expected finite [N,4] logits')
    shifted = logits - logits.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probs = np.exp(logp)
    return {'full4': (probs, logp)}

def metrics(y, probs, active, mode, log_probs=None):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, cohen_kappa_score
    y = np.asarray(y, dtype=int)
    probs = np.asarray(probs, dtype=np.float64)
    if mode != 'full4': raise ValueError(f'Unsupported decision mode: {mode}')
    labels = list(range(4))
    if len(y) == 0 or probs.shape != (len(y),4) or not set(y) <= set(active):
        raise ValueError('Invalid metric inputs')
    if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(1),1,atol=1e-6):
        raise ValueError('Invalid probability matrix')
    pred = probs.argmax(1)
    cm = confusion_matrix(y, pred, labels=list(range(4)))
    p,r,f,support = precision_recall_fscore_support(y,pred,labels=labels,zero_division=0)
    _,_,f_target,_ = precision_recall_fscore_support(y,pred,labels=active,zero_division=0)
    target_recalls = [cm[k,k]/cm[k].sum() for k in active if cm[k].sum()]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        qwk = float(cohen_kappa_score(y,pred,labels=labels,weights='quadratic'))
    nll = -np.mean(log_probs[np.arange(len(y)),y]) if log_probs is not None else -np.log(
        np.maximum(probs[np.arange(len(y)),y],np.finfo(np.float64).tiny)).mean()
    out = dict(sample_count=len(y), accuracy=float(np.mean(pred==y)),
        balanced_accuracy_gt_classes=float(np.mean(target_recalls)),
        macro_f1_gt_classes=float(np.mean(f_target)), macro_f1_output_space=float(np.mean(f)),
        mae_source_label_order=float(np.abs(pred-y).mean()), qwk_output_space=qwk,
        negative_log_likelihood=float(nll), out_of_target_prediction_rate=float(np.mean(~np.isin(pred,active))))
    for k in range(4): out[f'pred_count_{k}'] = int((pred==k).sum())
    per_class = pd.DataFrame(dict(source_label=labels, class_name=[LABEL_NAMES[k] for k in labels],
                                  precision=p,recall=r,f1=f,support=support))
    confusion = pd.DataFrame(cm[np.ix_(labels, labels)],
        columns=[f'pred_{k}' for k in labels],index=[f'true_{k}' for k in labels]).rename_axis('true_class').reset_index()
    return out, per_class, confusion

def reports(df, probs, active, mode, folder, identity, log_probs=None):
    folder.mkdir(parents=True, exist_ok=True)
    y = df.severity_label.to_numpy(dtype=int)
    result, classes, cm = metrics(y,probs,active,mode,log_probs)
    prediction = df[['utterance_id','path','speaker','word_key','word_text','original_severity_label']].copy()
    prediction['true_label'] = y
    prediction['pred_label'] = probs.argmax(1)
    prediction['correct'] = prediction.pred_label.to_numpy()==y
    for k in range(4): prediction[f'prob_{k}'] = probs[:,k]
    save_csv(prediction, folder/'utterance_predictions.csv')
    save_csv(classes, folder/'class_metrics.csv')
    save_csv(cm, folder/'confusion_matrix.csv')
    rows, speaker_probs, speaker_true = [], [], []
    for spk in sorted(df.speaker.unique()):
        mask = df.speaker.to_numpy()==spk
        labels = np.unique(y[mask])
        if len(labels)!=1: raise ValueError('Multiple labels for one speaker')
        mean = probs[mask].mean(0)
        speaker_probs.append(mean); speaker_true.append(labels[0])
        row=dict(speaker=spk, true_label=int(labels[0]), pred_label=int(mean.argmax()),
                 utterances=int(mask.sum()), utterance_accuracy=float((probs[mask].argmax(1)==labels[0]).mean()))
        row.update({f'mean_prob_{k}':float(mean[k]) for k in range(4)})
        rows.append(row)
    sm, sc, scm = metrics(np.array(speaker_true),np.array(speaker_probs),active,mode)
    save_csv(pd.DataFrame(rows), folder/'speaker_predictions.csv')
    save_csv(sc, folder/'speaker_class_metrics.csv')
    save_csv(scm, folder/'speaker_confusion_matrix.csv')
    result.update({f'speaker_{k}':v for k,v in sm.items()})
    result.update(identity, decision_mode=mode)
    save_json(result, folder/'metrics.json')
    return result

def summarize(rows, folder):
    table = pd.DataFrame(rows)
    save_csv(table, folder/'all_metrics.csv')
    numbers = list(METRIC_NAMES)+['speaker_'+k for k in METRIC_NAMES]
    deltas=[]
    for (protocol,mode,evaluation,fold), g in table.groupby(['protocol','decision_mode','evaluation','fold']):
        baseline=g[g.method=='ce']
        if len(baseline)!=1: continue
        base=baseline.iloc[0]
        for _,other in g[g.method!='ce'].iterrows():
            for metric in numbers:
                deltas.append(dict(protocol=protocol, decision_mode=mode, evaluation=evaluation,
                    fold=fold,method=other['method'],metric=metric,ce_value=base[metric],
                    swcs_value=other[metric],delta_swcs_minus_ce=other[metric]-base[metric]))
    if deltas: save_csv(pd.DataFrame(deltas), folder/'paired_swcs_minus_ce.csv')
    main=table[table.protocol=='SD']
    save_csv(main, folder/'main_metrics.csv')
    print('\nUTTERANCE-LEVEL MAIN RESULTS | full4 | fractions (not percentages)')
    print(main[['experiment','decision_mode','evaluation','accuracy','balanced_accuracy_gt_classes',
                'macro_f1_gt_classes','macro_f1_output_space','mae_source_label_order',
                'qwk_output_space','negative_log_likelihood','out_of_target_prediction_rate'
                ]].to_string(index=False, float_format=lambda x: f'{x:.6f}'))

def nonempty_path(value: str) -> Path:
    """Reject an empty CLI path before pathlib can convert it to '.'."""
    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError(
            '빈 경로는 사용할 수 없습니다. --train-root "$TRAIN_ROOT"의 변수가 '
            '비어 있는지 확인하십시오. 기본 학습 경로를 쓰려면 --train-root 옵션을 '
            '생략하거나 실제 절대경로를 직접 입력하십시오.'
        )
    try:
        return Path(value).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f'경로를 해석할 수 없습니다: {value!r}: {exc}') from exc

def make_jobs(root):
    jobs = []
    for directory, method in (('baseline', 'ce'), ('csrl', 'swcs')):
        base = root / 'SD' / directory
        candidates = sorted(base.glob('final_epoch_*.pt'))
        if len(candidates) != 1:
            raise RuntimeError(f'{base}: expected exactly one final_epoch_*.pt; found {len(candidates)}')
        asset = base / 'assets/frozen_encoder.pt'
        if not asset.is_file():
            raise FileNotFoundError(asset)
        jobs.append(dict(experiment=f'sd_{method}', protocol='SD', method=method,
                         fold='fold_SD', checkpoint=str(candidates[0]), asset=str(asset)))
    return jobs


def restore(job, device):
    asset = torch.load(job['asset'], map_location='cpu', weights_only=True)
    checkpoint = torch.load(job['checkpoint'], map_location='cpu', weights_only=True)
    encoder = CompletelyFrozenWav2Vec2(asset)
    extractor = Wav2Vec2FeatureExtractor.from_dict(asset['feature_extractor_config'])
    head = SeverityHead()
    head.load_state_dict(checkpoint['model_state_dict'], strict=True)
    for model in (encoder, head):
        model.requires_grad_(False).to(device).eval()
    return encoder, head, extractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-root', type=nonempty_path, default=TRAIN_ROOT)
    parser.add_argument('--manifest', type=nonempty_path, default=MANIFEST)
    parser.add_argument('--output-root', type=nonempty_path)
    parser.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    args = parser.parse_args()
    jobs = make_jobs(args.train_root)
    output = args.output_root or args.train_root / 'TORGO_zero_shot_sd_full4'
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output must be empty: {output}')
    df, mapping = read_manifest(args.manifest)
    validate_audio_headers(df)
    active = mapping['active_source_classes']
    set_seed(BASE_SEED)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    output.mkdir(parents=True, exist_ok=True)
    save_csv(df, output / 'evaluated_manifest.csv')
    save_csv(pd.DataFrame(jobs), output / 'source_checkpoints.csv')
    config = dict(status='running', seed=BASE_SEED, batch_size=TEST_BATCH,
                  models=jobs, manifest=str(args.manifest), mapping=mapping,
                  target_training=False, target_calibration=False, target_model_selection=False,
                  output_classes=4, feature_cache=False,
                  ensemble=False)
    save_json(config, output / 'run_config.json')
    rows = []
    for experiment in dict.fromkeys(j['experiment'] for j in jobs):
        selected = [j for j in jobs if j['experiment'] == experiment]
        for job in selected:
            encoder, head, extractor = restore(job, device)
            logits = infer_logits(df, extractor, encoder, head, device, experiment + '/' + job['fold'])
            folder = output / experiment / job['fold']
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / 'logits4.npy', logits, allow_pickle=False)
            probs, logs = decision_probabilities(logits, active)['full4']
            identity = {k:job[k] for k in ('experiment','protocol','method','fold')}
            identity.update(evaluation='individual', model_count=1)
            rows.append(reports(df, probs, active, 'full4', folder / 'full4', identity, logs))
            del encoder, head, extractor, logits
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        summarize(rows, output)
    config.update(status='completed', completed_models=len(jobs))
    save_json(config, output / 'run_config.json')
    print(f'Results: {output}')

if __name__ == '__main__':
    main()
