from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2FeatureExtractor

PROTOCOL = 'LOSO'
METHOD = 'swcs'
EPOCHS = 50
SWC_LAMBDA = 0.05
SWC_TEMPERATURE = 0.15
METHOD_EPOCHS = {METHOD: EPOCHS}
AUDIO_MODEL_NAME = 'facebook/wav2vec2-base'
LABEL_NAMES = ['very_low', 'low', 'medium', 'high']
NUM_CLASSES = 4
SAMPLE_RATE = 16000
WAV2VEC_DIM = HEAD_DIM = 768
BASE_SEED = 42
HEAD_LR = 5e-4
BETAS = (0.9, 0.98)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
USE_AMP = True
NUM_WORKERS = 0
METHOD_TRAIN_BATCH = {'ce_only': 16, 'swcs': 8}
METHOD_EVAL_BATCH = {'ce_only': 16, 'swcs': 8}
METHOD_GRAD_ACCUM = {'ce_only': 1, 'swcs': 2}
SWC_SPEAKERS_PER_CLASS = 2
SWC_GROUP_SIZE = 8
SWC_WORDS_PER_UPDATE = 2
UTTERANCES_PER_UPDATE = 16
JOINT_BATCH_SEED_OFFSET = 27271627
EXPECTED_TRAIN_WORDS = 155
SPEAKER_LABELS = {'F02': 2, 'F03': 3, 'F04': 1, 'F05': 0, 'M01': 3, 'M04': 3, 'M05': 1, 'M07': 2, 'M08': 0, 'M09': 0, 'M10': 0, 'M11': 1, 'M12': 3, 'M14': 0, 'M16': 2}


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
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def amp_context(device: torch.device):
    if USE_AMP and device.type == 'cuda':
        return torch.autocast('cuda')
    return nullcontext()


def make_scaler(device: torch.device):
    if not (USE_AMP and device.type == 'cuda'):
        return None
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        try:
            return torch.amp.GradScaler('cuda')
        except TypeError:
            return torch.amp.GradScaler()
    return torch.cuda.amp.GradScaler()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8-sig')
    tmp.replace(path)


def atomic_torch_save(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(dict(obj), tmp)
    tmp.replace(path)


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def read_split(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = ['utterance_id', 'path', 'speaker', 'severity_name', 'severity_label']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f'{path}: missing columns {missing}')
    for col in ['utterance_id', 'path', 'speaker', 'severity_name']:
        df[col] = df[col].astype(str).str.strip()
    df['speaker'] = df['speaker'].str.upper()
    df['severity_name'] = df['severity_name'].str.lower()
    df['severity_label'] = pd.to_numeric(df['severity_label'], errors='raise').astype(int)
    invalid_labels = sorted(set(df['severity_label'].unique()) - set(range(NUM_CLASSES)))
    if invalid_labels:
        raise RuntimeError(f'{path}: invalid severity labels {invalid_labels}')
    if 'word_key' in df.columns:
        identity_col = 'word_key'
    elif 'word_code' in df.columns:
        identity_col = 'word_code'
    elif 'word_text' in df.columns:
        identity_col = 'word_text'
    else:
        raise RuntimeError(f'{path}: no word_key, word_code or word_text column')
    df['word_key'] = df[identity_col].astype(str).str.strip()
    if 'word_text' in df.columns:
        text = df['word_text'].astype(str).str.strip()
        df['text_word'] = np.where(text.ne(''), text, df['word_key'])
    else:
        df['text_word'] = df['word_key']
    if df['word_key'].eq('').any():
        raise RuntimeError(f'{path}: empty word identity')
    if df['utterance_id'].duplicated().any():
        raise RuntimeError(f'{path}: duplicated utterance_id')
    if df['path'].duplicated().any():
        raise RuntimeError(f'{path}: duplicated audio path')
    speaker_mapping = df[['speaker', 'severity_label']].drop_duplicates().groupby('speaker')['severity_label'].nunique()
    if (speaker_mapping != 1).any():
        raise RuntimeError(f'{path}: a speaker maps to multiple severity labels')
    missing_audio = [p for p in df['path'] if not Path(p).is_file()]
    if missing_audio:
        raise FileNotFoundError(f'{path}: {len(missing_audio)} missing audio files; first={missing_audio[0]}')
    return df.reset_index(drop=True)


def load_audio(path: str) -> np.ndarray:
    waveform, sr = sf.read(path, dtype='float32', always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1:
        raise RuntimeError(f'{path}: invalid waveform shape {waveform.shape}')
    if sr != SAMPLE_RATE:
        raise RuntimeError(f'{path}: sample rate={sr}, expected={SAMPLE_RATE}')
    if waveform.size == 0:
        raise RuntimeError(f'{path}: empty waveform')
    if not np.isfinite(waveform).all():
        raise RuntimeError(f'{path}: waveform contains NaN/Inf')
    return waveform.astype(np.float32, copy=False)


class WaveformDataset(Dataset):

    def __init__(self, csv_path: Path):
        self.df = read_split(csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[int(index)]
        return {'dataset_index': int(index), 'utterance_id': str(row['utterance_id']), 'path': str(row['path']), 'speaker': str(row['speaker']), 'severity_label': int(row['severity_label']), 'severity_name': str(row['severity_name']), 'word_key': str(row['word_key']), 'text_word': str(row['text_word']), 'waveform': load_audio(str(row['path']))}


class AudioCollator:

    def __init__(self, extractor: AutoFeatureExtractor):
        self.extractor = extractor

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        waveforms = [item['waveform'] for item in batch]
        lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
        encoded = self.extractor(waveforms, sampling_rate=SAMPLE_RATE, padding=True, return_attention_mask=False, return_tensors='pt')
        metadata = [{k: v for k, v in item.items() if k != 'waveform'} for item in batch]
        return {'input_values': encoded['input_values'], 'input_lengths': lengths, 'metadata': metadata}


class CompletelyFrozenWav2Vec2(nn.Module):

    def __init__(self):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(AUDIO_MODEL_NAME)
        if self.wav2vec2.config.feat_extract_norm != 'group':
            raise RuntimeError('Expected group-norm Wav2Vec2; backbone attention_mask remains None.')
        if hasattr(self.wav2vec2, 'gradient_checkpointing_disable'):
            self.wav2vec2.gradient_checkpointing_disable()
        if hasattr(self.wav2vec2.encoder, 'gradient_checkpointing'):
            self.wav2vec2.encoder.gradient_checkpointing = False
        if hasattr(self.wav2vec2.config, 'gradient_checkpointing'):
            self.wav2vec2.config.gradient_checkpointing = False
        for parameter in self.wav2vec2.parameters():
            parameter.requires_grad = False
        feature_encoder = self.wav2vec2.feature_extractor
        if hasattr(feature_encoder, '_freeze_parameters'):
            feature_encoder._freeze_parameters()
        else:
            for parameter in feature_encoder.parameters():
                parameter.requires_grad = False
            if hasattr(feature_encoder, '_requires_grad'):
                feature_encoder._requires_grad = False
        for parameter in self.wav2vec2.feature_projection.parameters():
            parameter.requires_grad = False
        self.eval()
        self.validate_frozen()

    def validate_frozen(self) -> None:
        trainable = [name for name, parameter in self.wav2vec2.named_parameters() if parameter.requires_grad]
        if trainable:
            raise RuntimeError(f'Wav2Vec2 is not completely frozen. First trainable parameters: {trainable[:10]}')

    def train(self, mode: bool=True):
        super().train(False)
        self.wav2vec2.eval()
        return self

    def eval(self):
        return self.train(False)

    def trainable_parameter_count(self) -> int:
        return int(sum((p.numel() for p in self.wav2vec2.parameters() if p.requires_grad)))

    @torch.no_grad()
    def forward(self, input_values: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        hidden = self.wav2vec2(input_values=input_values, attention_mask=None, return_dict=True).last_hidden_state
        output_lengths = self.wav2vec2._get_feat_extract_output_lengths(input_lengths).to(hidden.device, dtype=torch.long).clamp(min=1, max=hidden.size(1))
        frame_ids = torch.arange(hidden.size(1), device=hidden.device)[None, :]
        valid = frame_ids < output_lengths[:, None]
        valid_float = valid.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * valid_float).sum(dim=1)
        pooled = pooled / output_lengths.to(hidden.dtype).unsqueeze(1)
        if pooled.shape[-1] != WAV2VEC_DIM:
            raise RuntimeError(f'Wav2Vec2 pooled dim={pooled.shape[-1]}, expected={WAV2VEC_DIM}')
        return pooled


class SeverityHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(WAV2VEC_DIM, HEAD_DIM), nn.ReLU())
        self.classifier = nn.Linear(HEAD_DIM, NUM_CLASSES)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.head(h)
        logits = self.classifier(z)
        return {'z': z, 'logits': logits}


def build_optimizer(encoder: CompletelyFrozenWav2Vec2, model: SeverityHead) -> torch.optim.Optimizer:
    encoder.validate_frozen()
    head_params = [p for p in model.parameters() if p.requires_grad]
    if not head_params:
        raise RuntimeError('No trainable head/classifier parameters')
    return torch.optim.Adam(head_params, lr=HEAD_LR, betas=BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)


class SameWordStructuredBatch8Sampler:

    def __init__(self, dataset: WaveformDataset):
        self.dataset = dataset
        self.df = dataset.df.reset_index(drop=True)
        self.pool: Dict[str, Dict[int, Dict[str, List[int]]]] = {}
        for idx, row in self.df.iterrows():
            word = str(row['word_key'])
            severity = int(row['severity_label'])
            speaker = str(row['speaker'])
            self.pool.setdefault(word, {})
            self.pool[word].setdefault(severity, {})
            self.pool[word][severity].setdefault(speaker, [])
            self.pool[word][severity][speaker].append(int(idx))
        self.all_words = sorted(self.pool.keys())
        if len(self.all_words) != EXPECTED_TRAIN_WORDS:
            raise RuntimeError(f'same-word sampler words={len(self.all_words)}, expected={EXPECTED_TRAIN_WORDS}')
        for word in self.all_words:
            for severity in range(NUM_CLASSES):
                speakers = sorted(self.pool.get(word, {}).get(severity, {}).keys())
                if len(speakers) < SWC_SPEAKERS_PER_CLASS:
                    raise RuntimeError(f'word={word}, severity={severity}: only {len(speakers)} speakers; need {SWC_SPEAKERS_PER_CLASS}')

    def sample_batch(self, rng: np.random.Generator, word: Optional[str]=None) -> Tuple[List[int], str, Dict[int, List[str]]]:
        if word is None:
            word = str(self.all_words[int(rng.integers(0, len(self.all_words)))])
        else:
            word = str(word)
            if word not in self.pool:
                raise RuntimeError(f'Requested same-word Batch8 for unknown word={word}')
        indices: List[int] = []
        selected_speakers: Dict[int, List[str]] = {}
        for severity in range(NUM_CLASSES):
            speaker_pool = sorted(self.pool[word][severity].keys())
            chosen = rng.choice(speaker_pool, size=SWC_SPEAKERS_PER_CLASS, replace=False).tolist()
            chosen = [str(x) for x in chosen]
            selected_speakers[severity] = chosen
            for speaker in chosen:
                utterance_pool = self.pool[word][severity][speaker]
                pick = int(utterance_pool[int(rng.integers(0, len(utterance_pool)))])
                indices.append(pick)
        if len(indices) != 8:
            raise RuntimeError(f'same-word Batch8 size={len(indices)}, expected=8')
        if len(set(indices)) != 8:
            raise RuntimeError('same-word Batch8 contains duplicate utterance indices')
        sampled = self.df.iloc[indices].reset_index(drop=True)
        if sampled['word_key'].nunique() != 1:
            raise RuntimeError('same-word Batch8 contains multiple words')
        if str(sampled['word_key'].iloc[0]) != word:
            raise RuntimeError('same-word Batch8 word mismatch')
        expected_labels = [0, 0, 1, 1, 2, 2, 3, 3]
        actual_labels = sampled['severity_label'].astype(int).tolist()
        if actual_labels != expected_labels:
            raise RuntimeError(f'same-word Batch8 label order={actual_labels}, expected={expected_labels}')
        for severity in range(NUM_CLASSES):
            start = severity * SWC_SPEAKERS_PER_CLASS
            stop = start + SWC_SPEAKERS_PER_CLASS
            block = sampled.iloc[start:stop]
            if block['speaker'].nunique() != 2:
                raise RuntimeError(f'severity={severity}: need two different speakers')
        return (indices, word, selected_speakers)


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], detailed: bool=False) -> Dict[str, Any]:
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.size == 0:
        raise RuntimeError('metric input is empty')
    labels = list(range(NUM_CLASSES))
    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, labels=labels, average='macro', zero_division=0)
    try:
        qwk = float(cohen_kappa_score(true, pred, weights='quadratic'))
    except Exception:
        qwk = math.nan
    result: Dict[str, Any] = {'accuracy': float(accuracy_score(true, pred)), 'balanced_accuracy': float(balanced_accuracy_score(true, pred)), 'macro_precision': float(precision), 'macro_recall': float(recall), 'macro_f1': float(f1), 'mean_absolute_class_error': float(np.mean(np.abs(true - pred))), 'within_one_class_accuracy': float(np.mean(np.abs(true - pred) <= 1)), 'quadratic_weighted_kappa': qwk}
    if detailed:
        result['confusion_matrix'] = confusion_matrix(true, pred, labels=labels).tolist()
    return result


def optimizer_update(modules: Sequence[nn.Module], optimizer: torch.optim.Optimizer, scaler) -> Dict[str, Any]:
    params = [p for module in modules for p in module.parameters() if p.requires_grad]
    if scaler is not None:
        scaler.unscale_(optimizer)
    finite = True
    for p in params:
        if p.grad is not None and (not torch.isfinite(p.grad).all()):
            finite = False
            break
    if finite:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM, error_if_nonfinite=False)
        grad_norm = float(grad_norm_tensor.detach().float().item())
    else:
        grad_norm = math.inf
    clipped = bool(math.isfinite(grad_norm) and grad_norm > MAX_GRAD_NORM)
    if scaler is not None:
        old_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        new_scale = float(scaler.get_scale())
        successful = bool(finite and math.isfinite(grad_norm) and (new_scale >= old_scale))
        amp_scale = new_scale
    else:
        successful = bool(finite and math.isfinite(grad_norm))
        if successful:
            optimizer.step()
        amp_scale = 1.0
    optimizer.zero_grad(set_to_none=True)
    return {'successful': successful, 'grad_norm': grad_norm, 'clipped': clipped, 'amp_scale': amp_scale}


@torch.no_grad()
def evaluate_test_from_waveforms(model: SeverityHead, encoder: CompletelyFrozenWav2Vec2, extractor: AutoFeatureExtractor, test_csv: Path, device: torch.device, batch_size: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
    if batch_size not in set(METHOD_EVAL_BATCH.values()):
        raise ValueError(f'이 실행의 SD 평가 배치는 16(CE) 또는 8(SWCS)입니다: {batch_size}')
    model.eval()
    encoder.eval()
    dataset = WaveformDataset(test_csv)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, collate_fn=AudioCollator(extractor), pin_memory=device.type == 'cuda', worker_init_fn=seed_worker)
    y_true: List[int] = []
    y_pred: List[int] = []
    rows: List[Dict[str, Any]] = []
    ce_sum = 0.0
    for batch in tqdm(loader, desc='SOURCE TEST', leave=False):
        values = batch['input_values'].to(device, non_blocking=True)
        lengths = batch['input_lengths'].to(device, non_blocking=True)
        with amp_context(device):
            h = encoder(values, lengths)
            output = model(h)
            labels = torch.tensor([int(m['severity_label']) for m in batch['metadata']], dtype=torch.long, device=device)
            ce = F.cross_entropy(output['logits'], labels)
        logits = output['logits'].float()
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        ce_sum += float(ce.item()) * labels.numel()
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        for i, meta in enumerate(batch['metadata']):
            true_label = int(labels[i].item())
            pred_label = int(pred[i].item())
            row: Dict[str, Any] = {'utterance_id': str(meta['utterance_id']), 'path': str(meta['path']), 'speaker': str(meta['speaker']), 'word_key': str(meta['word_key']), 'text_word': str(meta['text_word']), 'true_label': true_label, 'true_name': LABEL_NAMES[true_label], 'pred_label': pred_label, 'pred_name': LABEL_NAMES[pred_label], 'correct': int(true_label == pred_label)}
            for class_index, class_name in enumerate(LABEL_NAMES):
                row[f'prob_{class_name}'] = float(probs[i, class_index].item())
            rows.append(row)
        del values, lengths, h, output, labels, ce, logits, probs, pred, batch
    metrics = classification_metrics(y_true, y_pred, detailed=False)
    metrics.update({'split': 'test', 'sample_count': int(len(y_true)), 'ce_loss': float(ce_sum / len(y_true)), 'prediction_counts': np.bincount(np.asarray(y_pred, dtype=np.int64), minlength=NUM_CLASSES).astype(int).tolist()})
    return (metrics, pd.DataFrame(rows))


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value


def save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(json_safe(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def cpu_copy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple((cpu_copy(v) for v in value))
    return copy.deepcopy(value)


def tensor_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(key.encode('utf-8'))
        h.update(str(t.dtype).encode('ascii'))
        h.update(str(tuple(t.shape)).encode('ascii'))
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def capture_rng() -> Dict[str, Any]:
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        if len(state['torch_cuda']) != torch.cuda.device_count():
            raise RuntimeError('재개 시 가시 CUDA 장치 수가 기존 실행과 다릅니다.')
        torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])


@contextmanager
def isolated_evaluation_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


@contextmanager
def output_lock(root: Path):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.run.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'다른 프로세스가 이 출력 폴더를 사용 중입니다: {root}') from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def updates_for_epoch(dataset_rows: int) -> int:
    if dataset_rows <= 0:
        raise ValueError('TRAIN이 비어 있습니다.')
    return math.ceil(dataset_rows / 8)


def sample_two_episodes(sampler: SameWordStructuredBatch8Sampler, rng: np.random.Generator) -> Tuple[List[int], List[str]]:
    words = [str(w) for w in rng.choice(sampler.all_words, size=2, replace=False).tolist()]
    if len(set(words)) != 2:
        raise RuntimeError('두 에피소드는 서로 다른 단어여야 합니다.')
    indices = []
    for word in words:
        group, sampled_word, _ = sampler.sample_batch(rng, word=word)
        if sampled_word != word or len(group) != 8:
            raise RuntimeError('에피소드 샘플링 결과가 요청과 다릅니다.')
        indices.extend(group)
    if len(indices) != 16 or len(set(indices)) != 16:
        raise RuntimeError('두 에피소드의 합계는 중복되지 않는 16개 발화여야 합니다.')
    return (indices, words)


def validate_episode8(metadata: Sequence[Mapping[str, Any]], word: str) -> None:
    if len(metadata) != SWC_GROUP_SIZE:
        raise RuntimeError('물리 학습 배치는 한 단어의 8발화여야 합니다.')
    if {str(m['word_key']) for m in metadata} != {str(word)}:
        raise RuntimeError('한 에피소드에 다른 단어가 섞였습니다.')
    if [int(m['severity_label']) for m in metadata] != [0, 0, 1, 1, 2, 2, 3, 3]:
        raise RuntimeError('에피소드의 중증도 순서가 잘못되었습니다.')
    for c in range(NUM_CLASSES):
        if len({str(m['speaker']) for m in metadata[2 * c:2 * c + 2]}) != SWC_SPEAKERS_PER_CLASS:
            raise RuntimeError('동일 중증도 positive는 서로 다른 화자여야 합니다.')
    for key in ('utterance_id', 'path'):
        if len({str(m[key]) for m in metadata}) != SWC_GROUP_SIZE:
            raise RuntimeError(f'한 에피소드에 중복 {key}가 있습니다.')


def validate_two_episode_update(metadata: Sequence[Mapping[str, Any]], words: Sequence[str]) -> None:
    if len(metadata) != UTTERANCES_PER_UPDATE or len(words) != SWC_WORDS_PER_UPDATE or len(set(words)) != SWC_WORDS_PER_UPDATE:
        raise RuntimeError('한 업데이트에는 서로 다른 두 단어의 8발화 에피소드 두 개가 필요합니다.')
    for ep, word in enumerate(words):
        validate_episode8(metadata[SWC_GROUP_SIZE * ep:SWC_GROUP_SIZE * (ep + 1)], word)
    for key in ('utterance_id', 'path'):
        if len({str(m[key]) for m in metadata}) != UTTERANCES_PER_UPDATE:
            raise RuntimeError(f'두 에피소드에 중복 {key}가 있습니다.')


def load_asset_encoder(asset: Mapping[str, Any], device: torch.device):
    config = Wav2Vec2Config.from_dict(asset['encoder_config'])
    if asset.get('attention_implementation'):
        config._attn_implementation = asset['attention_implementation']
    encoder = CompletelyFrozenWav2Vec2.__new__(CompletelyFrozenWav2Vec2)
    nn.Module.__init__(encoder)
    encoder.wav2vec2 = Wav2Vec2Model(config)
    encoder.load_state_dict(asset['encoder_state_dict'], strict=True)
    if hasattr(encoder.wav2vec2, 'gradient_checkpointing_disable'):
        encoder.wav2vec2.gradient_checkpointing_disable()
    if hasattr(encoder.wav2vec2.feature_extractor, '_freeze_parameters'):
        encoder.wav2vec2.feature_extractor._freeze_parameters()
    encoder.requires_grad_(False)
    encoder.to(device).eval()
    encoder.validate_frozen()
    extractor = Wav2Vec2FeatureExtractor.from_dict(asset['feature_extractor_config'])
    return (encoder, extractor)


def sameword_allseverity_contrastive_loss(z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    if z.ndim != 2:
        raise RuntimeError(f'same-word z must be 2D, got {tuple(z.shape)}')
    if z.size(0) != SWC_GROUP_SIZE:
        raise RuntimeError(f'same-word z batch={z.size(0)}, expected={SWC_GROUP_SIZE}')
    zn = F.normalize(z.float(), p=2, dim=-1, eps=1e-08)
    similarity = zn @ zn.T
    labels = torch.arange(NUM_CLASSES, device=z.device).repeat_interleave(SWC_SPEAKERS_PER_CLASS)
    anchor_losses: List[torch.Tensor] = []
    anchor_correct: List[torch.Tensor] = []
    positive_sims: List[torch.Tensor] = []
    negative_sims: List[torch.Tensor] = []
    class_correct: Dict[int, List[torch.Tensor]] = {severity: [] for severity in range(NUM_CLASSES)}
    for anchor in range(SWC_GROUP_SIZE):
        severity = int(labels[anchor].item())
        class_start = severity * SWC_SPEAKERS_PER_CLASS
        class_stop = class_start + SWC_SPEAKERS_PER_CLASS
        positive_candidates = [idx for idx in range(class_start, class_stop) if idx != anchor]
        if len(positive_candidates) != 1:
            raise RuntimeError(f'anchor={anchor}: expected exactly one same-severity positive')
        positive = positive_candidates[0]
        negatives = [idx for idx in range(SWC_GROUP_SIZE) if int(labels[idx].item()) != severity]
        if len(negatives) != 6:
            raise RuntimeError(f'anchor={anchor}: negatives={len(negatives)}, expected=6')
        sims = torch.stack([similarity[anchor, positive]] + [similarity[anchor, idx] for idx in negatives])
        logits = sims / SWC_TEMPERATURE
        target = torch.zeros(1, dtype=torch.long, device=z.device)
        anchor_losses.append(F.cross_entropy(logits.unsqueeze(0), target))
        correct = (logits.argmax() == 0).float()
        anchor_correct.append(correct)
        class_correct[severity].append(correct)
        positive_sims.append(sims[0])
        negative_sims.extend(list(sims[1:]))
    loss = torch.stack(anchor_losses).mean()
    with torch.no_grad():
        pos_mean = torch.stack(positive_sims).mean()
        neg_mean = torch.stack(negative_sims).mean()
        retrieval_acc = torch.stack(anchor_correct).mean()
        diagnostics = {'positive_cosine_mean': float(pos_mean.item()), 'negative_cosine_mean': float(neg_mean.item()), 'contrastive_cosine_gap': float((pos_mean - neg_mean).item()), 'contrastive_retrieval_acc': float(retrieval_acc.item()), 'very_low_retrieval_acc': float(torch.stack(class_correct[0]).mean().item()), 'low_retrieval_acc': float(torch.stack(class_correct[1]).mean().item()), 'medium_retrieval_acc': float(torch.stack(class_correct[2]).mean().item()), 'high_retrieval_acc': float(torch.stack(class_correct[3]).mean().item())}
    return (loss, diagnostics)


def episode8_loss(output: Mapping[str, torch.Tensor], labels: torch.Tensor, method: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, float]]]:
    if method not in METHOD_EPOCHS:
        raise ValueError(f'지원하지 않는 방법: {method}')
    logits = output['logits']
    if tuple(logits.shape) != (SWC_GROUP_SIZE, NUM_CLASSES):
        raise RuntimeError(f'에피소드 logits는 (8,4)여야 합니다: {tuple(logits.shape)}')
    if labels.detach().cpu().tolist() != [0, 0, 1, 1, 2, 2, 3, 3]:
        raise RuntimeError('에피소드 label 순서가 SWCS 입력 규칙과 다릅니다.')
    ce = F.cross_entropy(logits, labels, reduction='mean')
    if method == 'ce_only':
        return (ce, ce, ce.detach().new_zeros(()), [])
    z = output['z']
    if z.ndim != 2 or z.shape[0] != SWC_GROUP_SIZE:
        raise RuntimeError('한 에피소드의 post-ReLU z는 8발화여야 합니다.')
    swcs, diagnostics = sameword_allseverity_contrastive_loss(z)
    return (ce + SWC_LAMBDA * swcs, ce, swcs, [diagnostics])


def train_one_epoch_swcs_batch8(encoder, model, dataset, extractor, optimizer, scaler, device: torch.device, epoch: int, method: str, sampler: SameWordStructuredBatch8Sampler) -> Dict[str, Any]:
    if method not in METHOD_EPOCHS or not 1 <= epoch <= METHOD_EPOCHS[method]:
        raise ValueError('고정 방법/에폭 범위 오류')
    if method != 'swcs' or METHOD_TRAIN_BATCH[method] != 8 or SWC_GROUP_SIZE != 8 or (SWC_WORDS_PER_UPDATE != 2) or (METHOD_GRAD_ACCUM[method] != 2) or (UTTERANCES_PER_UPDATE != 16):
        raise RuntimeError('이 실행은 물리 Batch8 x 두 에피소드 / optimizer 1회로 고정됩니다.')
    encoder.eval()
    encoder.validate_frozen()
    model.train()
    collator = AudioCollator(extractor)
    seed = BASE_SEED + int(epoch) * 10007 + JOINT_BATCH_SEED_OFFSET
    rng = np.random.default_rng(seed)
    updates = updates_for_epoch(len(dataset))
    optimizer.zero_grad(set_to_none=True)
    ys, ps, grad_norms, diag_rows, first_pairs = ([], [], [], [], [])
    ce_sum = swcs_sum = total_sum = 0.0
    successful = clipped = forward_calls = backward_calls = 0
    sample_hash, id_hash = (hashlib.sha256(), hashlib.sha256())
    scale = float(scaler.get_scale()) if scaler is not None else 1.0
    start = time.perf_counter()
    for u in tqdm(range(updates), desc=f'{method} E{epoch:02d} / Batch8 x2', leave=False):
        indices, words = sample_two_episodes(sampler, rng)
        sample_hash.update(np.asarray(indices, dtype='<i8').tobytes())
        validate_two_episode_update(sampler.df.iloc[indices].to_dict('records'), words)
        if u < 8:
            first_pairs.append(words)
        for episode, word in enumerate(words):
            lo = SWC_GROUP_SIZE * episode
            group_indices = indices[lo:lo + SWC_GROUP_SIZE]
            batch = collator([dataset[i] for i in group_indices])
            validate_episode8(batch['metadata'], word)
            for item in batch['metadata']:
                id_hash.update((str(item['utterance_id']) + '\n').encode('utf-8'))
            values = batch['input_values'].to(device, non_blocking=True)
            lengths = batch['input_lengths'].to(device, non_blocking=True)
            if values.ndim != 2 or values.shape[0] != METHOD_TRAIN_BATCH[method]:
                raise RuntimeError('backbone에 입력하는 물리 배치는 8발화로 고정됩니다.')
            labels = torch.tensor([int(x['severity_label']) for x in batch['metadata']], device=device, dtype=torch.long)
            with amp_context(device):
                output = model(encoder(values, lengths))
                total, ce, swcs, diag = episode8_loss(output, labels, method)
                mean_contribution = total.float() / METHOD_GRAD_ACCUM[method]
            forward_calls += 1
            if not torch.isfinite(mean_contribution.detach()):
                raise FloatingPointError(f'{method} E{epoch} update{u} episode{episode}: 유한하지 않은 손실')
            if scaler is None:
                mean_contribution.backward()
            else:
                scaler.scale(mean_contribution).backward()
            backward_calls += 1
            with torch.no_grad():
                ys.extend(labels.cpu().tolist())
                ps.extend(output['logits'].argmax(-1).cpu().tolist())
                ce_sum += float(ce.item()) / METHOD_GRAD_ACCUM[method]
                swcs_sum += float(swcs.item()) / METHOD_GRAD_ACCUM[method]
                total_sum += float(total.item()) / METHOD_GRAD_ACCUM[method]
                diag_rows.extend(diag)
            del batch, values, lengths, labels, output, total, ce, swcs, diag, mean_contribution
        update = optimizer_update((encoder, model), optimizer, scaler)
        successful += int(update['successful'])
        clipped += int(update['clipped'])
        if math.isfinite(update['grad_norm']):
            grad_norms.append(float(update['grad_norm']))
        scale = float(update['amp_scale'])
    expected_calls = updates * SWC_WORDS_PER_UPDATE
    if len(ys) != updates * UTTERANCES_PER_UPDATE or forward_calls != expected_calls or backward_calls != expected_calls or (not successful):
        raise RuntimeError('학습 발화 노출/호출 수 오류 또는 모든 업데이트 실패')
    result = classification_metrics(ys, ps)
    result.update({'method': method, 'epoch': epoch, 'sampler_rng_seed': seed, 'dataset_rows': len(dataset), 'physical_train_batch': METHOD_TRAIN_BATCH[method], 'gradient_accumulation_steps': METHOD_GRAD_ACCUM[method], 'episodes_per_update': SWC_WORDS_PER_UPDATE, 'utterances_per_update': UTTERANCES_PER_UPDATE, 'physical_forward_calls': forward_calls, 'backward_calls': backward_calls, 'padding_scope': 'one_8_utterance_episode', 'attempted_optimizer_updates': updates, 'successful_optimizer_updates': successful, 'skipped_optimizer_updates': updates - successful, 'sampled_utterance_exposures': len(ys), 'sample_sequence_sha256': sample_hash.hexdigest(), 'sample_utterance_id_sequence_sha256': id_hash.hexdigest(), 'sampled_word_pairs_first8': first_pairs, 'ce_loss': ce_sum / updates, 'swc_loss': swcs_sum / updates if method == 'swcs' else None, 'total_loss': total_sum / updates, 'swc_lambda': SWC_LAMBDA if method == 'swcs' else 0.0, 'swc_temperature': SWC_TEMPERATURE if method == 'swcs' else None, 'swc_group_count': len(diag_rows), 'clipping_updates': clipped, 'grad_norm_mean_preclip': float(np.mean(grad_norms)) if grad_norms else None, 'amp_scale_end': scale, 'train_elapsed_seconds': time.perf_counter() - start})
    if diag_rows:
        result.update({k: float(np.mean([d[k] for d in diag_rows])) for k in diag_rows[0]})
    return result


TRAIN_FUNCTION = train_one_epoch_swcs_batch8


def validate_training_splits(train, test, heldout):
    expected_train_speakers = set(SPEAKER_LABELS) - ({heldout} if heldout else set())
    expected_test_speakers = {heldout} if heldout else set(SPEAKER_LABELS)
    common = {f'D{i}' for i in range(10)} | {f'L{chr(i)}' for i in range(65, 91)} | {f'C{i}' for i in range(1, 20)} | {f'CW{i}' for i in range(1, 101)}
    required = {'block', 'word_key', 'word_code_raw', 'audio_version', 'microphone', 'is_common', 'is_uncommon'}
    for frame, speakers, is_train in [(train, expected_train_speakers, True), (test, expected_test_speakers, False)]:
        if not required <= set(frame.columns):
            raise ValueError(f'Missing split columns: {sorted(required - set(frame.columns))}')
        if set(frame['speaker']) != speakers:
            raise ValueError('Unexpected split speaker set')
        keys = [(r.speaker, r.block, r.word_key) for r in frame.itertuples()]
        if is_train:
            expected = {(s, f'B{b}', w) for s in speakers for b in (1, 2, 3) for w in common}
        else:
            expected = {(s, f'B{b}', f'B{b}_UW{i}') for s in speakers for b in (1, 2, 3) for i in range(1, 101)}
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError('Missing, duplicated or unexpected speaker/block/word identity')
        for row in frame.itertuples():
            label = SPEAKER_LABELS[row.speaker]
            if row.severity_label != label or row.severity_name != LABEL_NAMES[label]:
                raise ValueError(f'Invalid label: {row.utterance_id}')
            if row.audio_version != 'original' or row.microphone != 'M5':
                raise ValueError('Training splits must use original/M5')
            if int(row.is_common) != int(is_train) or int(row.is_uncommon) != int(not is_train):
                raise ValueError('Invalid common/uncommon assignment')
            expected_word = row.word_key if is_train else row.word_key.split('_', 1)[1]
            if row.word_code_raw != expected_word:
                raise ValueError('Invalid word identity')
    for field in ['utterance_id', 'path', 'word_key']:
        if set(train[field]) & set(test[field]):
            raise ValueError(f'Train/test overlap: {field}')
    if heldout and set(train['speaker']) & set(test['speaker']):
        raise ValueError('LOSO speaker overlap')
    return {'train_rows': len(train), 'test_rows': len(test), 'train_speakers': len(expected_train_speakers), 'test_speakers': len(expected_test_speakers), 'train_words': 155, 'test_words': 300}


def audio_stat_hash(frame):
    h = hashlib.sha256()
    for row in frame.itertuples():
        path = Path(row.path)
        stat = path.stat()
        h.update(f'{path.resolve()}\t{stat.st_size}\t{stat.st_mtime_ns}\n'.encode())
    return h.hexdigest()


def prepare_frozen_backbone(root, device):
    path = root / 'assets' / 'frozen_encoder.pt'
    manifest_path = root / 'assets' / 'backbone.json'
    if path.exists():
        if not manifest_path.exists():
            raise RuntimeError('Incomplete backbone asset; use a new output directory')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['file_sha256'] != sha256_file(path):
            raise RuntimeError('Saved backbone asset changed')
        asset = torch_load(path, torch.device('cpu'))
        encoder, extractor = load_asset_encoder(asset, device)
    else:
        if manifest_path.exists() or any(root.rglob('last_state.pt')) or any(root.rglob('final_epoch_*.pt')):
            raise RuntimeError('Backbone asset missing for an existing run')
        set_seed(BASE_SEED)
        extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
        encoder = CompletelyFrozenWav2Vec2().to(device).eval()
        asset = {'audio_model_name': AUDIO_MODEL_NAME, 'encoder_state_dict': cpu_copy(encoder.state_dict()), 'encoder_config': encoder.wav2vec2.config.to_dict(), 'feature_extractor_config': extractor.to_dict(), 'attention_implementation': getattr(encoder.wav2vec2.config, '_attn_implementation', None)}
        atomic_torch_save(asset, path)
        manifest = {'audio_model_name': AUDIO_MODEL_NAME, 'file': str(path.relative_to(root)), 'file_sha256': sha256_file(path)}
        save_json(manifest, manifest_path)
    del asset
    return encoder, extractor, manifest


def training_config(source, dataset, test, heldout, args, device, backbone, initial_hash):
    return {'format': 'uaspeech_fixed_training_v1', 'protocol': PROTOCOL, 'method': METHOD, 'seed': BASE_SEED, 'epochs': EPOCHS, 'lambda': SWC_LAMBDA if METHOD == 'swcs' else 0.0, 'temperature': SWC_TEMPERATURE if METHOD == 'swcs' else None, 'heldout_speaker': heldout, 'train_csv_sha256': sha256_file(source / 'train.csv'), 'test_csv_sha256': sha256_file(source / 'test.csv'), 'audio_stat_sha256': audio_stat_hash(pd.concat([dataset.df, test], ignore_index=True)), 'initial_head_sha256': initial_hash, 'backbone_file_sha256': backbone['file_sha256'], 'script_sha256': sha256_file(Path(__file__)), 'lr': HEAD_LR, 'betas': list(BETAS), 'physical_train_batch': METHOD_TRAIN_BATCH[METHOD], 'eval_batch': METHOD_EVAL_BATCH[METHOD], 'episodes_per_update': 2, 'utterances_per_update': 16, 'updates_per_epoch': updates_for_epoch(len(dataset)), 'sampler': 'same_word_4_severities_2_distinct_speakers', 'input_mode': 'direct_waveform', 'amp': USE_AMP and device.type == 'cuda', 'device_type': device.type, 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu', 'versions': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'numpy', 'pandas', 'soundfile', 'scikit-learn']}}


def make_checkpoint(model, config, epoch):
    state = cpu_copy(model.state_dict())
    return {'config': config, 'epoch': epoch, 'model_state_dict': state, 'head_state_sha256': tensor_state_hash(state), 'head_architecture': '768 -> 768 -> ReLU -> 4', 'selection_basis': 'fixed_epoch'}


def verify_checkpoint(state, config):
    if state['config'] != config:
        raise RuntimeError('Checkpoint configuration differs from this run')
    if state['head_state_sha256'] != tensor_state_hash(state['model_state_dict']):
        raise RuntimeError('Checkpoint head hash mismatch')
    if not 0 <= state['epoch'] <= EPOCHS:
        raise RuntimeError('Invalid checkpoint epoch')


def train_or_load(root, config, dataset, sampler, encoder, extractor, device, initial, stage):
    final_path = root / f'final_epoch_{EPOCHS:03d}.pt'
    state_path = root / 'last_state.pt'
    set_seed(BASE_SEED)
    model = SeverityHead().to(device)
    model.load_state_dict(initial, strict=True)
    if final_path.exists():
        final = torch_load(final_path, torch.device('cpu'))
        verify_checkpoint(final, config)
        if final['epoch'] != EPOCHS:
            raise RuntimeError('Final checkpoint is incomplete')
        model.load_state_dict(final['model_state_dict'], strict=True)
        return model
    if stage == 'evaluate':
        raise FileNotFoundError(final_path)
    optimizer = build_optimizer(encoder, model)
    scaler = make_scaler(device)
    history, last = [], 0
    if state_path.exists():
        state = torch_load(state_path, torch.device('cpu'))
        verify_checkpoint(state, config)
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        if (scaler is None) != (state['scaler_state_dict'] is None):
            raise RuntimeError('AMP mode changed on resume')
        if scaler is not None:
            scaler.load_state_dict(state['scaler_state_dict'])
        last, history = state['epoch'], state['train_history']
        if [r['epoch'] for r in history] != list(range(1, last + 1)):
            raise RuntimeError('Invalid training history')
        restore_rng(state['rng_state'])
        del state
    print(f'{PROTOCOL} {METHOD} resume_epoch={last} final_epoch={EPOCHS} updates_per_epoch={updates_for_epoch(len(dataset))}', flush=True)
    for epoch in range(last + 1, EPOCHS + 1):
        record = TRAIN_FUNCTION(encoder, model, dataset, extractor, optimizer, scaler, device, epoch, METHOD, sampler)
        history.append(record)
        state = make_checkpoint(model, config, epoch)
        state.update(optimizer_state_dict=cpu_copy(optimizer.state_dict()), scaler_state_dict=cpu_copy(scaler.state_dict()) if scaler is not None else None, rng_state=capture_rng(), train_history=history)
        atomic_torch_save(state, state_path)
        save_json({'records': history}, root / 'train_history.json')
        print(f"{PROTOCOL} {METHOD} epoch={epoch}/{EPOCHS} train_acc={record['accuracy']:.6f} train_macro_f1={record['macro_f1']:.6f} ce={record['ce_loss']:.6f} total={record['total_loss']:.6f}", flush=True)
        del state
    atomic_torch_save(make_checkpoint(model, config, EPOCHS), final_path)
    save_json({'records': history}, root / 'train_history.json')
    return model


def run_fold(root, source, heldout, args, encoder, extractor, backbone, device, initial):
    root.mkdir(parents=True, exist_ok=True)
    dataset = WaveformDataset(source / 'train.csv')
    test = read_split(source / 'test.csv')
    info = validate_training_splits(dataset.df, test, heldout)
    sampler = SameWordStructuredBatch8Sampler(dataset)
    config = training_config(source, dataset, test, heldout, args, device, backbone, tensor_state_hash(initial))
    config_path = root / 'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise RuntimeError(f'Configuration changed; use a new output directory: {root}')
    else:
        save_json(config, config_path)
    save_json(info, root / 'split_check.json')
    model = train_or_load(root, config, dataset, sampler, encoder, extractor, device, initial, args.stage)
    if args.stage != 'train':
        with isolated_evaluation_rng():
            metrics, predictions = evaluate_test_from_waveforms(model, encoder, extractor, source / 'test.csv', device, METHOD_EVAL_BATCH[METHOD])
        metrics['confusion_matrix'] = confusion_matrix(predictions['true_label'], predictions['pred_label'], labels=list(range(4))).tolist()
        metrics.update(protocol=PROTOCOL, method=METHOD, epoch=EPOCHS, heldout_speaker=heldout, selection_basis='fixed_epoch')
        if heldout:
            metrics['metric_scope'] = 'single held-out speaker; use pooled_test_metrics.json for overall four-class performance'
        predictions.insert(0, 'fold', f'fold_{heldout}' if heldout else 'SD')
        save_csv(predictions, root / 'test_predictions.csv')
        save_json(metrics, root / 'test_metrics.json')
        print(json.dumps(json_safe(metrics), ensure_ascii=False), flush=True)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def summarize_loso(root, selected_folds):
    frames = [pd.read_csv(root / f'fold_{speaker}' / 'test_predictions.csv') for speaker in selected_folds]
    predictions = pd.concat(frames, ignore_index=True)
    if predictions['utterance_id'].duplicated().any():
        raise RuntimeError('Duplicate utterances in pooled predictions')
    metrics = classification_metrics(predictions['true_label'], predictions['pred_label'], detailed=True)
    metrics.update(protocol='LOSO', method=METHOD, epoch=EPOCHS, folds=selected_folds, complete_15_folds=set(selected_folds) == set(SPEAKER_LABELS), sample_count=len(predictions), aggregation='pooled_utterance_predictions', selection_basis='fixed_epoch')
    if set(selected_folds) == set(SPEAKER_LABELS):
        filename = 'pooled_test'
    else:
        filename = 'subset_' + '_'.join(selected_folds)
    save_csv(predictions, root / f'{filename}_predictions.csv')
    save_json(metrics, root / f'{filename}_metrics.json')
    print(json.dumps(json_safe(metrics), ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits-root', type=Path, default=Path('/home/wooje/바탕화면/code/1_preprocessing/splits_train'))
    parser.add_argument('--output-dir', type=Path, default=Path('/home/wooje/바탕화면/code/3_training/results') / PROTOCOL / ('baseline' if METHOD == 'ce_only' else 'csrl'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stage', choices=['all', 'train', 'evaluate'], default='all')
    parser.add_argument('--check-only', action='store_true')
    if PROTOCOL == 'LOSO':
        parser.add_argument('--folds', nargs='+', choices=sorted(SPEAKER_LABELS), default=sorted(SPEAKER_LABELS))
    return parser.parse_args()


def main():
    args = parse_args()
    args.splits_root = args.splits_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    selected = args.folds if PROTOCOL == 'LOSO' else [None]
    if len(set(selected)) != len(selected):
        raise ValueError('Duplicate fold selection')
    sources = [(speaker, args.splits_root / 'LOSO' / f'fold_{speaker}' if speaker else args.splits_root / 'SD') for speaker in selected]
    for speaker, source in sources:
        train, test = read_split(source / 'train.csv'), read_split(source / 'test.csv')
        print(json.dumps(dict(fold=speaker or 'SD', **validate_training_splits(train, test, speaker)), ensure_ascii=False), flush=True)
    if args.check_only:
        return
    device = torch.device(args.device)
    if device.type not in {'cpu', 'cuda'} or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Use an available CUDA device or --device cpu')
    with output_lock(args.output_dir):
        encoder, extractor, backbone = prepare_frozen_backbone(args.output_dir, device)
        set_seed(BASE_SEED)
        initial = cpu_copy(SeverityHead().state_dict())
        for speaker, source in sources:
            root = args.output_dir / f'fold_{speaker}' if speaker else args.output_dir
            run_fold(root, source, speaker, args, encoder, extractor, backbone, device, initial)
        if PROTOCOL == 'LOSO' and args.stage != 'train':
            summarize_loso(args.output_dir, selected)


if __name__ == '__main__':
    main()
