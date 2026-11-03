#!/usr/bin/env python3

# -*- coding: utf-8 -*-

from __future__ import annotations



"""SD 최종 CE-only / CSRL의 ground-truth severity t-SNE 시각화.



목적

----

현재 최종 SD 학습 실행에서 저장된 Wav2Vec2 frozen backbone과 두 severity head를 정확히 복원하고,

SD TEST uncommon 4,500발화의 post-ReLU latent z(768-D)를 시각화한다.



대상 모델

- CE-only: final_epoch_020.pt (E20, 원래 학습/eval batch 16)

- CSRL:    final_epoch_040.pt (E40, 원래 학습 batch 8 x 2 episodes, eval batch 8)

- CSRL lambda=0.20, temperature=1.50

- backbone/head는 모두 eval() + requires_grad=False이며 분석 중 절대 업데이트하지 않는다.



시각화 절차

1\) 저장된 SD TEST 4,500발화에서 공통 frozen Wav2Vec2 h를 batch=8로 한 번 추출한다.

2\) 같은 h를 최종 CE-only head와 최종 CSRL head에 넣어 각 post-ReLU z(768-D)를 얻는다.

3\) 각 방법별 z를 PCA(50-D)로 축소한다. PCA는 각 방법의 z에 독립적으로 fit한다.

4\) PCA 50-D -> t-SNE 2-D를 각 방법에 독립적으로 fit한다.

5\) t-SNE 좌표는 모델 예측이 아니라 ground-truth severity 4개 클래스로 색칠한다.



중요

- CE/CSRL t-SNE는 독립적으로 fit되므로 두 그림의 x/y 좌표 자체를 직접 대응시키지 않는다.

- cluster의 절대 거리/크기를 정량 근거로 사용하지 않고 qualitative visualization으로 해석한다.

- LOSO, HuBERT, speaker plot, predicted-severity coloring은 수행하지 않는다.



생성 파일

- sd_ce_vs_csrl_true_severity_tsne.png



실행

  CUDA_VISIBLE_DEVICES=1 python3 visualize_sd_ground_truth_final.py --device cuda

검사만

  python3 visualize_sd_ground_truth_final.py --check-only



이 파일은 최종 학습 Python 파일을 import하지 않는 단일 실행 파일이다.

"""

import argparse

import gc

import hashlib

import importlib.metadata

import inspect

import json

import math

import os

import platform

import random

import time

import warnings

from contextlib import contextmanager, nullcontext

from dataclasses import dataclass

from pathlib import Path

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple



os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

import soundfile as sf

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm



try:

    from transformers import AutoFeatureExtractor, Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2FeatureExtractor

    TRANSFORMERS_IMPORT_ERROR = None

except ImportError as exc:

    AutoFeatureExtractor = Wav2Vec2Config = Wav2Vec2Model = Wav2Vec2FeatureExtractor = None

    TRANSFORMERS_IMPORT_ERROR = str(exc)



RUN_ROOT = Path('/home/wooje/바탕화면/code/3_training/results/SD')

METHOD_DIRS = {'ce_only':'baseline', 'swcs':'csrl'}
SPLITS_ROOT = Path('/home/wooje/바탕화면/code/1_preprocessing/splits_train/SD')

POSTHOC_RELATIVE = Path('posthoc/sd_ground_truth_tsne_final')

ANALYSIS_ID = 'sd_ground_truth_tsne_ce20_swcs40_pca50_v1'

EXPERIMENT_ID = 'sd_ce20_b16_swcs40_b8_matched_utterances_lambda020_tau150_v1'

CHECKPOINT_FORMAT = 'sd_head_with_shared_frozen_backbone_v1'

AUDIO_MODEL_NAME = 'facebook/wav2vec2-base'


METHOD_EPOCHS = {'ce_only': 20, 'swcs': 40}

METHOD_TRAIN_BATCH = {'ce_only': 16, 'swcs': 8}

METHOD_EVAL_BATCH = {'ce_only': 16, 'swcs': 8}  # 원래 severity 평가 배치; 시각화 특징 추출은 공통 8

METHOD_GRAD_ACCUM = {'ce_only': 1, 'swcs': 2}

FEATURE_BATCH = 8

NUM_WORKERS = 0

USE_AMP = True

BASE_SEED = 42

SAMPLE_RATE = 16000

WAV2VEC_DIM = HEAD_DIM = 768

NUM_CLASSES = 4

LABEL_NAMES = ['very_low', 'low', 'medium', 'high']

SD_SPEAKERS = ('F02','F03','F04','F05','M01','M04','M05','M07','M08',

               'M09','M10','M11','M12','M14','M16')

SD_TRAIN_ROWS, SD_TEST_ROWS = 6975, 4500

EXPECTED_TRAIN_WORDS, SD_TEST_WORDS = 155, 300

COMMON_PER_SPEAKER, UNCOMMON_PER_SPEAKER = 465, 300

UTTERANCES_PER_UPDATE = 16

SWC_LAMBDA, SWC_TEMPERATURE = 0.20, 1.50





# 아래 함수들은 최종 학습 코드의 대응 구현을 유지한다.



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

    if isinstance(value, float) and not math.isfinite(value):

        return None

    return value







def save_json(obj: Mapping[str, Any], path: Path) -> None:

    """QWK가 정의되지 않는 경우 NaN 대신 JSON null로 기록한다."""

    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + '.tmp')

    tmp.write_text(json.dumps(json_safe(obj), ensure_ascii=False, indent=2,

                              allow_nan=False), encoding='utf-8')

    tmp.replace(path)







def load_json(path: Path) -> Dict[str, Any]:

    return json.loads(path.read_text(encoding="utf-8"))







def save_csv(df: pd.DataFrame, path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    df.to_csv(tmp, index=False, encoding="utf-8-sig")

    tmp.replace(path)







def torch_load(path: Path, device: torch.device):

    try:

        return torch.load(path, map_location=device, weights_only=False)

    except TypeError:

        return torch.load(path, map_location=device)







def tensor_state_hash(state: Mapping[str, torch.Tensor]) -> str:

    h = hashlib.sha256()

    for key in sorted(state):

        t = state[key].detach().cpu().contiguous()

        h.update(key.encode('utf-8'))

        h.update(str(t.dtype).encode('ascii'))

        h.update(str(tuple(t.shape)).encode('ascii'))

        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())

    return h.hexdigest()







def frame_signature(frame: pd.DataFrame) -> str:

    cols = ['utterance_id', 'path', 'speaker', 'severity_label', 'word_key']

    payload = frame[cols].astype(str).to_csv(index=False, lineterminator='\n')

    return hashlib.sha256(payload.encode('utf-8')).hexdigest()







def contract_hash(contract: Mapping[str, Any]) -> str:

    return hashlib.sha256(json.dumps(json_safe(contract), sort_keys=True,

                                     ensure_ascii=False).encode('utf-8')).hexdigest()







def read_split(path: Path) -> pd.DataFrame:

    if not path.is_file():

        raise FileNotFoundError(path)



    df = pd.read_csv(path)



    required = [

        "utterance_id",

        "path",

        "speaker",

        "severity_name",

        "severity_label",

    ]

    missing = [c for c in required if c not in df.columns]

    if missing:

        raise RuntimeError(f"{path}: missing columns {missing}")



    for col in ["utterance_id", "path", "speaker", "severity_name"]:

        df[col] = df[col].astype(str).str.strip()



    df["speaker"] = df["speaker"].str.upper()

    df["severity_name"] = df["severity_name"].str.lower()

    df["severity_label"] = pd.to_numeric(

        df["severity_label"], errors="raise"

    ).astype(int)



    invalid_labels = sorted(

        set(df["severity_label"].unique()) - set(range(NUM_CLASSES))

    )

    if invalid_labels:

        raise RuntimeError(f"{path}: invalid severity labels {invalid_labels}")



    # Preserve the current split word-identity fallback.

    if "word_key" in df.columns:

        identity_col = "word_key"

    elif "word_code" in df.columns:

        identity_col = "word_code"

    elif "word_text" in df.columns:

        identity_col = "word_text"

    else:

        raise RuntimeError(f"{path}: no word_key, word_code or word_text column")



    df["word_key"] = df[identity_col].astype(str).str.strip()



    # Text is not used by this experiment. Keep it only for prediction logs.

    if "word_text" in df.columns:

        text = df["word_text"].astype(str).str.strip()

        df["text_word"] = np.where(text.ne(""), text, df["word_key"])

    else:

        df["text_word"] = df["word_key"]



    if df["word_key"].eq("").any():

        raise RuntimeError(f"{path}: empty word identity")



    if df["utterance_id"].duplicated().any():

        raise RuntimeError(f"{path}: duplicated utterance_id")

    if df["path"].duplicated().any():

        raise RuntimeError(f"{path}: duplicated audio path")



    # One speaker must map to exactly one severity.

    speaker_mapping = (

        df[["speaker", "severity_label"]]

        .drop_duplicates()

        .groupby("speaker")["severity_label"]

        .nunique()

    )

    if (speaker_mapping != 1).any():

        raise RuntimeError(f"{path}: a speaker maps to multiple severity labels")



    missing_audio = [p for p in df["path"] if not Path(p).is_file()]

    if missing_audio:

        raise FileNotFoundError(

            f"{path}: {len(missing_audio)} missing audio files; "

            f"first={missing_audio[0]}"

        )



    return df.reset_index(drop=True)







def load_audio(path: str) -> np.ndarray:

    waveform, sr = sf.read(path, dtype="float32", always_2d=False)



    if waveform.ndim == 2:

        waveform = waveform.mean(axis=1)

    if waveform.ndim != 1:

        raise RuntimeError(f"{path}: invalid waveform shape {waveform.shape}")

    if sr != SAMPLE_RATE:

        raise RuntimeError(f"{path}: sample rate={sr}, expected={SAMPLE_RATE}")

    if waveform.size == 0:

        raise RuntimeError(f"{path}: empty waveform")

    if not np.isfinite(waveform).all():

        raise RuntimeError(f"{path}: waveform contains NaN/Inf")



    return waveform.astype(np.float32, copy=False)







class WaveformDataset(Dataset):

    def __init__(self, csv_path: Path):

        self.df = read_split(csv_path)



    def __len__(self) -> int:

        return len(self.df)



    def __getitem__(self, index: int) -> Dict[str, Any]:

        row = self.df.iloc[int(index)]

        return {

            "dataset_index": int(index),

            "utterance_id": str(row["utterance_id"]),

            "path": str(row["path"]),

            "speaker": str(row["speaker"]),

            "severity_label": int(row["severity_label"]),

            "severity_name": str(row["severity_name"]),

            "word_key": str(row["word_key"]),

            "text_word": str(row["text_word"]),

            "waveform": load_audio(str(row["path"])),

        }







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







def method_batch_settings(method: str) -> Dict[str, Any]:

    """모든 실행/저장 경로가 참조하는 방법별 고정 배치 계약."""

    if method not in METHOD_EPOCHS:

        raise ValueError(f'지원하지 않는 방법: {method}')

    calls = METHOD_GRAD_ACCUM[method]

    return {

        'physical_train_batch': METHOD_TRAIN_BATCH[method],

        'test_batch': METHOD_EVAL_BATCH[method],

        'torgo_batch': METHOD_EVAL_BATCH[method],

        'gradient_accumulation_steps': calls,

        'utterances_per_update': UTTERANCES_PER_UPDATE,

        'forward_calls_per_update': calls, 'backward_calls_per_update': calls,

        'optimizer_steps_per_update': 1,

        'padding_scope': ('two_8_utterance_episodes_together' if method == 'ce_only'

                          else 'one_8_utterance_episode'),

    }







class CompletelyFrozenWav2Vec2(nn.Module):

    """저장된 config/state로만 복원한다. 사전학습 모델 다운로드를 수행하지 않는다."""

    def __init__(self, config):

        super().__init__()

        require_transformers()

        self.wav2vec2 = Wav2Vec2Model(config)

        self.requires_grad_(False)

        self.eval()





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





def load_asset_encoder(asset: Mapping[str, Any], device: torch.device):

    require_transformers()

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

    return encoder, extractor





# -----------------------------------------------------------------------------

# 저장된 SD 실행 검사. 입력 파일에는 쓰지 않는다.

# -----------------------------------------------------------------------------

def require_transformers() -> None:

    if Wav2Vec2Model is None:

        raise RuntimeError('학습에 사용한 transformers 환경에서 실행하십시오. '

                           f'현재 import 오류: {TRANSFORMERS_IMPORT_ERROR}')





def choose_device(name: str) -> torch.device:

    if name == 'auto':

        name = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(name)

    if device.type == 'cuda' and not torch.cuda.is_available():

        raise RuntimeError('CUDA를 사용할 수 없습니다. GPU/conda 환경을 확인하십시오.')

    if device.type == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):

        raise RuntimeError('MPS를 사용할 수 없습니다.')

    if device.type not in ('cpu', 'cuda', 'mps'):

        raise ValueError(f'지원하지 않는 장치: {device}')

    return device





def environment_info(device: torch.device) -> Dict[str, Any]:

    versions = {}

    for package in ('torch', 'transformers', 'numpy', 'pandas', 'scikit-learn',

                    'scipy', 'soundfile', 'joblib'):

        try:

            versions[package] = importlib.metadata.version(package)

        except importlib.metadata.PackageNotFoundError:

            versions[package] = None

    result = {'python':platform.python_version(), 'packages':versions,

              'device_type':device.type, 'cuda_runtime':torch.version.cuda,

              'cudnn':torch.backends.cudnn.version(),

              'amp_enabled':USE_AMP and device.type == 'cuda',

              'amp_dtype':str(torch.get_autocast_dtype('cuda')) if device.type == 'cuda' else None,

              'float32_matmul_precision':torch.get_float32_matmul_precision(),

              'cuda_matmul_allow_tf32':bool(torch.backends.cuda.matmul.allow_tf32),

              'cudnn_allow_tf32':bool(torch.backends.cudnn.allow_tf32)}

    if device.type == 'cuda':

        result['gpu_name'] = torch.cuda.get_device_name(device)

        result['gpu_capability'] = list(torch.cuda.get_device_capability(device))

    return result





def must_equal(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:

    for key, value in expected.items():

        if key not in actual or actual[key] != value:

            raise RuntimeError(f'{label}: {key} 불일치. 실제={actual.get(key)!r}, 기대={value!r}')





def checked_child(root: Path, relative: str) -> Path:

    rel = Path(relative)

    path = (root / rel).resolve()

    if rel.is_absolute() or not path.is_relative_to(root.resolve()):

        raise RuntimeError(f'실행 폴더 밖을 가리키는 상대 경로: {relative}')

    return path





def validate_sd_frames(train: pd.DataFrame, test: pd.DataFrame) -> None:

    """저장된 SD 분할을 그대로 검사한다. 새 split/샘플러는 만들지 않는다."""

    for name, frame, nrows, nwords, nspk in (

        ('TRAIN', train, SD_TRAIN_ROWS, EXPECTED_TRAIN_WORDS, COMMON_PER_SPEAKER),

        ('TEST', test, SD_TEST_ROWS, SD_TEST_WORDS, UNCOMMON_PER_SPEAKER),

    ):

        if len(frame) != nrows or frame['word_key'].nunique() != nwords:

            raise RuntimeError(f'{name}: 예상 발화/단어 수 {nrows}/{nwords}와 다릅니다.')

        if set(frame['speaker']) != set(SD_SPEAKERS):

            raise RuntimeError(f'{name}: 15명 화자 목록 불일치')

        if set(frame['severity_label']) != set(range(NUM_CLASSES)):

            raise RuntimeError(f'{name}: 원래 4개 중증도 라벨이 없습니다.')

        if not frame['severity_label'].map(dict(enumerate(LABEL_NAMES))).eq(frame['severity_name']).all():

            raise RuntimeError(f'{name}: 중증도 label/name 대응 불일치')

        for col in ('utterance_id', 'path', 'speaker', 'word_key'):

            if frame[col].isna().any() or frame[col].astype(str).str.strip().eq('').any():

                raise RuntimeError(f'{name}: {col} 누락')

        for col in ('utterance_id', 'path'):

            if frame[col].duplicated().any():

                raise RuntimeError(f'{name}: {col} 중복')

        words = set(frame['word_key'])

        for sp, group in frame.groupby('speaker'):

            if len(group) != nspk or set(group['word_key']) != words:

                raise RuntimeError(f'{name}/{sp}: 전체 단어 또는 발화 수 불일치')

            if group['severity_label'].nunique() != 1:

                raise RuntimeError(f'{name}/{sp}: 중증도 라벨 불일치')

    for col in ('utterance_id', 'path', 'word_key'):

        if set(train[col]) & set(test[col]):

            raise RuntimeError(f'common/uncommon 사이 {col} 중복')

    if (train.groupby('speaker')['severity_label'].first().to_dict()

            != test.groupby('speaker')['severity_label'].first().to_dict()):

        raise RuntimeError('TRAIN/TEST 화자-중증도 대응 불일치')





def audio_stat_digest(frame: pd.DataFrame) -> str:

    """캐시 무효화용 파일 경로/크기/수정시각 해시. 파형 바이트 해시가 아님."""

    digest = hashlib.sha256()

    for row in frame.itertuples(index=False):

        path = Path(row.path)

        stat = path.stat()

        digest.update(json.dumps([str(row.utterance_id), str(path.resolve()),

                                  stat.st_size, stat.st_mtime_ns],

                                 ensure_ascii=False).encode('utf-8') + b'\n')

    return digest.hexdigest()





@dataclass

class SavedInputs:

    root: Path

    cfg: Dict[str, Any]

    asset_meta: Dict[str, Any]

    asset_payload: Dict[str, Any]

    checkpoints: Dict[str, Dict[str, Any]]

    frames: Dict[str, pd.DataFrame]

    paths_sha256: Dict[str, str]

    audio_stats: Dict[str, str]





def inspect_saved_inputs(root: Path) -> SavedInputs:
    root = root.expanduser().resolve()
    frames = {s: read_split(SPLITS_ROOT / f'{s}.csv') for s in ('train', 'test')}
    validate_sd_frames(frames['train'], frames['test'])
    hashes = {str(SPLITS_ROOT / f'{s}.csv'): sha256_file(SPLITS_ROOT / f'{s}.csv') for s in frames}
    checkpoints, configs, assets, metas = {}, {}, {}, {}
    for method, epoch in METHOD_EPOCHS.items():
        directory = root / METHOD_DIRS[method]
        path = directory / f'final_epoch_{epoch:03d}.pt'
        config_path = directory / 'config.json'
        asset_path = directory / 'assets/frozen_encoder.pt'
        meta_path = directory / 'assets/backbone.json'
        cp = torch_load(path, torch.device('cpu'))
        cfg = cp['config']
        if load_json(config_path) != cfg:
            raise RuntimeError(f'{method}: checkpoint/config.json mismatch')
        must_equal(cfg, {'format':'uaspeech_fixed_training_v1', 'protocol':'SD', 'method':method,
                        'seed':42, 'epochs':epoch, 'heldout_speaker':None,
                        'lambda':SWC_LAMBDA if method == 'swcs' else 0.0,
                        'temperature':SWC_TEMPERATURE if method == 'swcs' else None,
                        'physical_train_batch':METHOD_TRAIN_BATCH[method],
                        'eval_batch':METHOD_EVAL_BATCH[method]}, 'training configuration')
        must_equal(cp, {'epoch':epoch, 'selection_basis':'fixed_epoch'}, 'checkpoint')
        for split in frames:
            if cfg[f'{split}_csv_sha256'] != hashes[str(SPLITS_ROOT / f'{split}.csv')]:
                raise RuntimeError(f'{method}: {split} CSV differs from training')
        if tensor_state_hash(cp['model_state_dict']) != cp['head_state_sha256']:
            raise RuntimeError(f'{method}: head hash mismatch')
        head = SeverityHead()
        head.load_state_dict(cp['model_state_dict'], strict=True)
        if not all(torch.isfinite(t).all() for t in head.state_dict().values()):
            raise RuntimeError(f'{method}: nonfinite head weights')
        meta = load_json(meta_path)
        asset_sha = sha256_file(asset_path)
        if asset_sha != meta['file_sha256'] or asset_sha != cfg['backbone_file_sha256']:
            raise RuntimeError(f'{method}: backbone file hash mismatch')
        asset = torch_load(asset_path, torch.device('cpu'))
        must_equal(asset, {'audio_model_name':AUDIO_MODEL_NAME}, 'backbone')
        must_equal(asset['encoder_config'], {'hidden_size':768, 'feat_extract_norm':'group'}, 'backbone config')
        must_equal(asset['feature_extractor_config'], {'sampling_rate':16000}, 'feature extractor')
        meta = dict(meta, encoder_state_sha256=tensor_state_hash(asset['encoder_state_dict']))
        cp = dict(cp, swc_enabled=method == 'swcs', swc_lambda=cfg['lambda'],
                  swc_temperature=cfg['temperature'], physical_train_batch=cfg['physical_train_batch'],
                  test_batch=cfg['eval_batch'], initial_head_sha256=cfg['initial_head_sha256'])
        checkpoints[method], configs[method], assets[method], metas[method] = cp, cfg, asset, meta
        for p in [path, config_path, asset_path, meta_path]:
            hashes[str(p)] = sha256_file(p)
    if metas['ce_only']['encoder_state_sha256'] != metas['swcs']['encoder_state_sha256']:
        raise RuntimeError('Baseline and CSRL backbone weights differ; shared-h analysis is not valid')
    for key in ['feature_extractor_config', 'attention_implementation']:
        if assets['ce_only'].get(key) != assets['swcs'].get(key):
            raise RuntimeError(f'Baseline/CSRL {key} mismatch')
    if configs['ce_only']['initial_head_sha256'] != configs['swcs']['initial_head_sha256']:
        raise RuntimeError('Initial head hashes differ')
    cfg = {'run_id':contract_hash(configs), 'training_configs':configs}
    stats = {s:audio_stat_digest(f) for s,f in frames.items()}
    return SavedInputs(root, cfg, metas['ce_only'], assets['ce_only'], checkpoints, frames, hashes, stats)





def checkpoint_manifest(inputs: SavedInputs) -> Dict[str, Any]:

    manifest = {}

    for method, cp in inputs.checkpoints.items():

        relative = str(inputs.root / METHOD_DIRS[method] / f'final_epoch_{METHOD_EPOCHS[method]:03d}.pt')

        manifest[method] = {

            'checkpoint_path':str(inputs.root / relative),

            'checkpoint_file_sha256':inputs.paths_sha256[relative],

            'head_state_sha256':cp['head_state_sha256'],

            'epoch':cp['epoch'], 'swc_enabled':cp['swc_enabled'],

            'swc_lambda':cp['swc_lambda'], 'swc_temperature':cp['swc_temperature'],

            'original_training_batch':cp['physical_train_batch'],

            'original_severity_evaluation_batch':cp['test_batch'],

            'probe_feature_batch':FEATURE_BATCH,

        }

    return manifest





def verify_originals_unchanged(inputs: SavedInputs) -> None:

    for relative, digest in inputs.paths_sha256.items():

        if sha256_file(inputs.root / relative) != digest:

            raise RuntimeError(f'분석 중 원래 학습 파일이 변경됐습니다: {relative}')

    for split, frame in inputs.frames.items():

        if audio_stat_digest(frame) != inputs.audio_stats[split]:

            raise RuntimeError(f'분석 중 {split}의 음성 파일 상태가 변경됐습니다.')





def assert_frozen(module: nn.Module, name: str) -> None:

    if any(m.training for m in module.modules()):

        raise RuntimeError(f'{name}: eval() 상태가 아닙니다.')

    if any(p.requires_grad or p.grad is not None for p in module.parameters()):

        raise RuntimeError(f'{name}: requires_grad 또는 gradient가 남아 있습니다.')





def restore_models(inputs: SavedInputs, device: torch.device):

    encoder, extractor = load_asset_encoder(inputs.asset_payload, device)

    encoder.requires_grad_(False).eval()

    assert_frozen(encoder, 'backbone')

    if tensor_state_hash(encoder.state_dict()) != inputs.asset_meta['encoder_state_sha256']:

        raise RuntimeError('복원한 backbone 가중치가 저장본과 다릅니다.')

    heads = {}

    for method, cp in inputs.checkpoints.items():

        head = SeverityHead().to(device)

        head.load_state_dict(cp['model_state_dict'], strict=True)

        head.requires_grad_(False).eval()

        assert_frozen(head, method)

        heads[method] = head

    return encoder, extractor, heads





# -----------------------------------------------------------------------------

# 공통 h 추출 / 방법별 z. 재사용 캐시는 현재 분석 폴더의 파일만 허용한다.

# -----------------------------------------------------------------------------

class FrameWaveformDataset(WaveformDataset):

    def __init__(self, frame: pd.DataFrame):

        self.df = frame.reset_index(drop=True).copy()





def feature_signature(inputs: SavedInputs, split: str, runtime: Mapping[str, Any]) -> Dict[str, Any]:

    return {

        'analysis_id':ANALYSIS_ID, 'run_id':inputs.cfg['run_id'], 'split':split,

        'frame_sha256':frame_signature(inputs.frames[split]),

        'saved_csv_sha256':inputs.paths_sha256[str(SPLITS_ROOT / f'{split}.csv')],

        'audio_path_size_mtime_sha256':inputs.audio_stats[split],

        'backbone_file_sha256':inputs.asset_meta['file_sha256'],

        'backbone_state_sha256':inputs.asset_meta['encoder_state_sha256'],

        'feature_extractor_config_sha256':contract_hash(inputs.asset_payload['feature_extractor_config']),

        'attention_implementation':inputs.asset_payload.get('attention_implementation'),

        'physical_feature_batch':FEATURE_BATCH, 'shuffle':False, 'drop_last':False,

        'pooling':'valid_frame_mean', 'attention_mask':None,

        'padding_scope':'one_batch_within_one_split', 'stored_dtype':'float32',

        'runtime':dict(runtime),

    }





def array_sha256(array: np.ndarray) -> str:

    a = np.ascontiguousarray(array)

    h = hashlib.sha256()

    h.update(str(a.shape).encode('ascii')); h.update(str(a.dtype).encode('ascii'))

    h.update(a.tobytes())

    return h.hexdigest()





def validate_embeddings(x: np.ndarray, rows: int, name: str) -> None:

    if x.shape != (rows, WAV2VEC_DIM) or x.dtype != np.float32:

        raise RuntimeError(f'{name}: 특징 shape/dtype 오류 {x.shape}/{x.dtype}')

    if not np.isfinite(x).all():

        raise FloatingPointError(f'{name}: 특징에 NaN/Inf가 있습니다.')





@torch.no_grad()

def extract_h(frame: pd.DataFrame, encoder: nn.Module, extractor,

              device: torch.device, split: str):

    encoder.eval(); assert_frozen(encoder, 'backbone')

    dataset = FrameWaveformDataset(frame)

    loader = DataLoader(dataset, batch_size=FEATURE_BATCH, shuffle=False, drop_last=False,

                        num_workers=NUM_WORKERS, collate_fn=AudioCollator(extractor),

                        pin_memory=device.type == 'cuda', worker_init_fn=seed_worker)

    h = np.empty((len(frame), WAV2VEC_DIM), dtype=np.float32)

    seen, sizes, offset = [], [], 0

    for batch in tqdm(loader, desc=f'공통 h 추출: {split}, batch={FEATURE_BATCH}', leave=True):

        values = batch['input_values'].to(device, non_blocking=True)

        lengths = batch['input_lengths'].to(device, non_blocking=True)

        n = len(batch['metadata'])

        if n <= 0 or n > FEATURE_BATCH or values.shape[0] != n:

            raise RuntimeError('특징 추출 배치 크기 오류')

        indices = [int(m['dataset_index']) for m in batch['metadata']]

        if indices != list(range(offset, offset+n)):

            raise RuntimeError('특징 추출 데이터 순서가 바뀌었습니다.')

        with amp_context(device):

            pooled = encoder(values, lengths)

        if tuple(pooled.shape) != (n, WAV2VEC_DIM) or not torch.isfinite(pooled).all():

            raise RuntimeError('공통 h의 차원 또는 유한성 오류')

        h[offset:offset+n] = pooled.float().cpu().numpy()

        seen.extend(str(m['utterance_id']) for m in batch['metadata'])

        sizes.append(n); offset += n

        del batch, values, lengths, pooled

    if seen != frame['utterance_id'].astype(str).tolist() or offset != len(frame):

        raise RuntimeError('추출 후 발화 수/순서 불일치')

    validate_embeddings(h, len(frame), split)

    assert_frozen(encoder, 'backbone')

    info = {'sample_count':len(frame), 'batch_count':len(sizes),

            'batch_size':FEATURE_BATCH, 'last_batch_size':sizes[-1],

            'observed_batch_sizes':sorted(set(sizes)), 'shuffle':False,

            'drop_last':False, 'h_sha256':array_sha256(h)}

    return h, info





def save_h_cache(path: Path, h: np.ndarray, ids: np.ndarray,

                 signature: Mapping[str, Any], info: Mapping[str, Any]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + '.tmp')

    with tmp.open('wb') as f:

        np.savez_compressed(f, h=h, utterance_ids=ids.astype(str),

                            signature_json=np.asarray(json.dumps(json_safe(signature), sort_keys=True, ensure_ascii=False)),

                            extraction_json=np.asarray(json.dumps(json_safe(info), sort_keys=True, ensure_ascii=False)))

    tmp.replace(path)





def read_h_cache(path: Path, frame: pd.DataFrame, signature: Mapping[str, Any]):

    with np.load(path, allow_pickle=False) as data:

        saved = json.loads(str(data['signature_json'].item()))

        if saved != json_safe(signature):

            raise RuntimeError('현재 모델/분할/배치/전처리/환경과 h 캐시가 다릅니다.')

        if not np.array_equal(data['utterance_ids'].astype(str), frame['utterance_id'].astype(str).to_numpy()):

            raise RuntimeError('h 캐시의 발화 ID 순서가 다릅니다.')

        h = data['h'].copy()

        info = json.loads(str(data['extraction_json'].item()))

    validate_embeddings(h, len(frame), str(path))

    if info['h_sha256'] != array_sha256(h):

        raise RuntimeError('h 캐시 특징값 해시 불일치')

    return h, info





def extract_or_load_h(inputs: SavedInputs, output: Path, split: str,

                      encoder, extractor, device, runtime, force: bool):

    frame = inputs.frames[split]

    path = output / 'cache' / f'h_{split}.npz'

    signature = feature_signature(inputs, split, runtime)

    if path.is_file() and not force:

        try:

            h, info = read_h_cache(path, frame, signature)

        except (ValueError, KeyError, OSError, RuntimeError) as exc:

            raise RuntimeError(f'{path}: 캐시를 사용할 수 없습니다. '

                               '확인 후 --force-reextract-h로 다시 추출하십시오.') from exc

        print(f'기존 {split} 공통 h 캐시 사용: {path}', flush=True)

        return h, {**info, 'loaded_from_cache':True}

    h, info = extract_h(frame, encoder, extractor, device, split)

    if audio_stat_digest(frame) != inputs.audio_stats[split]:

        raise RuntimeError(f'{split}: 특징 추출 중 음성 파일이 변경됐습니다.')

    save_h_cache(path, h, frame['utterance_id'].astype(str).to_numpy(), signature, info)

    save_json({'signature':signature, 'extraction':info}, path.with_suffix('.json'))

    return h, {**info, 'loaded_from_cache':False}





@torch.no_grad()

def h_to_z(head: SeverityHead, h: np.ndarray, device: torch.device) -> np.ndarray:

    """두 head에 동일한 h, 동일한 batch=*8과* AMP 조건을 적용해 z만 추출한다."""

    validate_embeddings(h, len(h), '입력 h')

    head.eval(); assert_frozen(head, 'severity head')

    before = tensor_state_hash(head.state_dict())

    z = np.empty_like(h, dtype=np.float32)

    for start in range(0, len(h), FEATURE_BATCH):

        end = min(start+FEATURE_BATCH, len(h))

        batch = torch.from_numpy(h[start:end]).to(device)

        with amp_context(device):

            # classifier의 4차원 logits가 아닌 post-ReLU 768차원 z만 사용.

            output = head.head(batch)

        z[start:end] = output.float().cpu().numpy()

    validate_embeddings(z, len(h), 'z')

    if np.any(z < 0):

        raise RuntimeError('post-ReLU z에서 음수가 검출됐습니다.')

    if tensor_state_hash(head.state_dict()) != before:

        raise RuntimeError('특징 추출 중 head 가중치가 변경됐습니다.')

    assert_frozen(head, 'severity head')

    return z





# -----------------------------------------------------------------------------

# SD ground-truth severity PCA -> t-SNE visualization

# -----------------------------------------------------------------------------

PCA_COMPONENTS = 50

TSNE_RANDOM_STATE = 42

TSNE_PERPLEXITY = 30.0

TSNE_LEARNING_RATE = 'auto'

TSNE_INIT = 'pca'

TSNE_ITERATIONS = 2000

POINT_SIZE = 18

FIGURE_DPI = 300

FIGSIZE = (11.6, 4.8)

SEVERITY_ALPHA = 0.64

SEVERITY_COLORS = {

    0: '#1f77b4',

    1: '#ff7f0e',

    2: '#2ca02c',

    3: '#d62728',

}





def validate_visualization_test_frame(frame: pd.DataFrame) -> None:

    if len(frame) != SD_TEST_ROWS:

        raise RuntimeError(f'SD TEST rows={len(frame)}, expected={SD_TEST_ROWS}')

    if frame['word_key'].nunique() != SD_TEST_WORDS:

        raise RuntimeError(f'SD TEST words={frame["word_key"].nunique()}, expected={SD_TEST_WORDS}')

    if set(frame['speaker']) != set(SD_SPEAKERS):

        raise RuntimeError('SD TEST speaker set mismatch')

    if set(frame['severity_label']) != set(range(NUM_CLASSES)):

        raise RuntimeError('SD TEST must contain all four severity labels')

    counts = frame.groupby('speaker').size()

    if not (counts == UNCOMMON_PER_SPEAKER).all():

        raise RuntimeError(f'SD TEST per-speaker count mismatch: {counts.to_dict()}')





def run_pca_tsne(z: np.ndarray, label: str):

    validate_embeddings(z, SD_TEST_ROWS, f'{label} z')

    if PCA_COMPONENTS >= min(z.shape):

        raise RuntimeError('PCA_COMPONENTS must be smaller than min(N,D)')

    pca = PCA(n_components=PCA_COMPONENTS, svd_solver='randomized', random_state=BASE_SEED)

    z50 = pca.fit_transform(z).astype(np.float32, copy=False)

    if z50.shape != (SD_TEST_ROWS, PCA_COMPONENTS) or not np.isfinite(z50).all():

        raise RuntimeError(f'{label}: invalid PCA output {z50.shape}')

    explained = float(pca.explained_variance_ratio_.sum())

    kwargs = dict(

        n_components=2,

        perplexity=TSNE_PERPLEXITY,

        learning_rate=TSNE_LEARNING_RATE,

        init=TSNE_INIT,

        random_state=TSNE_RANDOM_STATE,

        metric='euclidean',

        method='barnes_hut',

        angle=0.5,

    )

    sig = inspect.signature(TSNE.__init__)

    if 'max_iter' in sig.parameters:

        kwargs['max_iter'] = TSNE_ITERATIONS

    else:

        kwargs['n_iter'] = TSNE_ITERATIONS

    print(f'{label}: PCA {z.shape[1]}->{PCA_COMPONENTS} explained={100*explained:.2f}% | '

          f't-SNE perplexity={TSNE_PERPLEXITY:g}, iter={TSNE_ITERATIONS}, seed={TSNE_RANDOM_STATE}',

          flush=True)

    xy = TSNE(**kwargs).fit_transform(z50).astype(np.float32, copy=False)

    if xy.shape != (SD_TEST_ROWS, 2) or not np.isfinite(xy).all():

        raise RuntimeError(f'{label}: invalid t-SNE output {xy.shape}')

    return xy, explained





def save_ground_truth_severity_comparison(
    ce_xy: np.ndarray,
    csrl_xy: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    labels = np.asarray(labels, dtype=np.int64)

    for name, xy in (("CE-only", ce_xy), ("CSRL", csrl_xy)):
        if xy.shape != (SD_TEST_ROWS, 2):
            raise RuntimeError(f"{name}: plot input shape mismatch: {xy.shape}")
    if labels.shape != (SD_TEST_ROWS,):
        raise RuntimeError(f"label shape mismatch: {labels.shape}")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.15, top=0.88, wspace=0.22)

    panel_info = (
        (axes[0], ce_xy, "(a) CE-only"),
        (axes[1], csrl_xy, "(b) CSRL"),
    )

    legend_handles = []
    legend_labels = []

    for ax, xy, title in panel_info:
        for c, name in enumerate(LABEL_NAMES):
            mask = labels == c
            if not np.any(mask):
                raise RuntimeError(f"ground-truth class {c} is absent")
            artist = ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=POINT_SIZE,
                alpha=SEVERITY_ALPHA,
                c=SEVERITY_COLORS[c],
                label=name,
                linewidths=0,
            )
            if ax is axes[0]:
                legend_handles.append(artist)
                legend_labels.append(name)

        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("t-SNE 1", fontsize=10)
        ax.set_ylabel("t-SNE 2", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)

    fig.legend(
        legend_handles,
        legend_labels,
        title="Ground-truth severity",
        loc="center left",
        bbox_to_anchor=(0.875, 0.5),
        fontsize=8,
        title_fontsize=8,
        markerscale=1.7,
        frameon=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def checkpoint_manifest_for_print(inputs: SavedInputs) -> Dict[str, Any]:

    out = {}

    for method, cp in inputs.checkpoints.items():

        out[method] = {

            'epoch': int(cp['epoch']),

            'physical_train_batch': int(cp['physical_train_batch']),

            'checkpoint': str(inputs.root/METHOD_DIRS[method]/f'final_epoch_{METHOD_EPOCHS[method]:03d}.pt'),

            'swc_lambda': cp.get('swc_lambda'),

            'swc_temperature': cp.get('swc_temperature'),

        }

    return out





def run_visualization(args) -> None:

    set_seed(BASE_SEED)

    device = choose_device(args.device)

    print('최종 SD 모델/TEST split/공유 backbone 검사 중...', flush=True)

    inputs = inspect_saved_inputs(RUN_ROOT)

    validate_visualization_test_frame(inputs.frames['test'])

    manifest = checkpoint_manifest_for_print(inputs)

    print(f"CE-only: E{manifest['ce_only']['epoch']} | original batch={manifest['ce_only']['physical_train_batch']} | "

          f"{manifest['ce_only']['checkpoint']}")

    print(f"CSRL:    E{manifest['swcs']['epoch']} | original batch={manifest['swcs']['physical_train_batch']} | "

          f"lambda={manifest['swcs']['swc_lambda']} tau={manifest['swcs']['swc_temperature']} | "

          f"{manifest['swcs']['checkpoint']}")

    print(f'SD TEST={SD_TEST_ROWS} uncommon utterances / {SD_TEST_WORDS} words / 15 speakers')

    print('Visualization: shared h(batch=8) -> method-specific z(768) -> PCA(50) -> independent t-SNE -> ground-truth severity')

    print('LOSO/HuBERT/speaker/predicted-label visualization: disabled')

    if args.check_only:

        print('검사 통과. waveform forward/PCA/t-SNE/PNG 저장은 수행하지 않았습니다.')

        return



    require_transformers()

    # 최종 논문용 그림은 바탕화면에 PNG 한 장만 저장한다.
    output = Path.home() / "바탕화면"
    output.mkdir(parents=True, exist_ok=True)
    output_path = output / "sd_ce_vs_csrl_true_severity_tsne.png"

    encoder, extractor, heads = restore_models(inputs, device)
    before_encoder = tensor_state_hash(encoder.state_dict())
    head_hashes = {m: tensor_state_hash(h.state_dict()) for m, h in heads.items()}

    # 두 방법에 동일한 SD TEST audio, 동일 순서, 동일 batch=8로 공통 h를 한 번만 추출한다.
    h, h_info = extract_h(inputs.frames['test'], encoder, extractor, device, 'SD TEST')
    if len(h) != SD_TEST_ROWS or h_info['sample_count'] != SD_TEST_ROWS:
        raise RuntimeError('SD TEST h extraction count mismatch')
    if tensor_state_hash(encoder.state_dict()) != before_encoder:
        raise RuntimeError('visualization 중 backbone 가중치가 변경됐습니다.')

    true_labels = inputs.frames['test']['severity_label'].to_numpy(np.int64)
    pca_report = {}
    xy_by_method = {}

    for method in ('ce_only', 'swcs'):
        z = h_to_z(heads[method], h, device)
        if tensor_state_hash(heads[method].state_dict()) != head_hashes[method]:
            raise RuntimeError(f'{method}: visualization 중 head 가중치가 변경됐습니다.')

        display_name = 'CE-only' if method == 'ce_only' else 'CSRL'
        xy, explained = run_pca_tsne(z, display_name)
        xy_by_method[method] = xy
        pca_report[method] = explained

        del z
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    save_ground_truth_severity_comparison(
        ce_xy=xy_by_method['ce_only'],
        csrl_xy=xy_by_method['swcs'],
        labels=true_labels,
        output_path=output_path,
    )

    del xy_by_method
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    verify_originals_unchanged(inputs)

    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f'PNG 생성 실패: {output_path}')

    print('\n완료 | 최종 논문용 t-SNE 그림 1개 생성', flush=True)
    print(f"CE-only PCA50 explained variance: {100*pca_report['ce_only']:.2f}%")
    print(f"CSRL    PCA50 explained variance: {100*pca_report['swcs']:.2f}%")
    print(f' - {output_path}')
    print('주의: CE-only/CSRL t-SNE는 독립 fit이므로 축 좌표 자체는 직접 비교하지 않습니다.', flush=True)



def parse_args(argv: Optional[Sequence[str]] = None):

    parser = argparse.ArgumentParser(description='Current final SD CE-only vs CSRL ground-truth severity t-SNE')

    parser.add_argument('--device', default='auto', help='auto, cuda, cuda:0, cpu, mps')

    parser.add_argument('--check-only', action='store_true', help='최종 모델/split/checkpoint 구조 검사만')

    return parser.parse_args(argv)





def main(argv: Optional[Sequence[str]] = None) -> None:

    run_visualization(parse_args(argv))





if __name__ == '__main__':

    main()
