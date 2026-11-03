# CSRL 실행 가이드

이 문서는 현재 정리된 코드의 **실제 파일명·인자·출력 구조**를 기준으로 작성했다. 코드 파일은 변경하지 않는다. 아래 전체 절차를 실제 음성과 학습 환경에서 끝까지 실행한 상태는 아니므로, 문서의 명령을 검증 완료된 실험 결과로 해석하지 않는다.

기본 순서는 **환경 → 음성 검사 → 최종 학습 분할 → 튜닝 분할 → 튜닝(선택) → 최종 학습 → SD 분석 → TORGO 평가**다. 튜닝을 다시 수행하지 않고 현재 고정 설정으로 학습하려면 5단계를 생략할 수 있다.

## 1. 실행 환경과 경로 준비

Linux 또는 macOS에서 실행한다. 학습·튜닝의 파일 잠금에 `fcntl`을 사용한다. 예시는 bash/zsh와 CUDA 연구 환경 기준이며, CPU를 사용하려면 `CSRL_DEVICE=cpu`로 설정한다. 실제 학습 시간과 메모리 요구량은 아직 측정하지 않았다.

저장소 루트로 이동해 환경을 준비한다. 첫 경로는 자신의 저장소 위치로 바꾼다.

```bash
cd "/path/to/repository"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch transformers numpy pandas scikit-learn soundfile matplotlib tqdm
```

이 설치 예시는 과거 실험 환경을 복원한 lockfile이 아니다. GPU에 맞는 PyTorch 설치를 사용하고 새 실행의 패키지 버전을 보관한다. 최초 backbone 로딩에는 pretrained 모델 다운로드가 필요할 수 있다.

다음 변수에서 **UA-Speech와 TORGO 원본 데이터 경로는 반드시 실제 위치로 변경**한다. 결과·분할 폴더는 첫 실행에서 존재하지 않는 새 경로를 사용한다.

```bash
export CSRL_CODE_ROOT="$PWD"
export CSRL_UA_AUDIO_ROOT="/path/to/UASpeech/audio"
export CSRL_TORGO_ROOT="/path/to/TORGO_raw"
export CSRL_DEVICE="cuda"

export CSRL_INTEGRITY_OUT="$CSRL_CODE_ROOT/1_preprocessing/audio_integrity_run01"
export CSRL_SPLITS_TRAIN="$CSRL_CODE_ROOT/1_preprocessing/splits_train"
export CSRL_SPLITS_TUNING="$CSRL_CODE_ROOT/1_preprocessing/splits_tuning"
export CSRL_TUNE_OUT="$CSRL_CODE_ROOT/2_calibration/results_direct_waveform"
export CSRL_RESULTS="$CSRL_CODE_ROOT/3_training/results"
export CSRL_CONFUSION_OUT="$CSRL_RESULTS/SD/posthoc/confusion_matrix"
export CSRL_TORGO_MANIFEST_OUT="$CSRL_CODE_ROOT/1_preprocessing/torgo_manifest_run01"
export CSRL_TORGO_EVAL_OUT="$CSRL_RESULTS/TORGO_zero_shot_sd_full4_run01"
```

이후 명령은 저장소 루트에서 실행한다. 경로에 공백이 있어도 동작하도록 변수는 따옴표로 감싼다.

## 2. UA-Speech 음성 무결성 검사

필요한 원본 구조는 다음과 같다. `--audio-root`에는 `original`의 **상위 폴더**를 전달한다.

```text
UASpeech/audio/
  original/
    F02/F02_B1_C1_M5.wav
    ...
```

```bash
python 1_preprocessing/1_validate_uaspeech_audio_integrity.py \
  --audio-root "$CSRL_UA_AUDIO_ROOT" \
  --output-dir "$CSRL_INTEGRITY_OUT"
```

확인할 파일은 `combination_summary.csv`, `missing_unexpected.csv`, `duplicates.csv`, `summary.json`이다. 특히 **original/M5 조합의 `passed`와 11,475발화 구성**을 확인한다. 이 검사는 다른 버전·마이크도 조사하므로, 명령이 끝났다는 사실만으로 original/M5 검사가 통과했다고 판단하지 않는다.

이후 분할 생성기는 original/M5의 전체 목록을 사용한다. 별도의 duration outlier 표시가 있더라도 자동 제외하지 않는다. 제공 메타데이터에서 20초 초과로 표시된 36개를 제외하면 논문 수량과 달라진다. 음성은 16 kHz를 요구하며 자동 resampling은 없다.

## 3. 최종 학습용 SD·LOSO 분할 생성

```bash
python 1_preprocessing/2_make_uaspeech_training_splits.py \
  --audio-root "$CSRL_UA_AUDIO_ROOT" \
  --output-dir "$CSRL_SPLITS_TRAIN"
```

이 스크립트는 첨부 metadata CSV를 읽는 대신 **파일명과 원본 폴더를 직접 스캔**한다. 화자·블록·단어 코드·중증도 등을 CSV에 기록한다.

| 출력 | 학습 | 테스트 |
| --- | ---: | ---: |
| `SD/train.csv`, `SD/test.csv` | 6,975 common 발화 | 4,500 uncommon 발화 |
| `LOSO/fold_<화자>/train.csv`, `test.csv` | 6,510 common 발화, 다른 14명 | 300 uncommon 발화, 제외 화자 1명 |

15개 LOSO fold와 `split_summary.csv`, `split_info.json`이 생성된다. Common 단어 ID는 raw code이고, uncommon은 `B1_UW1`처럼 블록을 포함한다. `word_text`가 같다는 이유로 서로 다른 코드를 합치지 않는다.

이미 있는 분할 폴더에는 덮어쓰지 않는다. 새 분할을 만들려면 다른 `--output-dir`을 지정하고 이후 변수도 그 경로로 맞춘다.

## 4. 튜닝용 분할 생성

```bash
python 1_preprocessing/3_make_uaspeech_tuning_splits.py \
  --training-splits-root "$CSRL_SPLITS_TRAIN" \
  --output-dir "$CSRL_SPLITS_TUNING" \
  --seed 42
```

| 프로토콜 | 내부 학습 | 내부 검증 | 분할 방식 |
| --- | ---: | ---: | --- |
| SD | 5,580발화·124단어 | 1,395발화·31단어 | 정렬된 155단어를 seed 42로 무작위 순열 생성 후 31단어 선택. 단어 종류별 할당 없음 |
| LOSO, fold별 | 5,859발화 | 651발화 | 학습 화자 14명 안에서 화자별 비율을 유지한 발화 단위 10% |

SD는 한 단어의 모든 화자·블록 반복 발화를 함께 분리한다. LOSO는 화자별 46개 또는 47개 검증 발화를 선택하며, 같은 단어가 내부 학습·검증 양쪽에 있을 수 있다. 제외 화자의 테스트 데이터는 읽지 않는다.

`word_assignment.csv`는 **SD 단어 배정**이다. LOSO의 검증 발화 목록은 각 fold의 `val.csv`를 확인한다. `split_info.json`은 SD/LOSO 정책을 구분해서 기록한다.

LOSO에서 무작위 분할 뒤 특정 단어·중증도에 학습 화자가 2명 미만이면 해당 그룹을 기록한다. 튜닝 입력 검사는 이를 오류로 알리며, 자동 재추첨이나 단어 제외를 하지 않는다. Validation은 일반 분류 평가이므로 validation 자체에 contrastive episode 구성을 요구하지 않는다.

## 5. 하이퍼파라미터 튜닝 — 선택 단계

현재 고정 설정으로 최종 학습만 실행하려면 이 단계를 생략할 수 있다. 선택 과정을 다시 검토하거나 재실행하려면 다음을 수행한다.

```bash
python 2_calibration/tune_csrl_sd.py \
  --splits-root "$CSRL_SPLITS_TUNING" \
  --output-dir "$CSRL_TUNE_OUT/SD" \
  --device "$CSRL_DEVICE" --max-epochs 50

python 2_calibration/tune_csrl_loso.py \
  --splits-root "$CSRL_SPLITS_TUNING" \
  --output-dir "$CSRL_TUNE_OUT/LOSO" \
  --device "$CSRL_DEVICE" --max-epochs 50
```

- SD: λ 5개 × τ 6개 = **30조합**.
- LOSO: λ 5개 × τ 7개 = **fold당 35조합**, 전체 15개 fold.
- 각 조합은 최대 50에폭이며 검증 macro-F1의 기존 최고값을 3에폭 연속 갱신하지 못하면 종료한다.
- F1 동률에서 CE만 낮아져도 patience는 초기화되지 않는다.
- 먼저 종료된 조합은 건너뛰고 다른 조합은 계속 탐색한다.
- 최상의 모델 선택에는 기존 순위 기준인 macro-F1, CE 등의 tie-break를 사용한다.

한 LOSO fold만 먼저 확인하려면 해당 명령에 `--folds F02`를 추가한다. CUDA AMP 대신 FP32로 실행하려면 튜닝 명령에 `--fp32`를 사용한다. 실행 조건이 달라지면 새 출력 폴더를 사용한다.

출력은 조합별 `last.pt`, `history.json`과 SD 또는 각 fold의 `leaderboard.json`, `selected.json`, `best.pt`다. 선택된 최상 에폭과 실제 종료 에폭(`trained_epochs`)은 다를 수 있다.

**중요:** 이 결과가 최종 학습 코드에 자동으로 전달되지는 않는다. 현재 최종 학습은 아래에 적힌 λ·τ·고정 epoch를 사용해 새 모델을 학습한다. 튜닝 결과와 고정 설정이 다르면 그 차이를 기록해야 하며, 자동으로 선택값이 반영됐다고 해석하지 않는다.

## 6. 최종 학습 입력 검사

아래 네 명령은 최종 학습용 분할을 검사하고 종료한다. `--check-only`도 Python 패키지 import와 실제 CSV·음성 경로 확인을 필요로 하지만, 모델 다운로드·학습은 수행하지 않는다.

```bash
python 3_training/train_sd_baseline.py --splits-root "$CSRL_SPLITS_TRAIN" --check-only
python 3_training/train_sd_csrl.py --splits-root "$CSRL_SPLITS_TRAIN" --check-only
python 3_training/train_loso_baseline.py --splits-root "$CSRL_SPLITS_TRAIN" --check-only
python 3_training/train_loso_csrl.py --splits-root "$CSRL_SPLITS_TRAIN" --check-only
```

## 7. 최종 고정 에폭 학습·평가

| 스크립트 | 고정 epoch | λ | τ |
| --- | ---: | ---: | ---: |
| `train_sd_baseline.py` | 20 | 0 | 사용하지 않음 |
| `train_sd_csrl.py` | 40 | 0.20 | 1.50 |
| `train_loso_baseline.py` | 30 | 0 | 사용하지 않음 |
| `train_loso_csrl.py` | 50 | 0.05 | 0.15 |

```bash
python 3_training/train_sd_baseline.py \
  --splits-root "$CSRL_SPLITS_TRAIN" --output-dir "$CSRL_RESULTS/SD/baseline" \
  --device "$CSRL_DEVICE" --stage all

python 3_training/train_sd_csrl.py \
  --splits-root "$CSRL_SPLITS_TRAIN" --output-dir "$CSRL_RESULTS/SD/csrl" \
  --device "$CSRL_DEVICE" --stage all

python 3_training/train_loso_baseline.py \
  --splits-root "$CSRL_SPLITS_TRAIN" --output-dir "$CSRL_RESULTS/LOSO/baseline" \
  --device "$CSRL_DEVICE" --stage all

python 3_training/train_loso_csrl.py \
  --splits-root "$CSRL_SPLITS_TRAIN" --output-dir "$CSRL_RESULTS/LOSO/csrl" \
  --device "$CSRL_DEVICE" --stage all
```

LOSO의 `--folds F02`는 한 fold만 실행한다. 전체 15-fold 결과와 구분한다. 먼저 일부 fold만 실행했다가 같은 설정으로 전체 fold 명령을 실행하면 호환되는 저장 상태를 사용한다.

`--stage all`은 학습 후 테스트까지 수행한다. `--stage train`은 학습만, `--stage evaluate`는 이미 완성된 고정 최종 checkpoint의 평가만 수행한다. Epoch별 최적 모델을 테스트로 선택하지 않는다.

대표 결과 구조:

```text
3_training/results/
  SD/
    baseline/
      assets/{frozen_encoder.pt,backbone.json}
      config.json
      final_epoch_020.pt
      train_history.json
      test_predictions.csv
      test_metrics.json
    csrl/
      final_epoch_040.pt
      ...
  LOSO/
    baseline/
      assets/
      fold_F02/{config.json,final_epoch_030.pt,test_predictions.csv,test_metrics.json,...}
      ...
      pooled_test_predictions.csv
      pooled_test_metrics.json
    csrl/
      assets/
      fold_F02/{config.json,final_epoch_050.pt,test_predictions.csv,test_metrics.json,...}
      ...
      pooled_test_predictions.csv
      pooled_test_metrics.json
```

중단 후에는 **동일한 명령을 다시 실행**한다. 별도 `--resume` 인자는 없다. 코드·분할·실행 환경 등의 설정이 일치할 때만 재개한다. 변경된 설정이나 예전 정책의 결과 폴더에 이어 쓰지 않고 새 폴더를 사용한다.

## 8. SD 분석

두 SD 모델의 학습과 테스트 예측 생성이 완료된 뒤 실행한다. 분석은 모델을 업데이트하지 않는다.

### 8.1 혼동행렬

저장된 SD 예측을 읽으므로 음성 추론을 다시 하지 않는다.

```bash
python 4_analysis/confusion_matrix.py \
  --result-root "$CSRL_RESULTS/SD" --output-dir "$CSRL_CONFUSION_OUT" --check-only

python 4_analysis/confusion_matrix.py \
  --result-root "$CSRL_RESULTS/SD" --output-dir "$CSRL_CONFUSION_OUT"
```

### 8.2 중증도 실루엣 및 중증도 t-SNE

이 두 파일은 **경로 CLI 옵션이 없다.** 소스에 아래 경로가 고정되어 있다.

```text
RUN_ROOT   = /home/wooje/바탕화면/code/3_training/results/SD
SPLITS_ROOT = /home/wooje/바탕화면/code/1_preprocessing/splits_train/SD
```

위 경로를 그대로 사용하는 연구 컴퓨터에서는 직접 실행한다.

```bash
python 4_analysis/severity_silhouette.py --device "$CSRL_DEVICE" --check-only
python 4_analysis/visualize.py --device "$CSRL_DEVICE" --check-only
python 4_analysis/severity_silhouette.py --device "$CSRL_DEVICE"
python 4_analysis/visualize.py --device "$CSRL_DEVICE"
```

다른 저장소·결과 경로를 사용한다면 **위 직접 실행 대신** 다음 예시로 실행 중의 경로 값만 지정할 수 있다. 파일을 저장하거나 소스 코드를 수정하지 않으며, 모델·분할·분석 파라미터는 바꾸지 않는다. `main` 함수가 참조하는 전역 경로를 지정하므로 사용하지 않는 `--run-root` 인자를 전달하는 방식과 다르다.

```bash
python - <<'PY'
import os
import runpy
from pathlib import Path

code_root = Path(os.environ["CSRL_CODE_ROOT"])
for filename in ("severity_silhouette.py", "visualize.py"):
    namespace = runpy.run_path(
        str(code_root / "4_analysis" / filename),
        run_name="csrl_analysis",
    )
    entry = namespace["main"]
    entry.__globals__["RUN_ROOT"] = Path(os.environ["CSRL_RESULTS"]) / "SD"
    entry.__globals__["SPLITS_ROOT"] = Path(os.environ["CSRL_SPLITS_TRAIN"]) / "SD"
    entry(["--device", os.environ["CSRL_DEVICE"], "--check-only"])
    entry(["--device", os.environ["CSRL_DEVICE"]])
PY
```

출력은 SD 결과 폴더 아래의 `posthoc/sd_true_severity_silhouette_final`과 `posthoc/sd_ground_truth_tsne_final`에 저장된다.

### 8.3 화자 실루엣 및 화자 t-SNE

새 화자 분석 파일은 경로 옵션을 지원한다.

```bash
python 4_analysis/speaker_silhouette.py \
  --run-root "$CSRL_RESULTS/SD" --splits-root "$CSRL_SPLITS_TRAIN/SD" \
  --device "$CSRL_DEVICE" --check-only

python 4_analysis/speaker_silhouette.py \
  --run-root "$CSRL_RESULTS/SD" --splits-root "$CSRL_SPLITS_TRAIN/SD" \
  --device "$CSRL_DEVICE"
```

출력은 `SD/posthoc/sd_true_speaker_structure_final`에 저장된다. 실제 화자 15개를 클러스터 기준으로 사용하고, 4,500개 발화의 원래 768차원 표현에서 cosine 실루엣을 계산한다. 화자별 평균도 함께 기록한다.

중증도·화자 실루엣 모두 특징 추출은 공통 batch 8이다. t-SNE의 2차원 좌표에서 실루엣을 계산하지 않는다. 원래 CE 분류 평가의 batch 16과도 구분한다.

## 9. TORGO manifest 생성 및 검사

예상 디렉터리 구조:

```text
TORGO_raw/
  F/F03/Session1/prompts/<id>.txt
  F/F03/Session1/wav_headMic/<id>.wav
  M/M01/Session1/prompts/<id>.txt
  M/M01/Session1/wav_headMic/<id>.wav
  ...
```

```bash
python 1_preprocessing/4_make_torgo_zero_shot_manifest.py \
  --root "$CSRL_TORGO_ROOT" --audio-subdir wav_headMic \
  --output-dir "$CSRL_TORGO_MANIFEST_OUT" --require-16khz
```

기본 선택 조건은 lexical single-token prompt, `0.05초 초과·20초 이하`, `wav_headMic`이다. 출력 manifest와 summary/checks를 읽고 다음 구성을 확인한다.

| 화자 | 중증도 | 발화 수 |
| --- | --- | ---: |
| F03 | Very low | 401 |
| F04 | Very low | 176 |
| M03 | Very low | 306 |
| F01 | Low | 94 |
| M05 | Low | 352 |
| M01 | Medium | 279 |
| M02 | Medium | 293 |
| M04 | Medium | 207 |
| **합계** | **8명** | **2,108** |

Manifest 생성기의 일반 검사가 통과해도 정확히 이 수량이라는 뜻은 아니다. Zero-shot 평가기는 합계·화자별 개수·레이블·마이크·샘플링 속도를 추가 검사한다. 다르면 필터 조건과 데이터 버전을 확인하며, 수량 검사를 임의로 완화해 같은 실험이라고 표시하지 않는다.

Manifest 생성기는 기존 CSV를 덮어쓸 수 있으므로, 이 단계는 위 예시처럼 **새 출력 폴더**를 사용한다.

## 10. TORGO strict zero-shot 평가

SD 모델 두 개가 완료되어 있어야 한다. `--train-root`는 SD 폴더 자체가 아닌 **SD의 상위 results 폴더**다.

```bash
python 4_analysis/torgo_zero_shot.py \
  --train-root "$CSRL_RESULTS" \
  --manifest "$CSRL_TORGO_MANIFEST_OUT/torgo_zero_shot_manifest.csv" \
  --output-root "$CSRL_TORGO_EVAL_OUT" --device "$CSRL_DEVICE"
```

출력 폴더는 비어 있어야 한다. TORGO로 학습·fine-tuning·calibration·threshold 조정·모델 선택을 수행하지 않는다. 네 출력 클래스를 모두 유지하고, 정답에 없는 high를 예측해도 그대로 오분류에 포함한다.

대표 출력은 `main_metrics.csv`, `all_metrics.csv`, `paired_swcs_minus_ce.csv`, `run_config.json`과 모델별 logits·예측·혼동행렬이다.

## 11. 지표 해석과 실행 기록

- Accuracy/BACC/macro-F1은 JSON/주요 결과 CSV에서 **0~1 비율**이다. 논문 표의 백분율로 비교할 때 100을 곱한다. 혼동행렬 시각화의 row percentage와 단위를 혼동하지 않는다.
- LOSO 전체 성능은 baseline·CSRL 각각의 **`pooled_test_metrics.json`**을 읽는다. 단일 화자 fold의 macro-F1 평균으로 대체하지 않는다.
- 일부 fold만 실행한 결과는 `subset_...`로 저장되며 전체 15-fold 결과가 아니다.
- TORGO의 `macro_f1_gt_classes`는 정답 3클래스 평균, `macro_f1_output_space`는 전체 4클래스 평균이다. 둘 다 full4 argmax를 사용한다. 논문 Table VIII의 macro-F1이 어느 계산에 해당하는지는 원래 평가 기록과 대조해야 한다.
- 분할 CSV·선택 단어/발화 목록·설정 JSON·패키지 버전·모델 가중치를 같은 실행의 산출물로 보관한다. CSV에는 절대 음성 경로가 포함되므로 컴퓨터를 옮기면 원본 학습 시의 해시 조건과 달라질 수 있다.

설치한 환경 버전을 별도로 저장하려면 학습 결과 루트가 생성된 뒤 실행한다.

```bash
python -m pip freeze > "$CSRL_RESULTS/software_versions.txt"
```

현재 저장소에는 end-to-end 재현 완료 결과가 없다. 논문 수치와 비교할 때는 현재 코드로 새로 얻은 결과와 제출본에 보고한 결과를 구분한다.
