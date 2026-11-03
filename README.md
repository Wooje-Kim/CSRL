# CSRL: Cross-Speaker Severity Relation Learning

Code accompanying **Lexically Conditioned Contrastive Learning for Dysarthria Severity Classification**.

Wooje Kim, Dahee Oh, Sungdae Na, and Seong Tak Woo.

> **Release status:** The scripts have been organized and corrected against the manuscript design. Static and synthetic checks have been performed, but end-to-end training, analysis, and numerical reproduction of the submitted results have not yet been completed. This repository should not be treated as a verified recovery of the exact original experiment artifacts. The historically tested environment and original random validation memberships have not been recovered.

For step-by-step commands, see **[실행 가이드 / Korean run guide](RUN_GUIDE_KO.md)**.

## Method

CSRL learns severity relations across speakers under a common lexical condition.

- One episode contains eight utterances of the **same word**: two different speakers from each of four severity classes.
- Each anchor has one same-severity, different-speaker positive and six different-severity negatives. Self-comparison is excluded.
- Each optimizer update uses two distinct words. The two word-specific contrastive losses are calculated independently and averaged; there are no cross-word contrastive pairs.
- Cross-entropy and CSRL use the **same episode utterances** and the same post-ReLU classifier latent representation. No projection head is added.
- Inference uses the standard four-class classifier without word or speaker metadata.

The model consists of a frozen `facebook/wav2vec2-base` encoder, valid-frame mean pooling, and a trainable `Linear(768, 768) → ReLU → Linear(768, 4)` head. CSRL is applied to the 768-dimensional representation before the last linear layer.

The CE-only baseline also uses the structured two-word sampler. It forwards all 16 utterances together. CSRL forwards two separate eight-utterance episodes, averages their total losses, and performs one optimizer update. Their padding scopes differ.

## Repository structure

```text
.
├── README.md
├── RUN_GUIDE_KO.md
├── 1_preprocessing/
│   ├── 1_validate_uaspeech_audio_integrity.py
│   ├── 2_make_uaspeech_training_splits.py
│   ├── 3_make_uaspeech_tuning_splits.py
│   └── 4_make_torgo_zero_shot_manifest.py
├── 2_calibration/
│   ├── tune_csrl_sd.py
│   └── tune_csrl_loso.py
├── 3_training/
│   ├── train_sd_baseline.py
│   ├── train_sd_csrl.py
│   ├── train_loso_baseline.py
│   └── train_loso_csrl.py
└── 4_analysis/
    ├── confusion_matrix.py
    ├── severity_silhouette.py
    ├── speaker_silhouette.py
    ├── visualize.py
    └── torgo_zero_shot.py
```

`2_calibration` performs hyperparameter search; it does not calibrate predictions on TORGO.

## Environment

The scripts use Python and the following external packages:

```text
torch, transformers, numpy, pandas, scikit-learn, soundfile, matplotlib, tqdm
```

Training/tuning locks use `fcntl`, so the current scripts target Linux or macOS. Training defaults to CUDA and also supports CPU. CPU feasibility and GPU memory requirements have not been benchmarked. There is no verified environment lockfile in this release.

Example setup, to be validated on the research machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch transformers numpy pandas scikit-learn soundfile matplotlib tqdm
```

Use a PyTorch installation appropriate for the intended hardware. Initial backbone loading may download pretrained weights. Current final-training scripts do not pin a Hugging Face model revision; saved backbone assets and recorded environment information should be retained with the results.

## Data and protocols

Recordings are not distributed with the source. Obtain [UA-Speech from its provider](https://speechtechnology.web.illinois.edu/uaspeech/) and [TORGO from its provider](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html), and follow their access and use terms. In particular, the UA-Speech provider prohibits redistribution of its recordings.

### UA-Speech

The implementation uses **15 fixed speakers with dysarthria, original audio, microphone M5**, totaling 11,475 utterances. Control speakers are excluded. Input audio must be 16 kHz; the scripts do not automatically resample it.

| Stage | Training | Validation or testing |
| --- | --- | --- |
| Final speaker-dependent (SD) | 6,975 common-word utterances, all 15 speakers | 4,500 uncommon-word utterances, the same speakers |
| Final LOSO, per fold | 6,510 common-word utterances, the other 14 speakers | 300 uncommon-word utterances, the held-out speaker |
| SD tuning | 124 common-word identities, 5,580 utterances | 31 word identities, 1,395 utterances |
| LOSO tuning, per fold | 5,859 common-word utterances | 651 common-word utterances, speaker-stratified |

SD tuning shuffles the sorted 155 common-word codes using seed 42 and selects 31 validation words without word-type quotas. Every speaker and repetition of a selected word stays in the same split. LOSO tuning uses a 10% utterance split, with seven training speakers contributing 46 validation utterances and seven contributing 47. Training and validation may contain the same word identities under LOSO tuning. Held-out test speakers never enter that tuning split.

The final-training split generator scans the recording filenames under `original/`; it does not read a pre-existing metadata CSV. It retains the complete original/M5 inventory, including long recordings, rather than filtering duration-outlier flags. Common words use their raw code as identity; uncommon words use block-qualified codes such as `B1_UW1`. Identical word strings are not used to merge different inventory codes.

Severity labels increase in severity: `0=very_low`, `1=low`, `2=medium`, `3=high`.

| Severity | UA-Speech speakers |
| --- | --- |
| Very low | F05, M08, M09, M10, M14 |
| Low | F04, M05, M11 |
| Medium | F02, M07, M16 |
| High | F03, M01, M04, M12 |

### TORGO zero-shot evaluation

The evaluator applies the **SD-trained models** to a fixed 2,108-utterance, eight-speaker subset using `wav_headMic`. Its ground-truth categories are very low, low, and medium. All four source outputs are retained, and a prediction of high severity counts as an error.

The manifest builder selects lexical single-token prompts and, by default, recordings with `0.05 < duration ≤ 20` seconds. Use `--require-16khz` to enforce the expected rate. Its general validity checks do not establish the exact 2,108-row subset; the evaluator additionally checks the total and speaker-level counts. See the run guide for those counts.

TORGO is not used for training, fine-tuning, threshold tuning, calibration, or model selection. Its clinical category definitions differ from UA-Speech, which limits interpretation of cross-corpus severity correspondence.

## Settings encoded in the current scripts

| Setting | SD | LOSO |
| --- | --- | --- |
| CSRL λ | 0.20 | 0.05 |
| CSRL temperature τ | 1.50 | 0.15 |
| Final CE-only epochs | **20, fixed** | **30, fixed** |
| Final CSRL epochs | **40, fixed** | **50, fixed** |
| Tuning maximum epochs | 50 | 50 |
| Tuning early stopping | 3 consecutive epochs without exceeding the best validation macro-F1 seen so far | Same |

Both methods use Adam, learning rate `5e-4`, betas `(0.9, 0.98)`, weight decay 0, and gradient clipping at 1. The base seed is 42.

An epoch is `ceil(N_train / 8)` structured updates, with 16 sampled utterance exposures per update. Final SD training therefore uses 872 updates/13,952 exposures per epoch, and final LOSO uses 814 updates/13,024 exposures per epoch. It is not a single pass through all utterances. Different fixed epoch counts imply different total training budgets.

The default tuning grid has five λ values and six temperatures for SD (30 combinations), and seven temperatures for LOSO (35 combinations per fold). Each combination stops independently. Equal macro-F1 with a lower validation CE does not reset patience. All combinations are considered, and the existing validation ranking selects the saved best model.

**Tuning outputs are not automatically applied to final training.** Final scripts use their own fixed constants and initialize new models. `selected.json`/`best.pt` from tuning are not read by them. Do not describe tuning-selected epochs as the fixed final-training epochs.

## Execution order

1. Check the UA-Speech audio inventory and inspect the original/M5 report.
2. Generate final-training SD/LOSO splits.
3. Generate SD/LOSO tuning splits.
4. Run SD/LOSO hyperparameter search if auditing or repeating the selection stage.
5. Check the final-training inputs, then run the four fixed-epoch training scripts.
6. Analyze completed SD models: confusion matrices, severity/speaker silhouettes, and t-SNE.
7. Build/check the TORGO manifest and run strict zero-shot evaluation.

Detailed commands and output locations are in [RUN_GUIDE_KO.md](RUN_GUIDE_KO.md). Two analysis scripts, `severity_silhouette.py` and `visualize.py`, contain fixed path constants and do not accept path flags. The guide includes a path-only runtime override for using them outside the original Linux directory layout without editing source files.

Compatible training/tuning runs resume by repeating the same command; there is no `--resume` option. Changed source, configurations, or split hashes require a new output directory. Split generators require a new destination, and TORGO evaluation requires an empty destination.

## Reading results

- Final SD metrics: `3_training/results/SD/{baseline,csrl}/test_metrics.json`.
- Full LOSO metrics: `3_training/results/LOSO/{baseline,csrl}/pooled_test_metrics.json`, pooling all 4,500 predictions. A subset or a mean of single-speaker fold macro-F1 values is not the full LOSO result.
- Accuracy, balanced accuracy, and macro-F1 values in metric JSON are **fractions**, not percentages.
- TORGO reports both `macro_f1_gt_classes` (three true classes) and `macro_f1_output_space` (all four output classes). Both use the unchanged four-class argmax; neither masks the high-severity output.
- Severity and speaker silhouette analyses use the raw post-ReLU 768-dimensional representation and cosine distance on the same SD test set. PCA/t-SNE is used only for visualization. These post-hoc analyses share batch-8 encoder features; classification evaluation uses CE batch 16/CSRL batch 8, and TORGO uses batch 4.
- Lower speaker silhouette indicates weaker speaker-wise clustering, not demonstrated removal of speaker information.

## Manuscript reference results

These are **reported manuscript accuracy values, not results reproduced by the current release**.

| Evaluation | CE-only (%) | CSRL (%) |
| --- | ---: | ---: |
| UA-Speech SD | 81.71 | 90.62 |
| UA-Speech LOSO | 33.69 | 37.76 |
| TORGO zero-shot | 46.39 | 52.28 |

## Paper and license

Manuscript: *Lexically Conditioned Contrastive Learning for Dysarthria Severity Classification*, Wooje Kim, Dahee Oh, Sungdae Na, and Seong Tak Woo. Submitted to IEEE Access. No verified publication DOI or public paper URL is supplied here.

A repository license has not yet been specified. Dataset and pretrained-model terms are separate from the repository's eventual code license.
