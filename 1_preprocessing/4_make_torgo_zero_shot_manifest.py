#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a fixed TORGO isolated-word manifest for strict zero-shot evaluation.

Default input:
    /home/wooje/다운로드/TORGO_raw

Default audio stream:
    wav_headMic

TORGO severity mapping used for evaluation:
    F03, F04, M03 -> very_low (0)
    F01, M05      -> low      (1)
    M01, M02, M04 -> medium   (2)

There is no TORGO ground-truth speaker for class high (3) under this mapping.
IMPORTANT: the UA-Speech model must still keep all four outputs 0/1/2/3.
A class-3 prediction on TORGO is retained as an error; it is never disabled.

Outputs:
    <root>/zero_shot_manifest_wav_headMic/
        torgo_zero_shot_manifest.csv
        torgo_zero_shot_summary.csv
        torgo_zero_shot_excluded.csv
        torgo_zero_shot_checks.csv
        torgo_zero_shot_config.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import soundfile as sf


SPEAKER_LABELS = {
    "F03": ("very_low", 0),
    "F04": ("very_low", 0),
    "M03": ("very_low", 0),
    "F01": ("low", 1),
    "M05": ("low", 1),
    "M01": ("medium", 2),
    "M02": ("medium", 2),
    "M04": ("medium", 2),
}

TARGET_SPEAKERS = tuple(SPEAKER_LABELS.keys())

# English lexical token; allows internal apostrophe / hyphen.
WORD_RE = re.compile(r"^[A-Za-z]+(?:['’\-][A-Za-z]+)*$")

# Explicit non-word placeholders seen in TORGO prompts.
EXCLUDED_SINGLE_TOKENS = {"xxx"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path("/home/wooje/다운로드/TORGO_raw"),
    )
    p.add_argument(
        "--audio-subdir",
        choices=["wav_headMic", "wav_arrayMic"],
        default="wav_headMic",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--min-duration-sec", type=float, default=0.05)
    p.add_argument("--max-duration-sec", type=float, default=20.0)
    p.add_argument(
        "--require-16khz",
        action="store_true",
        help="Reject non-16-kHz WAVs instead of only reporting their rate.",
    )
    return p.parse_args()


def normalize_prompt(text: str) -> str:
    return " ".join(text.strip().split())


def classify_prompt(text: str):
    text = normalize_prompt(text)

    if not text:
        return False, "empty_prompt", None

    if text.casefold() in EXCLUDED_SINGLE_TOKENS:
        return False, "placeholder_prompt", None

    if text.startswith("[") or text.endswith("]"):
        return False, "bracketed_task_prompt", None

    if "/" in text or "\\" in text:
        return False, "path_like_prompt", None

    if len(text.split()) != 1:
        return False, "not_single_token", None

    if not WORD_RE.fullmatch(text):
        return False, "non_lexical_single_token", None

    return True, "kept", text


def inspect_wav(path: Path):
    try:
        info = sf.info(str(path))
        duration = (
            float(info.frames) / float(info.samplerate)
            if info.samplerate else None
        )
        return True, {
            "sample_rate": int(info.samplerate),
            "num_channels": int(info.channels),
            "num_frames": int(info.frames),
            "duration_sec": duration,
            "audio_subtype": str(info.subtype),
            "audio_error": "",
        }
    except Exception as e:
        return False, {
            "sample_rate": None,
            "num_channels": None,
            "num_frames": None,
            "duration_sec": None,
            "audio_subtype": None,
            "audio_error": f"{type(e).__name__}: {e}",
        }


def add_excluded(
    rows,
    speaker,
    session,
    prompt_id,
    prompt_path,
    audio_path,
    prompt_text,
    reason,
    details="",
):
    rows.append({
        "speaker": speaker,
        "session": session,
        "prompt_id": prompt_id,
        "prompt_path": str(prompt_path) if prompt_path else "",
        "audio_path": str(audio_path) if audio_path else "",
        "prompt_text": prompt_text,
        "exclude_reason": reason,
        "details": details,
    })


def build_manifest(
    root: Path,
    audio_subdir: str,
    min_duration_sec: float,
    max_duration_sec: float,
    require_16khz: bool,
):
    kept = []
    excluded = []

    for speaker in TARGET_SPEAKERS:
        severity_name, severity_label = SPEAKER_LABELS[speaker]
        group = "F" if speaker.startswith("F") else "M"
        speaker_dir = root / group / speaker

        if not speaker_dir.is_dir():
            add_excluded(
                excluded, speaker, "", "", "", "",
                "", "speaker_directory_missing", str(speaker_dir)
            )
            continue

        # Includes Session1, Session2, Session2_3, etc.
        prompt_paths = sorted(speaker_dir.glob("*/prompts/*.txt"))

        for prompt_path in prompt_paths:
            session = prompt_path.parent.parent.name
            prompt_id = prompt_path.stem
            audio_path = (
                prompt_path.parent.parent
                / audio_subdir
                / f"{prompt_id}.wav"
            )

            try:
                raw_prompt = prompt_path.read_text(
                    encoding="utf-8", errors="ignore"
                )
            except Exception as e:
                add_excluded(
                    excluded, speaker, session, prompt_id,
                    prompt_path.resolve(), audio_path,
                    "", "prompt_read_error",
                    f"{type(e).__name__}: {e}"
                )
                continue

            keep, reason, word = classify_prompt(raw_prompt)

            if not keep:
                add_excluded(
                    excluded, speaker, session, prompt_id,
                    prompt_path.resolve(), audio_path,
                    normalize_prompt(raw_prompt), reason
                )
                continue

            if not audio_path.is_file():
                add_excluded(
                    excluded, speaker, session, prompt_id,
                    prompt_path.resolve(), audio_path,
                    word, "matching_wav_missing"
                )
                continue

            ok, info = inspect_wav(audio_path)
            if not ok:
                add_excluded(
                    excluded, speaker, session, prompt_id,
                    prompt_path.resolve(), audio_path.resolve(),
                    word, "wav_open_failed", info["audio_error"]
                )
                continue

            duration = info["duration_sec"]
            if duration is None:
                add_excluded(
                    excluded, speaker, session, prompt_id,
                    prompt_path.resolve(), audio_path.resolve(),
                    word, "duration_missing"
                )
                continue

            if duration <= min_duration_sec:
                add_excluded(
                    excluded, speaker, session, prompt_id,
                    prompt_path.resolve(), audio_path.resolve(),
                    word, "duration_too_short",
                    f"{duration:.6f}"
                )
                continue

            if duration > max_duration_sec:
                add_excluded(
                    excluded, speaker, session, prompt_id,
                    prompt_path.resolve(), audio_path.resolve(),
                    word, "duration_too_long",
                    f"{duration:.6f}"
                )
                continue

            if require_16khz and info["sample_rate"] != 16000:
                add_excluded(
                    excluded, speaker, session, prompt_id,
                    prompt_path.resolve(), audio_path.resolve(),
                    word, "sample_rate_not_16khz",
                    str(info["sample_rate"])
                )
                continue

            kept.append({
                "utterance_id": (
                    f"{speaker}_{session}_{prompt_id}_{audio_subdir}"
                ),
                "path": str(audio_path.resolve()),
                "speaker": speaker,
                "gender": speaker[0],
                "session": session,
                "prompt_id": prompt_id,
                "word_text": word,
                "word_key": word.casefold(),
                "severity_name": severity_name,
                "severity_label": severity_label,
                "audio_subdir": audio_subdir,
                "sample_rate": info["sample_rate"],
                "num_channels": info["num_channels"],
                "num_frames": info["num_frames"],
                "duration_sec": round(float(duration), 6),
                "audio_subtype": info["audio_subtype"],
            })

    manifest = pd.DataFrame(kept)
    excluded_df = pd.DataFrame(excluded)

    if not manifest.empty:
        manifest = manifest.sort_values(
            [
                "severity_label", "speaker",
                "session", "prompt_id", "utterance_id"
            ],
            kind="stable",
        ).reset_index(drop=True)

    if not excluded_df.empty:
        excluded_df = excluded_df.sort_values(
            ["speaker", "session", "prompt_id", "exclude_reason"],
            kind="stable",
        ).reset_index(drop=True)

    return manifest, excluded_df


def validate_and_summarize(manifest: pd.DataFrame, audio_subdir: str):
    checks = []

    def add(name, observed, expected, passed):
        checks.append({
            "check": name,
            "observed": observed,
            "expected": expected,
            "passed": bool(passed),
        })

    if manifest.empty:
        add("manifest_nonempty", 0, ">0", False)
        return pd.DataFrame(checks), pd.DataFrame()

    observed_speakers = sorted(manifest["speaker"].unique().tolist())
    expected_speakers = sorted(TARGET_SPEAKERS)

    add(
        "all_8_dysarthric_speakers_present",
        ",".join(observed_speakers),
        ",".join(expected_speakers),
        observed_speakers == expected_speakers,
    )

    labels = sorted(manifest["severity_label"].unique().tolist())
    add(
        "ground_truth_labels_are_0_1_2",
        ",".join(map(str, labels)),
        "0,1,2",
        labels == [0, 1, 2],
    )

    add(
        "no_ground_truth_label_3",
        int((manifest["severity_label"] == 3).sum()),
        0,
        int((manifest["severity_label"] == 3).sum()) == 0,
    )

    add(
        "unique_utterance_id",
        manifest["utterance_id"].nunique(),
        len(manifest),
        manifest["utterance_id"].nunique() == len(manifest),
    )

    add(
        "unique_audio_path",
        manifest["path"].nunique(),
        len(manifest),
        manifest["path"].nunique() == len(manifest),
    )

    add(
        "all_selected_audio_exists",
        int(manifest["path"].map(lambda x: Path(x).is_file()).sum()),
        len(manifest),
        bool(manifest["path"].map(lambda x: Path(x).is_file()).all()),
    )

    add(
        "single_audio_stream",
        ",".join(sorted(manifest["audio_subdir"].unique())),
        audio_subdir,
        set(manifest["audio_subdir"].unique()) == {audio_subdir},
    )

    mapping_ok = True
    for speaker, (name, label) in SPEAKER_LABELS.items():
        g = manifest[manifest["speaker"] == speaker]
        if (
            g.empty
            or set(g["severity_name"]) != {name}
            or set(g["severity_label"]) != {label}
        ):
            mapping_ok = False
            break

    add(
        "speaker_to_severity_mapping_exact",
        mapping_ok,
        True,
        mapping_ok,
    )

    summary_rows = []
    for speaker in TARGET_SPEAKERS:
        g = manifest[manifest["speaker"] == speaker]
        name, label = SPEAKER_LABELS[speaker]

        summary_rows.append({
            "speaker": speaker,
            "severity_name": name,
            "severity_label": label,
            "utterances": len(g),
            "unique_words": int(g["word_key"].nunique()),
            "sessions": int(g["session"].nunique()),
            "duration_mean_sec": (
                round(float(g["duration_sec"].mean()), 6)
                if not g.empty else None
            ),
            "sample_rates": (
                json.dumps(
                    dict(Counter(g["sample_rate"].astype(int))),
                    sort_keys=True
                )
                if not g.empty else "{}"
            ),
            "channels": (
                json.dumps(
                    dict(Counter(g["num_channels"].astype(int))),
                    sort_keys=True
                )
                if not g.empty else "{}"
            ),
        })

    summary = pd.DataFrame(summary_rows)

    total = pd.DataFrame([{
        "speaker": "__TOTAL__",
        "severity_name": "",
        "severity_label": "",
        "utterances": len(manifest),
        "unique_words": int(manifest["word_key"].nunique()),
        "sessions": int(
            manifest[["speaker", "session"]].drop_duplicates().shape[0]
        ),
        "duration_mean_sec": round(
            float(manifest["duration_sec"].mean()), 6
        ),
        "sample_rates": json.dumps(
            dict(Counter(manifest["sample_rate"].astype(int))),
            sort_keys=True
        ),
        "channels": json.dumps(
            dict(Counter(manifest["num_channels"].astype(int))),
            sort_keys=True
        ),
    }])

    summary = pd.concat([summary, total], ignore_index=True)
    return pd.DataFrame(checks), summary


def main():
    args = parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"TORGO root not found: {root}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / f"zero_shot_manifest_{args.audio_subdir}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, excluded = build_manifest(
        root=root,
        audio_subdir=args.audio_subdir,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
        require_16khz=args.require_16khz,
    )

    checks, summary = validate_and_summarize(
        manifest, args.audio_subdir
    )

    manifest_path = output_dir / "torgo_zero_shot_manifest.csv"
    summary_path = output_dir / "torgo_zero_shot_summary.csv"
    excluded_path = output_dir / "torgo_zero_shot_excluded.csv"
    checks_path = output_dir / "torgo_zero_shot_checks.csv"
    config_path = output_dir / "torgo_zero_shot_config.json"

    manifest.to_csv(
        manifest_path, index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )
    excluded.to_csv(
        excluded_path, index=False, encoding="utf-8-sig"
    )
    checks.to_csv(
        checks_path, index=False, encoding="utf-8-sig"
    )

    config = {
        "protocol": "UA-Speech-trained -> TORGO strict zero-shot",
        "torgo_training": False,
        "torgo_finetuning": False,
        "torgo_calibration": False,
        "torgo_model_selection": False,
        "audio_subdir": args.audio_subdir,
        "speech_material": "clean isolated lexical words only",
        "speaker_labels": {
            s: {
                "severity_name": SPEAKER_LABELS[s][0],
                "severity_label": SPEAKER_LABELS[s][1],
            }
            for s in TARGET_SPEAKERS
        },
        "torgo_ground_truth_classes_present": [0, 1, 2],
        "ua_speech_model_outputs_to_keep": [0, 1, 2, 3],
        "class_3_policy": (
            "Do not disable class 3. If predicted, retain it and "
            "count it as an error."
        ),
    }

    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("TORGO ZERO-SHOT MANIFEST")
    print("=" * 80)
    print("Root         :", root)
    print("Audio stream :", args.audio_subdir)
    print("Kept         :", len(manifest))
    print("Excluded     :", len(excluded))
    print("Manifest     :", manifest_path)
    print()

    print("===== SPEAKER SUMMARY =====")
    print(summary.to_string(index=False))
    print()

    print("===== EXCLUSION REASONS =====")
    if excluded.empty:
        print("None")
    else:
        print(
            excluded["exclude_reason"]
            .value_counts()
            .rename_axis("reason")
            .reset_index(name="count")
            .to_string(index=False)
        )
    print()

    print("===== CHECKS =====")
    print(checks.to_string(index=False))
    print()

    if checks.empty or not bool(checks["passed"].all()):
        print(
            f"ERROR: validation failed. See {checks_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("All checks passed.")
    print(
        "Class 3 remains enabled in the UA-Speech model during zero-shot inference."
    )


if __name__ == "__main__":
    main()
