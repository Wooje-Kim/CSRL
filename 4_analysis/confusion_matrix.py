#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Final SD CE-only vs CSRL confusion-matrix analysis

===================================================



This script uses the EXACT saved UA-Speech SD TEST predictions produced by the

current final training run. It does not retrain or re-run audio inference.



Current final models

--------------------

CE-only:

    baseline/final_epoch_020.pt

CSRL:

    csrl/final_epoch_040.pt

    lambda=0.20, temperature=1.50



Data

----

SD TEST = 4,500 uncommon-word utterances from the same 15 speakers.

Ground-truth class support:

    very_low=1,500, low=900, medium=900, high=1,200



Main paper figure

-----------------

Two row-normalized confusion matrices are drawn with the SAME 0-100% scale:

    (a) CE-only

    (b) CSRL

Each cell contains row-normalized percentage and raw count by default.



Run

---

    python3 confusion_matrix_sd_final.py



Validation only

---------------

    python3 confusion_matrix_sd_final.py --check-only

"""



from __future__ import annotations



import argparse

import hashlib

import json

import math

from pathlib import Path

from typing import Any, Dict, Mapping, Optional, Tuple



import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.colors import Normalize

import numpy as np

import pandas as pd

import torch

from sklearn.metrics import (

    accuracy_score,

    balanced_accuracy_score,

    confusion_matrix,

    f1_score,

    precision_recall_fscore_support,

)





DEFAULT_RESULT_ROOT = Path('/home/wooje/바탕화면/code/3_training/results/SD')


# Save the final paper figure and diagnostic files directly to the desktop.
DEFAULT_OUTPUT_DIR = Path.home() / "바탕화면"


LABEL_IDS = [0, 1, 2, 3]

LABEL_KEYS = ["very_low", "low", "medium", "high"]

LABEL_DISPLAY = ["Very low", "Low", "Medium", "High"]

EXPECTED_TEST_ROWS = 4_500

EXPECTED_TRUE_COUNTS = {0: 1_500, 1: 900, 2: 900, 3: 1_200}

EXPECTED_METHODS = {

    "ce_only": {

        "display_name": "CE-only",

        "epoch": 20,

        "checkpoint": "final_epoch_020.pt",

        "swc_enabled": False,

        "swc_lambda": 0.0,

        "swc_temperature": None,

    },

    "swcs": {

        "display_name": "CSRL",

        "epoch": 40,

        "checkpoint": "final_epoch_040.pt",

        "swc_enabled": True,

        "swc_lambda": 0.20,

        "swc_temperature": 1.50,

    },

}





def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(

        description="Create paper-ready confusion matrices for final SD CE-only and CSRL models."

    )

    parser.add_argument(

        "--result-root",

        type=Path,

        default=DEFAULT_RESULT_ROOT,

        help="Final SD experiment root.",

    )

    parser.add_argument(

        "--output-dir",

        type=Path,

        default=DEFAULT_OUTPUT_DIR,

        help="Output directory. Default: ~/바탕화면",

    )

    parser.add_argument(

        "--annotation",

        choices=["percent_count", "percent", "count"],

        default="percent_count",

        help="Text shown in each heatmap cell.",

    )

    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("--check-only", action="store_true")

    return parser.parse_args()





def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(lambda: file.read(1024 * 1024), b""):

            digest.update(chunk)

    return digest.hexdigest()





def load_json(path: Path) -> Dict[str, Any]:

    if not path.is_file():

        raise FileNotFoundError(path)

    obj = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(obj, dict):

        raise RuntimeError(f"{path}: JSON object expected")

    return obj





def save_json(obj: Mapping[str, Any], path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(

        json.dumps(dict(obj), ensure_ascii=False, indent=2),

        encoding="utf-8",

    )





def save_csv(frame: pd.DataFrame, path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_csv(path, index=True, encoding="utf-8-sig")





def torch_load_cpu(path: Path) -> Mapping[str, Any]:

    try:

        obj = torch.load(path, map_location="cpu", weights_only=False)

    except TypeError:

        obj = torch.load(path, map_location="cpu")

    if not isinstance(obj, Mapping):

        raise RuntimeError(f"{path}: checkpoint mapping expected")

    return obj





def close_enough(actual: Any, expected: Any, atol: float = 1e-12) -> bool:

    if expected is None:

        return actual is None

    try:

        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol)

    except (TypeError, ValueError):

        return False





def verify_evaluation_hashes(method_dir: Path) -> None:

    complete_path = method_dir / "evaluation_complete.json"

    if not complete_path.is_file():

        return



    complete = load_json(complete_path)

    file_hashes = complete.get("files_sha256", {})

    if not isinstance(file_hashes, dict):

        raise RuntimeError(f"{complete_path}: files_sha256 must be an object")



    for name in ["test_predictions.csv", "test_metrics.json"]:

        expected_hash = file_hashes.get(name)

        if expected_hash is None:

            raise RuntimeError(f"{complete_path}: missing hash for {name}")

        path = method_dir / name

        if not path.is_file():

            raise FileNotFoundError(path)

        actual_hash = sha256_file(path)

        if actual_hash != str(expected_hash):

            raise RuntimeError(f"Saved final evaluation file changed: {path}")





def validate_checkpoint(method_dir: Path, method: str) -> Dict[str, Any]:

    expected = EXPECTED_METHODS[method]

    checkpoint_path = method_dir / expected["checkpoint"]

    if not checkpoint_path.is_file():

        raise FileNotFoundError(checkpoint_path)



    checkpoint = torch_load_cpu(checkpoint_path)
    cfg = checkpoint['config']
    checkpoint = dict(checkpoint, method=cfg['method'], protocol=cfg['protocol'], swc_enabled=cfg['method']=='swcs', swc_lambda=cfg['lambda'], swc_temperature=cfg['temperature'])




    checks = {

        "method": checkpoint.get("method") == method,

        "protocol": checkpoint.get("protocol") == "SD",

        "epoch": int(checkpoint.get("epoch", -1)) == int(expected["epoch"]),

        "swc_enabled": bool(checkpoint.get("swc_enabled", False))

        == bool(expected["swc_enabled"]),

        "swc_lambda": close_enough(

            checkpoint.get("swc_lambda"), expected["swc_lambda"]

        ),

        "swc_temperature": close_enough(

            checkpoint.get("swc_temperature"), expected["swc_temperature"]

        ),

        "model_state_dict": isinstance(checkpoint.get("model_state_dict"), Mapping),

    }

    failed = [name for name, passed in checks.items() if not passed]

    if failed:

        raise RuntimeError(

            f"{checkpoint_path}: checkpoint contract mismatch: {failed}"

        )



    return {

        "path": str(checkpoint_path),

        "sha256": sha256_file(checkpoint_path),

        "epoch": int(expected["epoch"]),

        "method": method,

        "swc_enabled": bool(expected["swc_enabled"]),

        "swc_lambda": expected["swc_lambda"],

        "swc_temperature": expected["swc_temperature"],

    }





def validate_predictions(

    result_root: Path,

    method: str,

) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:

    expected = EXPECTED_METHODS[method]

    method_dir = result_root / {'ce_only':'baseline', 'swcs':'csrl'}[method]

    if not method_dir.is_dir():

        raise FileNotFoundError(method_dir)



    verify_evaluation_hashes(method_dir)

    checkpoint_info = validate_checkpoint(method_dir, method)



    predictions_path = method_dir / "test_predictions.csv"

    metrics_path = method_dir / "test_metrics.json"

    if not predictions_path.is_file():

        raise FileNotFoundError(

            f"{predictions_path}\nFinal SD evaluation predictions are required."

        )

    if not metrics_path.is_file():

        raise FileNotFoundError(metrics_path)



    frame = pd.read_csv(predictions_path)

    required = {"utterance_id", "true_label", "pred_label"}

    missing = sorted(required - set(frame.columns))

    if missing:

        raise RuntimeError(f"{predictions_path}: missing columns {missing}")



    if len(frame) != EXPECTED_TEST_ROWS:

        raise RuntimeError(

            f"{predictions_path}: rows={len(frame)}, expected={EXPECTED_TEST_ROWS}"

        )

    if frame["utterance_id"].astype(str).duplicated().any():

        raise RuntimeError(f"{predictions_path}: duplicated utterance_id")



    frame["utterance_id"] = frame["utterance_id"].astype(str)

    frame["true_label"] = pd.to_numeric(frame["true_label"], errors="raise").astype(int)

    frame["pred_label"] = pd.to_numeric(frame["pred_label"], errors="raise").astype(int)



    if not set(frame["true_label"].unique()).issubset(set(LABEL_IDS)):

        raise RuntimeError(f"{predictions_path}: invalid true labels")

    if not set(frame["pred_label"].unique()).issubset(set(LABEL_IDS)):

        raise RuntimeError(f"{predictions_path}: invalid predicted labels")



    actual_counts = {

        int(k): int(v)

        for k, v in frame["true_label"].value_counts().sort_index().items()

    }

    if actual_counts != EXPECTED_TRUE_COUNTS:

        raise RuntimeError(

            "SD TEST ground-truth class counts mismatch: "

            f"actual={actual_counts}, expected={EXPECTED_TRUE_COUNTS}"

        )



    if "method" in frame.columns:

        values = set(frame["method"].astype(str).unique())

        if values != {method}:

            raise RuntimeError(

                f"{predictions_path}: method column={values}, expected={method}"

            )

    if "checkpoint_epoch" in frame.columns:

        epochs = set(pd.to_numeric(frame["checkpoint_epoch"], errors="raise").astype(int))

        if epochs != {int(expected["epoch"])}:

            raise RuntimeError(

                f"{predictions_path}: checkpoint_epoch={epochs}, expected={expected['epoch']}"

            )



    probability_columns = [f"prob_{name}" for name in LABEL_KEYS]

    if all(column in frame.columns for column in probability_columns):

        probabilities = frame[probability_columns].to_numpy(dtype=float)

        if not np.isfinite(probabilities).all():

            raise RuntimeError(f"{predictions_path}: NaN/Inf probabilities")

        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):

            raise RuntimeError(f"{predictions_path}: class probabilities do not sum to 1")

        probability_argmax = probabilities.argmax(axis=1)

        if not np.array_equal(probability_argmax, frame["pred_label"].to_numpy()):

            raise RuntimeError(f"{predictions_path}: pred_label != probability argmax")



    matrix = confusion_matrix(

        frame["true_label"].to_numpy(),

        frame["pred_label"].to_numpy(),

        labels=LABEL_IDS,

    ).astype(np.int64)



    metrics = load_json(metrics_path)

    if int(metrics.get("sample_count", -1)) != EXPECTED_TEST_ROWS:

        raise RuntimeError(f"{metrics_path}: sample_count mismatch")

    stored_matrix = np.asarray(metrics.get("confusion_matrix"), dtype=np.int64)

    if stored_matrix.shape != (4, 4) or not np.array_equal(stored_matrix, matrix):

        raise RuntimeError(

            f"{metrics_path}: stored confusion matrix differs from predictions"

        )



    return frame, matrix, {

        "display_name": expected["display_name"],

        "predictions_path": str(predictions_path),

        "predictions_sha256": sha256_file(predictions_path),

        "metrics_path": str(metrics_path),

        "metrics_sha256": sha256_file(metrics_path),

        "checkpoint": checkpoint_info,

    }





def row_normalized_percent(matrix: np.ndarray) -> np.ndarray:

    row_sums = matrix.sum(axis=1, keepdims=True)

    if np.any(row_sums <= 0):

        raise RuntimeError("Confusion matrix has an empty ground-truth class row")

    return matrix.astype(np.float64) / row_sums * 100.0





def class_summary(frame: pd.DataFrame, method: str) -> pd.DataFrame:

    y_true = frame["true_label"].to_numpy(dtype=np.int64)

    y_pred = frame["pred_label"].to_numpy(dtype=np.int64)

    precision, recall, f1, support = precision_recall_fscore_support(

        y_true,

        y_pred,

        labels=LABEL_IDS,

        zero_division=0,

    )

    return pd.DataFrame(

        {

            "method": method,

            "class_id": LABEL_IDS,

            "class_name": LABEL_KEYS,

            "support": support.astype(int),

            "precision": precision,

            "recall": recall,

            "f1": f1,

        }

    )





def aggregate_summary(frame: pd.DataFrame, method: str) -> Dict[str, Any]:

    y_true = frame["true_label"].to_numpy(dtype=np.int64)

    y_pred = frame["pred_label"].to_numpy(dtype=np.int64)

    return {

        "method": method,

        "samples": int(len(frame)),

        "accuracy": float(accuracy_score(y_true, y_pred)),

        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),

        "macro_f1": float(

            f1_score(y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0)

        ),

    }





def annotation_text(percent: float, count: int, mode: str) -> str:

    if mode == "percent_count":

        return f"{percent:.1f}%\n({count:,})"

    if mode == "percent":

        return f"{percent:.1f}%"

    if mode == "count":

        return f"{count:,}"

    raise ValueError(mode)





def text_color_for_value(value: float, image) -> str:

    rgba = image.cmap(image.norm(value))

    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]

    return "black" if luminance > 0.55 else "white"





def draw_matrix(

    ax,

    matrix: np.ndarray,

    percent: np.ndarray,

    title: str,

    annotation_mode: str,

    norm: Normalize,

):

    image = ax.imshow(percent, norm=norm, aspect="equal")



    ax.set_title(title, fontsize=12, pad=10)

    ax.set_xlabel("Predicted severity", fontsize=10)

    ax.set_ylabel("Ground-truth severity", fontsize=10)

    ax.set_xticks(np.arange(4), labels=LABEL_DISPLAY)

    ax.set_yticks(np.arange(4), labels=LABEL_DISPLAY)

    ax.tick_params(axis="x", labelrotation=25, labelsize=9)

    ax.tick_params(axis="y", labelsize=9)



    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)

    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)

    ax.grid(which="minor", linewidth=0.8)

    ax.tick_params(which="minor", bottom=False, left=False)



    for row in range(4):

        for col in range(4):

            value = float(percent[row, col])

            ax.text(

                col,

                row,

                annotation_text(value, int(matrix[row, col]), annotation_mode),

                ha="center",

                va="center",

                fontsize=8.5,

                color=text_color_for_value(value, image),

            )



    return image





def make_figure(

    ce_matrix: np.ndarray,

    swcs_matrix: np.ndarray,

    output_dir: Path,

    annotation_mode: str,

    dpi: int,

) -> Path:

    ce_percent = row_normalized_percent(ce_matrix)

    swcs_percent = row_normalized_percent(swcs_matrix)



    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)

    norm = Normalize(vmin=0.0, vmax=100.0)



    image = draw_matrix(

        axes[0],

        ce_matrix,

        ce_percent,

        "(a) CE-only",

        annotation_mode,

        norm,

    )

    draw_matrix(

        axes[1],

        swcs_matrix,

        swcs_percent,

        "(b) CSRL",

        annotation_mode,

        norm,

    )



    colorbar = figure.colorbar(image, ax=axes, fraction=0.035, pad=0.02)

    colorbar.set_label("Row-normalized percentage (%)", fontsize=10)

    colorbar.ax.tick_params(labelsize=9)



    png_path = output_dir / "sd_ce_vs_csrl_confusion_matrix.png"

    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    plt.close(figure)

    return png_path





def main() -> None:

    args = parse_args()

    result_root = args.result_root.expanduser().resolve()

    output_dir = args.output_dir.expanduser().resolve()



    if not result_root.is_dir():

        raise FileNotFoundError(result_root)



    print("Final SD prediction/checkpoint validation...")

    ce_frame, ce_matrix, ce_info = validate_predictions(result_root, "ce_only")

    swcs_frame, swcs_matrix, swcs_info = validate_predictions(result_root, "swcs")



    if not ce_frame["utterance_id"].equals(swcs_frame["utterance_id"]):

        raise RuntimeError("CE-only and CSRL TEST utterance order differs")

    if not ce_frame["true_label"].equals(swcs_frame["true_label"]):

        raise RuntimeError("CE-only and CSRL ground-truth labels differ")



    print(

        f"CE-only: E20 | predictions={ce_info['predictions_path']}\n"

        f"CSRL:    E40 | predictions={swcs_info['predictions_path']}"

    )

    print(f"SD TEST rows={len(ce_frame)} | true counts={EXPECTED_TRUE_COUNTS}")



    if args.check_only:

        print("CHECK-ONLY PASS: no figures or result files were written.")

        return



    output_dir.mkdir(parents=True, exist_ok=True)



    ce_percent = row_normalized_percent(ce_matrix)

    swcs_percent = row_normalized_percent(swcs_matrix)

    delta_percent = swcs_percent - ce_percent



    index = pd.Index(LABEL_DISPLAY, name="Ground-truth")

    columns = pd.Index(LABEL_DISPLAY, name="Predicted")



    save_csv(pd.DataFrame(ce_matrix, index=index, columns=columns), output_dir / "ce_only_confusion_counts.csv")

    save_csv(pd.DataFrame(swcs_matrix, index=index, columns=columns), output_dir / "swcs_confusion_counts.csv")

    save_csv(pd.DataFrame(ce_percent, index=index, columns=columns), output_dir / "ce_only_confusion_row_percent.csv")

    save_csv(pd.DataFrame(swcs_percent, index=index, columns=columns), output_dir / "swcs_confusion_row_percent.csv")

    save_csv(pd.DataFrame(delta_percent, index=index, columns=columns), output_dir / "swcs_minus_ce_row_percent_point.csv")



    class_frame = pd.concat(

        [

            class_summary(ce_frame, "CE-only"),

            class_summary(swcs_frame, "CSRL"),

        ],

        ignore_index=True,

    )

    class_frame.to_csv(

        output_dir / "sd_confusion_class_metrics.csv",

        index=False,

        encoding="utf-8-sig",

    )



    ce_class = class_frame[class_frame["method"] == "CE-only"].set_index("class_id")

    swcs_class = class_frame[class_frame["method"] == "CSRL"].set_index("class_id")

    delta_rows = []

    for class_id, class_name in zip(LABEL_IDS, LABEL_KEYS):

        delta_rows.append(

            {

                "class_id": class_id,

                "class_name": class_name,

                "support": int(ce_class.loc[class_id, "support"]),

                "ce_recall": float(ce_class.loc[class_id, "recall"]),

                "swcs_recall": float(swcs_class.loc[class_id, "recall"]),

                "swcs_minus_ce_recall_pp": float(

                    (swcs_class.loc[class_id, "recall"] - ce_class.loc[class_id, "recall"])

                    * 100.0

                ),

                "ce_f1": float(ce_class.loc[class_id, "f1"]),

                "swcs_f1": float(swcs_class.loc[class_id, "f1"]),

                "swcs_minus_ce_f1_pp": float(

                    (swcs_class.loc[class_id, "f1"] - ce_class.loc[class_id, "f1"])

                    * 100.0

                ),

            }

        )

    delta_frame = pd.DataFrame(delta_rows)

    delta_frame.to_csv(

        output_dir / "sd_confusion_class_delta.csv",

        index=False,

        encoding="utf-8-sig",

    )



    aggregate = pd.DataFrame(

        [

            aggregate_summary(ce_frame, "CE-only"),

            aggregate_summary(swcs_frame, "CSRL"),

        ]

    )

    aggregate.to_csv(

        output_dir / "sd_confusion_aggregate_metrics.csv",

        index=False,

        encoding="utf-8-sig",

    )



    png_path = make_figure(

        ce_matrix=ce_matrix,

        swcs_matrix=swcs_matrix,

        output_dir=output_dir,

        annotation_mode=args.annotation,

        dpi=int(args.dpi),

    )



    config = {

        "analysis": "final SD CE-only vs CSRL confusion matrix",

        "result_root": str(result_root),

        "test_samples": EXPECTED_TEST_ROWS,

        "ground_truth_counts": EXPECTED_TRUE_COUNTS,

        "label_order": LABEL_KEYS,

        "main_figure": {

            "normalization": "row-normalized by ground-truth class",

            "shared_scale_percent": [0.0, 100.0],

            "annotation": args.annotation,

            "png": str(png_path),

        },

        "ce_only": ce_info,

        "swcs": swcs_info,

    }

    save_json(config, output_dir / "sd_confusion_matrix_config.json")



    print("\nCE-only confusion matrix (counts):")

    print(pd.DataFrame(ce_matrix, index=LABEL_DISPLAY, columns=LABEL_DISPLAY).to_string())

    print("\nCSRL confusion matrix (counts):")

    print(pd.DataFrame(swcs_matrix, index=LABEL_DISPLAY, columns=LABEL_DISPLAY).to_string())



    print("\nPer-class recall (%):")

    for row in delta_rows:

        print(

            f"  {row['class_name']:9s} | CE={row['ce_recall']*100:6.2f} "

            f"CSRL={row['swcs_recall']*100:6.2f} "

            f"Delta={row['swcs_minus_ce_recall_pp']:+6.2f} pp"

        )



    print("\nSaved:")

    print(f" - {png_path}")

    print(f" - {output_dir / 'ce_only_confusion_counts.csv'}")

    print(f" - {output_dir / 'swcs_confusion_counts.csv'}")

    print(f" - {output_dir / 'sd_confusion_class_delta.csv'}")





if __name__ == "__main__":

    main()
