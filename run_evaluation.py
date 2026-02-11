"""
Zero-shot multi-label instrument detection evaluation using CLAP.

Computes:
- mAP (mean Average Precision)
- Recall@K

Assumes dataset built by build_instrument_dataset.py
"""

import json
from pathlib import Path
from typing import List, Dict

import librosa

import torch
import numpy as np
from transformers import ClapProcessor, ClapModel
from tqdm import tqdm


# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).parent
DATASET_JSON = BASE_DIR / "data" / "instrument_dataset.json"

SAMPLE_RATE = 48000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Recall cutoffs
RECALL_KS = [1, 3, 5]


# =========================
# AUDIO LOADING
# =========================

def load_audio(path: Path) -> torch.Tensor:
    y, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    y = torch.from_numpy(y).float().unsqueeze(0)
    return y

# =========================
# METRICS
# =========================

def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Computes Average Precision for a single sample.
    y_true: (L,) binary
    y_score: (L,) real-valued scores
    """

    # Sort by descending score
    order = np.argsort(-y_score)
    y_true = y_true[order]

    if y_true.sum() == 0:
        return 0.0

    cumsum = np.cumsum(y_true)
    precision = cumsum / (np.arange(len(y_true)) + 1)

    ap = (precision * y_true).sum() / y_true.sum()
    return ap


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Recall@K for a single sample.
    """
    top_k = np.argsort(-y_score)[:k]
    return y_true[top_k].sum() / max(1, y_true.sum())


# =========================
# MAIN EVALUATION
# =========================

def main():
    # -------------------------
    # Load dataset
    # -------------------------
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    assert len(dataset) > 0, "Dataset is empty"

    # -------------------------
    # Build closed label set
    # -------------------------
    label_set = sorted({
        label
        for sample in dataset
        for label in sample["labels"]
    })

    label_to_idx = {l: i for i, l in enumerate(label_set)}
    num_labels = len(label_set)

    print(f"Number of evaluation labels: {num_labels}")
    print("Labels:", label_set)

    # -------------------------
    # Load CLAP
    # -------------------------
    processor = ClapProcessor.from_pretrained(
        "laion/clap-htsat-unfused"
    )
    model = ClapModel.from_pretrained(
        "laion/clap-htsat-unfused"
    ).to(DEVICE)
    model.eval()

    # -------------------------
    # Encode text prompts
    # -------------------------
    text_inputs = processor(
        text=label_set,
        return_tensors="pt",
        padding=True
    )

    text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}

    with torch.no_grad():
        text_outputs = model.text_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )

        text_embeds = text_outputs.pooler_output
        text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)

    # -------------------------
    # Evaluation loop
    # -------------------------
    aps = []
    recalls = {k: [] for k in RECALL_KS}

    for sample in tqdm(dataset, desc="Evaluating"):
        audio_path = BASE_DIR / sample["audio_path"]
        waveform = load_audio(audio_path).to(DEVICE)

        audio_inputs = processor(
            audio=waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        audio_inputs = {k: v.to(DEVICE) for k, v in audio_inputs.items()}


        with torch.no_grad():
            audio_outputs = model.audio_model(
                input_features=audio_inputs["input_features"]
            )

            audio_embed = audio_outputs.pooler_output
            audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)


        scores = (audio_embed @ text_embeds.T).squeeze(0).cpu().numpy()

        # -------------------------
        # Ground truth vector
        # -------------------------
        y_true = np.zeros(num_labels, dtype=np.float32)
        for label in sample["labels"]:
            y_true[label_to_idx[label]] = 1.0

        # -------------------------
        # Metrics
        # -------------------------
        aps.append(average_precision(y_true, scores))

        for k in RECALL_KS:
            recalls[k].append(recall_at_k(y_true, scores, k))

    # -------------------------
    # Aggregate metrics
    # -------------------------
    mean_ap = float(np.mean(aps))
    mean_recalls = {
        f"Recall@{k}": float(np.mean(recalls[k]))
        for k in RECALL_KS
    }

    # -------------------------
    # Save results
    # -------------------------
    results = {
        "num_clips": len(dataset),
        "num_labels": num_labels,
        "mAP": mean_ap,
        **mean_recalls
    }

    output_path = BASE_DIR / "data" / "evaluation_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n===== Evaluation Results =====")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
