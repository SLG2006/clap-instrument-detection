"""
Builds a multi-label instrument detection dataset from AudioSet metadata.

Prerequisites:
- ffmpeg
- yt-dlp
- AudioSet CSV files
"""

import os
import csv
import json
import subprocess
from pathlib import Path
from collections import defaultdict

import librosa
import soundfile as sf
from tqdm import tqdm


# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).parent

AUDIOSET_DIR = PROJECT_ROOT / "audioset"
RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"
PROCESSED_AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
OUTPUT_JSON = PROJECT_ROOT / "data" / "instrument_dataset.json"

CLASS_LABELS_CSV = AUDIOSET_DIR / "class_labels_indices.csv"
SEGMENTS_CSV = AUDIOSET_DIR / "unbalanced_train_segments.csv"
MAX_CLIPS_PER_INSTRUMENT = 1

SAMPLE_RATE = 16000
CLIP_DURATION = 10.0

INSTRUMENT_TERMS = [
    "musical instrument",
    "plucked string instrument",
    "guitar",
    "electric guitar",
    "bass guitar",
    "acoustic guitar",
    "steel guitar, slide guitar",
    "tapping (guitar technique)",
    "piano",
    "electric piano",
    "drum kit",
    "drum machine",
    "drum",
    "snare drum",
    "drum roll",
    "bass drum",
    "rattle (instrument)",
    "brass instrument",
    "bowed string instrument",
    "violin, fiddle",
    "cello",
    "wind instrument, woodwind instrument",
    "drum and bass"
]

# =========================
# UTILITIES
# =========================

def ensure_directories():
    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def check_dependencies():
    for tool in ["ffmpeg", "yt-dlp"]:
        if subprocess.call(["which", tool], stdout=subprocess.DEVNULL) != 0:
            raise RuntimeError(f"{tool} is not installed or not in PATH")


# =========================
# AUDIOSET PARSING
# =========================

def load_label_map():
    label_map = {}

    with open(CLASS_LABELS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            label_id = row[1]          # ✅ correct AudioSet label ID
            label_name = row[2].lower()
            label_map[label_id] = label_name

    return label_map


def get_instrument_label_ids(label_map):
    """
    Return dict of label_id -> label_name where the name matches our instrument terms
    """
    instrument_ids = {}
    for label_id, name in label_map.items():
        for term in INSTRUMENT_TERMS:
            if term in name:
                instrument_ids[label_id] = name
                break
    return instrument_ids

def parse_segments(instrument_ids):
    segments = []

    with open(SEGMENTS_CSV, newline="", encoding="utf-8") as f:
        # Skip AudioSet metadata
        for _ in range(3):
            next(f)

        reader = csv.DictReader(
            f,
            fieldnames=["YTID", "start_seconds", "end_seconds", "positive_labels"]
        )

        for row in reader:
            try:
                start = float(row["start_seconds"])
                end = float(row["end_seconds"])
            except ValueError:
                continue

            labels = [
                l.strip().strip('"')
                for l in row["positive_labels"].split(",")
            ]


            matching_labels = [
                instrument_ids[l]
                for l in labels
                if l in instrument_ids
            ]

            if matching_labels:
                segments.append({
                    "ytid": row["YTID"],
                    "start": start,
                    "end": end,
                    "labels": matching_labels
                })

    return segments

def cap_segments_by_instrument(segments, max_per_instrument):
    counts = defaultdict(int)
    capped_segments = []

    for seg in segments:
        keep = False
        for label in seg["labels"]:
            if counts[label] < max_per_instrument:
                keep = True
                break

        if keep:
            capped_segments.append(seg)
            for label in seg["labels"]:
                if counts[label] < max_per_instrument:
                    counts[label] += 1

    print("Clips per instrument after capping:")
    for k, v in counts.items():
        print(f"{k}: {v}")

    return capped_segments

# =========================
# AUDIO HANDLING
# =========================

def download_audio(clip_id):
    output_path = RAW_AUDIO_DIR / f"{clip_id}.wav"
    if output_path.exists():
        return True

    url = f"https://www.youtube.com/watch?v={clip_id}"

    command = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(output_path),
        url
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return result.returncode == 0


def process_audio(clip_id, info):
    raw_path = RAW_AUDIO_DIR / f"{clip_id}.wav"
    processed_path = PROCESSED_AUDIO_DIR / f"{clip_id}.wav"

    if not raw_path.exists():
        return None

    try:
        y, sr = librosa.load(raw_path, sr=SAMPLE_RATE, mono=True)
        start_sample = int(info["start"] * sr)
        end_sample = int(info["end"] * sr)
        y = y[start_sample:end_sample]

        if len(y) < sr:
            return None

        sf.write(processed_path, y, sr)
        return processed_path

    except Exception:
        return None


# =========================
# MAIN PIPELINE
# =========================

def build_dataset():
    ensure_directories()
    check_dependencies()

    dataset = []

    print("Loading label map...")
    label_map = load_label_map()

    print("DEBUG: Checking instrument labels in AudioSet")
    count = 0
    for label_id, name in label_map.items():
        if any(term in name for term in INSTRUMENT_TERMS):
            print(label_id, name)
            count += 1
            if count >= 50:
                break
    print("DEBUG: End of instrument label check\n")

    print("Filtering instrument labels...")
    instrument_ids = get_instrument_label_ids(label_map)
    print(f"Number of instrument labels detected: {len(instrument_ids)}")

    # ===== DEBUG STEP: Check which instrument labels exist in CSV =====
    found_labels = set()

    with open(SEGMENTS_CSV, newline="", encoding="utf-8") as f:
        for _ in range(3):
            next(f)

        reader = csv.DictReader(
            f,
            fieldnames=["YTID", "start_seconds", "end_seconds", "positive_labels"]
        )

        for row in reader:
            labels = [
                l.strip().strip('"')
                for l in row["positive_labels"].split(",")
            ]
            for l in labels:
                if l in instrument_ids:
                    found_labels.add(l)

        print("Instrument labels actually present in CSV:")
        for l in found_labels:
            print(l, instrument_ids[l])
        print(f"Total labels found in CSV: {len(found_labels)}\n")
    # ===== END DEBUG STEP =====


    print("Parsing segments CSV...")
    segments = parse_segments(instrument_ids)
    segments = cap_segments_by_instrument(
        segments,
        MAX_CLIPS_PER_INSTRUMENT
    )

    print(f"Number of clips with instrument labels: {len(segments)}")

    if len(segments) == 0:
        print("No clips found. Check CSV file and instrument terms!")
        return

    print("Downloading and processing clips...")
    for segment in segments:
        ytid = segment["ytid"]

        if not download_audio(ytid):
            continue

        processed_path = process_audio(ytid, segment)
        if processed_path is None:
            continue

        dataset.append({
            "ytid": ytid,
            "audio_path": str(processed_path),
            "labels": segment["labels"]
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset built with {len(dataset)} clips")

if __name__ == "__main__":
    build_dataset()
