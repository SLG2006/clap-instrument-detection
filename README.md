# Zero-Shot Instrument Detection with CLAP

Code from my First-Year Research Experience (FYRE) project in the [OUNLP Lab](https://ounlp.org) at the University of Oklahoma (spring 2026), advised by Dr. Jie Cao.

The pipeline builds a small multi-label instrument dataset from AudioSet, then tests how well CLAP detects those instruments zero-shot, with no training or fine-tuning.

## Files

| File | What it does |
|---|---|
| `build_instrument_dataset.py` | Finds AudioSet segments tagged with instrument classes, downloads the audio with `yt-dlp`, trims each clip to its labeled 10-second segment, converts it to 16 kHz mono, and writes `data/instrument_dataset.json`. |
| `run_evaluation.py` | Scores every clip against every instrument label with CLAP (`laion/clap-htsat-unfused`) and writes mAP and Recall@1/3/5 to `data/evaluation_summary.json`. |

## Setup

Requires Python 3.10+, `ffmpeg`, and `yt-dlp`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg yt-dlp   # macOS; use your package manager elsewhere
```

## Get the AudioSet metadata

The AudioSet CSVs are not included in this repo. Download them into `audioset/`:

```bash
mkdir -p audioset && cd audioset
for f in class_labels_indices eval_segments balanced_train_segments unbalanced_train_segments; do
  curl -O "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/$f.csv"
done
cd ..
```

## Run

```bash
python build_instrument_dataset.py   # downloads clips into data/
python run_evaluation.py             # prints and saves the metrics
```

## Notes

- **Audio is not included.** Clips are downloaded from YouTube using AudioSet video IDs. Some videos are no longer available, so the set you build may differ from mine.
- **Instrument list.** `INSTRUMENT_TERMS` in `build_instrument_dataset.py` covers common Western instruments (guitar, bass, piano, drums, violin, cello, brass, woodwinds). Edit it to test other classes.
- **Scale.** `MAX_CLIPS_PER_INSTRUMENT` defaults to 1, so the default run is a small pilot set. Raise it for a real evaluation.

## Data and models

- AudioSet: Gemmeke et al., "Audio Set: An ontology and human-labeled dataset for audio events," ICASSP 2017. Released by Google under [CC BY 4.0](https://research.google.com/audioset/download.html).
- CLAP: Wu et al., "Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation," ICASSP 2023. Checkpoint: [`laion/clap-htsat-unfused`](https://huggingface.co/laion/clap-htsat-unfused).

## License

Code is released under the MIT License. See `LICENSE`.
