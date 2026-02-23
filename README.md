# Lilly IPO Colab Runner

This repo contains the minimal files to run `build_ipo_pairs.py` in Google Colab and save shard outputs to Google Drive.

## Contents
- `colab_ipo_shard.ipynb`: Colab notebook to run one shard.
- `scripts/build_ipo_pairs.py`: Candidate generation + scoring.
- `scripts/shard_prompts.py`: Local sharding utility.
- `scripts/merge_pairs.py`: Local merge utility.
- `experiments/affect_ipo/shards/`: 8 prompt shards (100 each).
- `experiments/egist_32b/unified_lilly_probe.pt`: 11D probe weights.
- `experiments/egist_32b/unified_dimensions.json`: dimension names.
- `experiments/lilly_subjective_32b/affect_classifier.pt`: text-affect classifier.

## Required (not included)
You must provide the Lilly adapter directory:
- `experiments/lilly_subjective_32b/final_lilly_32b`

This is not included due to size. Place it under the repo root in Colab (or mount from Drive).

## Quick start (Colab)
1. Open `colab_ipo_shard.ipynb` in Colab.
2. Mount Google Drive.
3. Set `REPO_DIR` to the repo location.
4. Place the adapter at:
   `experiments/lilly_subjective_32b/final_lilly_32b`
5. Set `SHARD_ID` (00..07) and run.

Outputs go to:
`/content/drive/MyDrive/lilly_ipo/pairs_shard_XX.jsonl`

## Local merge
After downloading all shards:
```bash
python scripts/merge_pairs.py --input-dir experiments/affect_ipo --output experiments/affect_ipo/pairs.jsonl
```
