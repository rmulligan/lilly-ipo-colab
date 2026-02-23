#!/usr/bin/env python3
import json
import random
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    items = json.loads(Path(args.input).read_text())
    random.seed(args.seed)
    random.shuffle(items)
    if args.max_samples:
        items = items[: args.max_samples]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = args.shards
    shard_size = (len(items) + n - 1) // n
    for i in range(n):
        shard_items = items[i*shard_size:(i+1)*shard_size]
        shard_path = out_dir / f"prompts_shard_{i:02d}.json"
        shard_path.write_text(json.dumps(shard_items, indent=2))
        print(f"Wrote {shard_path} ({len(shard_items)} items)")

if __name__ == "__main__":
    main()
