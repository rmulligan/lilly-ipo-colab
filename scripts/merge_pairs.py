#!/usr/bin/env python3
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_path = Path(args.output)
    files = sorted(in_dir.glob("pairs_shard_*.jsonl"))
    if not files:
        raise SystemExit("No pairs_shard_*.jsonl files found")

    with out_path.open("w") as out:
        for f in files:
            for line in f.read_text().splitlines():
                if line.strip():
                    out.write(line + "\n")
    print(f"Merged {len(files)} files -> {out_path}")

if __name__ == "__main__":
    main()
