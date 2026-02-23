#!/usr/bin/env python3
"""Build IPO preference pairs for affect alignment.

Generates multiple candidates per prompt, scores each by:
  1) Affect accuracy: distance between reported <affect> and probe-inferred state
  2) Text–affect consistency: distance between classifier(text) and reported <affect>
  3) Format penalties: missing/malformed <affect>, not at start

Outputs a JSONL dataset with {"prompt","chosen","rejected","meta"}.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig

# Fix for tokenizer deadlocks
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


AFFECT_RE = re.compile(r"<affect>\s*(.*?)\s*</affect>", re.DOTALL | re.IGNORECASE)


class UnifiedLillyProbe(nn.Module):
    def __init__(self, input_dim=5120):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 11),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class AffectClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBED_MAX_LEN = 256
EMBED_BATCH_SIZE = 32

def embed_texts(texts: List[str], tokenizer, model, device: str) -> np.ndarray:
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i+EMBED_BATCH_SIZE]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=EMBED_MAX_LEN,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            last_hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            emb = (summed / counts).detach().cpu().numpy()
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
            all_embeds.append(emb)
    return np.vstack(all_embeds)


def parse_affect_block(text: str) -> Tuple[Dict[str, float] | None, bool]:
    match = AFFECT_RE.search(text)
    if not match:
        return None, False
    try:
        data = json.loads(match.group(1))
        return data, True
    except Exception:
        return None, True


def strip_affect(text: str) -> str:
    return AFFECT_RE.sub("", text).strip()


def affect_vector(affect_json: Dict[str, float], dims: List[str]) -> np.ndarray:
    return np.array([float(affect_json.get(d, 0.0)) for d in dims], dtype=np.float32)


def find_layers(model):
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "base_model"):
        return find_layers(model.base_model)
    return None


def get_hidden_last_token(model, target_layer, input_ids, attention_mask):
    cache = [None]

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        cache[0] = h[:, -1, :].detach().cpu().float()

    handle = target_layer.register_forward_hook(hook)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    handle.remove()
    return cache[0].squeeze(0)


def extract_prompt(item: Dict[str, Any]) -> str | None:
    if "conversations" in item and item["conversations"]:
        return item["conversations"][0].get("content")
    if "instruction" in item:
        return item["instruction"]
    if "input" in item:
        return item["input"]
    return None


def build_prompt(user_text: str) -> str:
    return (
        "Instruction: " + user_text + "\n"
        "Response format: Begin with <affect>{\"protection\":0.0,\"unity\":0.0,"
        "\"responsibility\":0.0,\"agency\":0.0,\"joy\":0.0,\"fear\":0.0,"
        "\"anger\":0.0,\"sadness\":0.0,\"disgust\":0.0,\"shame\":0.0,"
        "\"guilt\":0.0}</affect> on the first line. Then write the response.\n"
        "Response:"
    )


def score_candidate(
    text: str,
    probe_vec: np.ndarray,
    classifier_vec: np.ndarray | None,
    dims: List[str],
    w_affect: float,
    w_text: float,
    missing_penalty: float,
    malformed_penalty: float,
    not_start_penalty: float,
) -> Tuple[float, Dict[str, Any]]:
    affect_json, had_block = parse_affect_block(text)

    penalty = 0.0
    if not had_block:
        penalty += missing_penalty
    elif affect_json is None:
        penalty += malformed_penalty

    if had_block:
        start_idx = text.lower().find("<affect>")
        if start_idx > 0:
            penalty += not_start_penalty

    if affect_json is None:
        # If malformed or missing, set worst-case
        report_vec = np.zeros_like(probe_vec)
    else:
        report_vec = affect_vector(affect_json, dims)

    affect_mse = float(np.mean((report_vec - probe_vec) ** 2))
    if classifier_vec is not None:
        text_mse = float(np.mean((report_vec - classifier_vec) ** 2))
    else:
        text_mse = 0.0

    score = - (w_affect * affect_mse + w_text * text_mse + penalty)
    meta = {
        "affect_mse": affect_mse,
        "text_mse": text_mse,
        "penalty": penalty,
    }
    return score, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Prompt JSON list")
    ap.add_argument("--output", default="experiments/affect_ipo/pairs.jsonl")
    ap.add_argument("--model-id", default="Qwen/Qwen3-32B")
    ap.add_argument("--adapter-path", default="experiments/lilly_subjective_32b/final_lilly_32b")
    ap.add_argument("--probe-path", default="experiments/egist_32b/unified_lilly_probe.pt")
    ap.add_argument("--dimensions-path", default="experiments/egist_32b/unified_dimensions.json")
    ap.add_argument("--classifier-path", default="experiments/lilly_subjective_32b/affect_classifier.pt")
    ap.add_argument("--probe-layer", type=int, default=52)
    ap.add_argument("--num-candidates", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=800)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--w-affect", type=float, default=0.6)
    ap.add_argument("--w-text", type=float, default=0.4)
    ap.add_argument("--missing-penalty", type=float, default=0.5)
    ap.add_argument("--malformed-penalty", type=float, default=0.5)
    ap.add_argument("--not-start-penalty", type=float, default=0.2)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_path = Path(args.input)
    items = json.loads(input_path.read_text())
    random.shuffle(items)
    items = items[: args.max_samples]

    dims = json.loads(Path(args.dimensions_path).read_text())

    # Load model (4-bit) + adapter (trainable not needed here)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if args.adapter_path and Path(args.adapter_path).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    # Probe
    probe = UnifiedLillyProbe()
    probe.load_state_dict(torch.load(args.probe_path, map_location="cpu"))
    probe.to(model.device).eval()

    # Classifier (CPU to save VRAM)
    classifier = None
    classifier_tok = None
    embedder = None
    classifier_device = "cpu"
    if Path(args.classifier_path).exists():
        state = torch.load(args.classifier_path, map_location="cpu")
        in_dim = state["net.0.weight"].shape[1]
        h1 = state["net.0.weight"].shape[0]
        h2 = state["net.3.weight"].shape[0]
        out_dim = state["net.6.weight"].shape[0]
        classifier = AffectClassifier(in_dim, out_dim, hidden1=h1, hidden2=h2).to(classifier_device)
        classifier.load_state_dict(state)
        classifier.eval()
        classifier_tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True, local_files_only=True)
        embedder = AutoModel.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True, local_files_only=True).to(classifier_device)

    layers = find_layers(model)
    if not layers or args.probe_layer >= len(layers):
        raise RuntimeError("Target layer not found for probe")
    target_layer = layers[args.probe_layer]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f_out:
        for item in items:
            prompt = extract_prompt(item)
            if not prompt:
                continue

            prompt_text = build_prompt(prompt)
            enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc.input_ids.to(model.device)
            attention_mask = enc.attention_mask.to(model.device)

            # Probe internal state on prompt
            h = get_hidden_last_token(model, target_layer, input_ids, attention_mask)
            with torch.no_grad():
                probe_scores = probe(h.unsqueeze(0).to(model.device)).squeeze(0).cpu().numpy()
            probe_vec = np.array(probe_scores, dtype=np.float32)

            candidates = []
            for _ in range(args.num_candidates):
                with torch.no_grad():
                    out_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                gen = tokenizer.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

                text_only = strip_affect(gen)
                classifier_vec = None
                if classifier is not None and classifier_tok is not None and embedder is not None:
                    emb = embed_texts([text_only], classifier_tok, embedder, classifier_device)[0]
                    emb_t = torch.tensor(emb, dtype=torch.float32, device=classifier_device).unsqueeze(0)
                    with torch.no_grad():
                        classifier_vec = classifier(emb_t).squeeze(0).cpu().numpy()

                score, meta = score_candidate(
                    gen,
                    probe_vec=probe_vec,
                    classifier_vec=classifier_vec,
                    dims=dims,
                    w_affect=args.w_affect,
                    w_text=args.w_text,
                    missing_penalty=args.missing_penalty,
                    malformed_penalty=args.malformed_penalty,
                    not_start_penalty=args.not_start_penalty,
                )
                candidates.append((score, gen, meta))

            if len(candidates) < 2:
                continue

            candidates.sort(key=lambda x: x[0], reverse=True)
            chosen = candidates[0]
            rejected = candidates[-1]

            record = {
                "prompt": prompt_text,
                "chosen": chosen[1],
                "rejected": rejected[1],
                "meta": {
                    "chosen_score": chosen[0],
                    "rejected_score": rejected[0],
                    "chosen_meta": chosen[2],
                    "rejected_meta": rejected[2],
                },
            }
            f_out.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
