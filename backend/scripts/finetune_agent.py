"""Fine-tunes a small local, open-source seq2seq model (LoRA) to replace the
Gemini/Anthropic API call for ONE ARES agent.

Base model: google/flan-t5-small (~80M params, instruction-tuned, seq2seq).
Chosen because:
  - It is small enough to fine-tune on CPU in minutes, not hours.
  - It is already instruction-tuned, so it has a head start on "follow this
    instruction and produce structured output" even before fine-tuning,
    unlike a raw (non-instruction-tuned) base model.
  - Seq2seq (encoder-decoder) is architecturally the right shape for this
    task: prompt in, structured JSON out, rather than open-ended
    continuation like a decoder-only model.

This trains a LoRA adapter (not a full fine-tune) — much cheaper, and the
adapter is portable/small (a few MB) on top of the shared frozen base model,
so all 5 agents can share one loaded base model in memory at inference time
and swap in their own small adapter.

Usage:
    python scripts/finetune_agent.py --agent planner
    python scripts/finetune_agent.py --agent literature --epochs 40
    python scripts/finetune_agent.py --agent all          # trains all 5 in sequence
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

BASE_MODEL = "google/flan-t5-small"
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "finetune" / "raw_pairs.jsonl"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "finetuned"

AGENTS = ["planner", "literature", "gap", "methodology", "reviewer"]

MAX_INPUT_TOKENS = 320
MAX_TARGET_TOKENS = 256


def load_agent_pairs(agent: str) -> list[dict]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found — run scripts/build_finetune_dataset.py first."
        )
    pairs = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["agent"] == agent:
                pairs.append(row)
    return pairs


def train_one_agent(agent: str, epochs: int, out_dir: Path) -> dict:
    pairs = load_agent_pairs(agent)
    if len(pairs) < 2:
        raise ValueError(
            f"Only {len(pairs)} training example(s) for agent '{agent}' — need at least 2 "
            "(one for train, one for held-out eval). Generate more sessions first."
        )

    print(f"\n=== Training agent: {agent} ({len(pairs)} examples) ===")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    # r=8/target=[q,v] was sized for the original ~20-examples/agent dataset.
    # With 966 total pairs now (up to 368 for literature), that undersized
    # the adapter's capacity to actually fit the richer training signal —
    # bumped to r=16 across all attention AND feed-forward projections
    # (T5's gated-GELU FFN exposes wi_0/wi_1/wo, not a single "wi").
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Hold out a slice as the eval set used for best-checkpoint selection
    # (see load_best_model_at_end below). Previously capped at 5 regardless
    # of dataset size — with datasets now up to 363 pairs, 5 examples
    # (1.4%) made "best checkpoint" selection nearly noise. Scales further
    # now, still capped well below train set size.
    n_eval = max(1, min(20, len(pairs) // 8))
    eval_pairs = pairs[-n_eval:]
    train_pairs = pairs[:-n_eval] if len(pairs) > n_eval else pairs

    def to_dataset(rows: list[dict]) -> Dataset:
        return Dataset.from_dict({
            "prompt": [r["prompt"] for r in rows],
            "completion": [r["completion"] for r in rows],
        })

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["prompt"], max_length=MAX_INPUT_TOKENS, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            text_target=batch["completion"],
            max_length=MAX_TARGET_TOKENS,
            truncation=True,
            padding="max_length",
        )
        labels["input_ids"] = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
            for seq in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = to_dataset(train_pairs).map(tokenize, batched=True, remove_columns=["prompt", "completion"])
    eval_ds = to_dataset(eval_pairs).map(tokenize, batched=True, remove_columns=["prompt", "completion"])

    agent_out = out_dir / agent
    agent_out.mkdir(parents=True, exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir=str(agent_out / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=3e-4,
        logging_steps=5,
        eval_strategy="epoch",
        # Previously always kept the FINAL epoch regardless of whether a
        # later epoch had started overfitting (eval_loss ticking back up).
        # Now keeps whichever checkpoint actually had the lowest eval loss.
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        predict_with_generate=False,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    model.save_pretrained(str(agent_out / "adapter"))
    tokenizer.save_pretrained(str(agent_out / "adapter"))

    # Quick generation smoke test on the held-out example.
    model.eval()
    sample_prompt = eval_pairs[0]["prompt"]
    inputs = tokenizer(sample_prompt, max_length=MAX_INPUT_TOKENS, truncation=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=MAX_TARGET_TOKENS)
    generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    summary = {
        "agent": agent,
        "train_examples": len(train_pairs),
        "eval_examples": len(eval_pairs),
        "epochs": epochs,
        "final_train_loss": result.training_loss,
        "elapsed_seconds": round(elapsed, 1),
        "adapter_path": str(agent_out / "adapter"),
        "sample_generation_preview": generated_text[:300],
    }
    with open(agent_out / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Done in {elapsed:.1f}s. Final train loss: {result.training_loss:.4f}")
    print(f"Adapter saved to {agent_out / 'adapter'}")
    print(f"Sample generation on held-out prompt (first 300 chars):\n{generated_text[:300]}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=AGENTS + ["all"])
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    targets = AGENTS if args.agent == "all" else [args.agent]

    all_summaries = []
    for agent in targets:
        summary = train_one_agent(agent, args.epochs, MODELS_DIR)
        all_summaries.append(summary)

    print("\n=== Summary ===")
    for s in all_summaries:
        print(f"{s['agent']}: loss={s['final_train_loss']:.4f}, time={s['elapsed_seconds']}s -> {s['adapter_path']}")


if __name__ == "__main__":
    main()
