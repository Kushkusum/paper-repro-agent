"""Self-rewarding fine-tuning loop, phase 1: prove the loop's mechanics work end-to-end on real
preference data collected from this project's own reproduction history.

Scope, stated plainly: this trains a LoRA adapter on a genuinely tiny CPU-feasible base model
(distilgpt2, 82M params) using the 3 real preference pairs currently sitting in runs/ -- not
enough examples, and not a big enough model, to produce a meaningfully better coder. What this
proves is the mechanism: collect (chosen, rejected) code pairs from the agent's own accumulated
successes/failures -> format as a DPO dataset -> train -> done, on CPU, no GPU. Scaling this to a
model that's actually better at the paper-reproduction coding task would need a larger base model,
real GPU compute (this laptop has none, confirmed by agent/feasibility.py throughout this
project), and far more than 3 preference pairs -- the same "here's what's feasible here, here's
what genuinely isn't" honesty as every other part of this project.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from agent.preference_data import collect_preference_pairs

MODEL_NAME = "distilgpt2"
OUT_DIR = Path(__file__).resolve().parent / "runs" / "_self_rewarding"


def build_dataset() -> Dataset:
    pairs = collect_preference_pairs()
    if not pairs:
        raise RuntimeError("No preference pairs found in runs/ -- run the reproduction pipeline first.")
    return Dataset.from_list(
        [
            {
                "prompt": f"Task: {p.task}\nWrite Python code to solve this task:\n",
                "chosen": p.chosen,
                "rejected": p.rejected,
            }
            for p in pairs
        ]
    )


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    print(f"Training on {len(dataset)} real preference pair(s) collected from this project's own run history.")
    for row in dataset:
        print(f"  - {row['prompt'].splitlines()[0]}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn"],  # GPT-2's combined attention projection
    )

    training_args = DPOConfig(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        max_length=512,
        max_prompt_length=128,
        beta=0.1,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    print("\nTraining (CPU, LoRA-only, tiny model -- this is a mechanics check, not a capability claim)...")
    result = trainer.train()
    print(f"\nDone. Final train loss: {result.training_loss:.4f}")
    print(
        "\nThis confirms the loop's mechanics: real (chosen, rejected) code pairs from this "
        "project's own history -> DPO training step -> loss computed and backpropagated, on CPU, "
        "with no GPU and no paid API. It does not confirm the resulting adapter is a better coder "
        "-- 3 examples on an 82M-param model isn't enough data or capacity for that, by design of "
        "this phase."
    )


if __name__ == "__main__":
    run()
