import argparse
import json
import gc
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
import torch
from sklearn.metrics import accuracy_score
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DATA_DIR = REPO_ROOT / "sample_data"
CONFIG_DIR = REPO_ROOT / "configs"


def resolve_results_dir(ft_model_ref: str) -> Path:
    model_path = Path(ft_model_ref)
    if model_path.exists():
        target_dir = model_path if model_path.is_dir() else model_path.parent
        if target_dir.name.startswith("checkpoint-"):
            target_dir = target_dir.parent
        out_dir = target_dir / "evaluations"
    else:
        out_dir = REPO_ROOT / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_evaluation_outputs(
    out_dir: Path,
    args,
    infer_cfg: dict,
    ft_model_ref: str,
    base_model_ref: str,
    data_file: Path,
    true_labels,
    ft_preds,
    base_preds,
    ft_acc: float,
    base_acc: float,
    ft_use_base_prompt: bool,
    base_use_base_prompt: bool,
    ft_max_seq_length: int,
    base_max_seq_length: int,
    ft_timing: dict,
    base_timing: dict,
):
    split = args.split
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"eval_{split}_{timestamp}.json"
    details_path = out_dir / f"eval_{split}_{timestamp}_details.csv"

    n_samples = len(true_labels)
    ft_correct = sum(p == t for p, t in zip(ft_preds, true_labels))
    base_correct = sum(p == t for p, t in zip(base_preds, true_labels))

    summary = {
        "timestamp": datetime.now().isoformat(),
        "run_context": {
            "args": vars(args),
            "infer_cfg": infer_cfg,
            "resolved": {
                "data_file": str(data_file),
                "split": split,
                "n_samples": n_samples,
                "fine_tuned_model": ft_model_ref,
                "base_model": base_model_ref,
                "prompt_mode": {
                    "fine_tuned_use_base_prompt": ft_use_base_prompt,
                    "base_use_base_prompt": base_use_base_prompt,
                },
                "max_seq_length": {
                    "fine_tuned": ft_max_seq_length,
                    "base": base_max_seq_length,
                },
            },
        },
        "metrics": {
            "fine_tuned_accuracy": ft_acc,
            "base_accuracy": base_acc,
            "delta_ft_minus_base": ft_acc - base_acc,
            "fine_tuned_correct": ft_correct,
            "base_correct": base_correct,
            "timing_seconds": {
                "fine_tuned": ft_timing,
                "base": base_timing,
            },
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(
        {
            "true_label": true_labels,
            "pred_fine_tuned": ft_preds,
            "pred_base": base_preds,
            "correct_fine_tuned": [p == t for p, t in zip(ft_preds, true_labels)],
            "correct_base": [p == t for p, t in zip(base_preds, true_labels)],
        }
    ).to_csv(details_path, index=False)

    return summary_path, details_path


def resolve_path(path_str: str) -> Path:
    """Resolve a path from multiple sensible roots and return the first existing candidate."""
    candidate = Path(path_str)
    candidates = [
        candidate,
        REPO_ROOT / candidate,
        WORKSPACE_ROOT / candidate,
    ]
    for item in candidates:
        if item.exists():
            return item.resolve()
    # Fall back to repo-relative location for downstream errors.
    return (REPO_ROOT / candidate).resolve()


def find_latest_run(adapter_root: Path) -> Path:
    run_dirs = [p for p in adapter_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {adapter_root}")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def is_adapter_run_dir(path: Path) -> bool:
    """Return True when path looks like a saved adapter run directory."""
    return (path / "adapter_config.json").exists() and (path / "tokenizer_config.json").exists()


def resolve_model_path(args, infer_cfg) -> str:
    if args.model_path:
        # Allow either a local path or a Hugging Face model id.
        maybe_local = resolve_path(args.model_path)
        if maybe_local.exists():
            return str(maybe_local)
        return args.model_path

    adapter_root = resolve_path(infer_cfg["adapter_path"])
    if not adapter_root.exists():
        raise FileNotFoundError(
            f"Adapter root not found: {adapter_root}. "
            "Check configs/inference.yaml -> adapter_path"
        )

    if args.run_dir:
        run_dir = resolve_path(args.run_dir)
    else:
        # Support both adapter_path styles:
        # 1) a parent directory containing many timestamped runs
        # 2) a specific run directory containing adapter files
        run_dir = adapter_root if is_adapter_run_dir(adapter_root) else find_latest_run(adapter_root)

    if run_dir.name == "evaluations":
        raise ValueError(
            "Resolved run directory points to an evaluations folder, not a model folder. "
            "Use --run-dir checkpoints/<timestamp> or set configs/inference.yaml -> adapter_path "
            "to a run directory containing adapter_config.json."
        )

    if args.best_from_run:
        trainer_state_path = run_dir / "trainer_state.json"
        if trainer_state_path.exists():
            with open(trainer_state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            best_ckpt = state.get("best_model_checkpoint")
            if best_ckpt:
                best_path = Path(best_ckpt)
                if best_path.exists():
                    return str(best_path.resolve())
                maybe_relative = run_dir / best_ckpt
                if maybe_relative.exists():
                    return str(maybe_relative.resolve())
        print(f"[warn] Could not resolve best checkpoint from {trainer_state_path}; using run dir.")

    if is_adapter_run_dir(run_dir):
        return str(run_dir.resolve())

    raise FileNotFoundError(
        f"Could not find adapter files in resolved run directory: {run_dir}. "
        "Expected files like adapter_config.json and tokenizer_config.json."
    )


def load_model_and_tokenizer(model_name: str, infer_cfg: dict, max_seq_length: int):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=int(max_seq_length),
        dtype=None,
        load_in_4bit=infer_cfg.get("load_in_4bit", True),
    )
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
    tokenizer = get_chat_template(tokenizer, chat_template=infer_cfg["chat_template"])
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def normalize_prediction(pred: str, valid_labels):
    raw = pred.strip()
    if raw in valid_labels:
        return raw

    lowered = raw.lower()
    lut = {x.lower(): x for x in valid_labels}
    if lowered in lut:
        return lut[lowered]

    first = raw.split("\n")[0].strip()
    if first in valid_labels:
        return first
    if first.lower() in lut:
        return lut[first.lower()]

    # Fallback: pick first label that appears in the output.
    for label in valid_labels:
        if label.lower() in lowered:
            return label
    return raw


def predict_label(model, tokenizer, text: str, max_new_tokens: int, valid_labels, use_base_prompt: bool) -> str:
    if use_base_prompt:
        labels_block = "\n".join(f"- {x}" for x in sorted(valid_labels))
        system_prompt = (
            "You are an intent classification assistant. "
            "Choose exactly 1 label from the allowed list and output only that label."
        )
        user_prompt = (
            "Task: Classify the banking intent of the text below.\n"
            "Allowed labels:\n"
            f"{labels_block}\n\n"
            f"Text: {text}\n\n"
            "Output format: Return only one label from the list, with no explanation."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages = [{"role": "user", "content": f"Classify the intent: {text}"}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = outputs[0][inputs["input_ids"].shape[1] :]
    decoded = tokenizer.decode(gen, skip_special_tokens=True)
    return decoded.strip().split("\n")[0].strip()


def evaluate_model(
    model,
    tokenizer,
    df: pd.DataFrame,
    id2label: dict,
    max_new_tokens: int,
    use_base_prompt: bool,
):
    true_labels = [id2label[int(x)] for x in df["label"].tolist()]
    # Use only the subset of intents present in the evaluation dataset
    valid_labels = set(true_labels)
    preds = []

    total = len(df)
    print(f"[progress] Evaluating {total} samples...")
    for index, text in enumerate(tqdm(df["text"].tolist(), desc="evaluating", unit="sample"), start=1):
        raw_pred = predict_label(
            model,
            tokenizer,
            text,
            max_new_tokens=max_new_tokens,
            valid_labels=valid_labels,
            use_base_prompt=use_base_prompt,
        )
        preds.append(normalize_prediction(raw_pred, valid_labels))
        if index % 25 == 0 or index == total:
            print(f"[progress] {index}/{total} samples done")

    acc = accuracy_score(true_labels, preds)
    return acc, preds, true_labels


def evaluate_checkpoint(
    model_ref: str,
    infer_cfg: dict,
    max_seq_length: int,
    df: pd.DataFrame,
    id2label: dict,
    max_new_tokens: int,
    use_base_prompt: bool,
):
    start = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(model_ref, infer_cfg, max_seq_length=max_seq_length)

    try:
        acc, preds, true_labels = evaluate_model(
            model,
            tokenizer,
            df,
            id2label,
            max_new_tokens=max_new_tokens,
            use_base_prompt=use_base_prompt,
        )
    finally:
        del model
        del tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    elapsed_seconds = time.perf_counter() - start
    n_samples = max(len(df), 1)
    timing = {
        "elapsed_seconds": round(elapsed_seconds, 4),
        "avg_seconds_per_sample": round(elapsed_seconds / n_samples, 6),
    }
    return acc, preds, true_labels, timing


def main():
    parser = argparse.ArgumentParser(description="Evaluate latest or specific LoRA checkpoint.")
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIR / "inference.yaml"),
        help="Path to inference config yaml.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split file to evaluate.",
    )
    parser.add_argument("--model-path", default=None, help="Direct model/adapter path to evaluate.")
    parser.add_argument("--run-dir", default=None, help="Run directory under adapter_path.")
    parser.add_argument(
        "--best-from-run",
        action="store_true",
        help="Use best_model_checkpoint from trainer_state.json in run dir.",
    )
    args = parser.parse_args()

    with open(resolve_path(args.config), "r", encoding="utf-8") as f:
        infer_cfg = yaml.safe_load(f)

    label_map_path = resolve_path(infer_cfg["label_text_map"])
    if not label_map_path.exists():
        label_map_path = CONFIG_DIR / infer_cfg["label_text_map"]

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_text_map = json.load(f)
    id2label = {int(k): v for k, v in label_text_map.items()}

    data_file = DATA_DIR / f"{args.split}.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset split not found: {data_file}")
    df = pd.read_csv(data_file)

    ft_model_ref = resolve_model_path(args, infer_cfg)
    base_model_ref = infer_cfg["base_model"]
    ft_max_seq_length = int(infer_cfg["max_seq_length"])
    base_max_seq_length = int(infer_cfg.get("base_model_max_seq_length", ft_max_seq_length))

    print(f"[info] Evaluation split: {args.split} -> {data_file}")
    print(f"[info] Fine-tuned model: {ft_model_ref}")
    print(f"[info] Base model: {base_model_ref}")
    print(f"[info] max_seq_length (fine-tuned): {ft_max_seq_length}")
    print(f"[info] max_seq_length (base): {base_max_seq_length}")
    print(f"[info] max_new_tokens: {infer_cfg['max_new_tokens']}")

    ft_use_base_prompt = False
    base_use_base_prompt = True

    ft_acc, ft_preds, true_labels, ft_timing = evaluate_checkpoint(
        ft_model_ref,
        infer_cfg,
        ft_max_seq_length,
        df,
        id2label,
        max_new_tokens=int(infer_cfg["max_new_tokens"]),
        use_base_prompt=ft_use_base_prompt,
    )

    base_acc, base_preds, _, base_timing = evaluate_checkpoint(
        base_model_ref,
        infer_cfg,
        base_max_seq_length,
        df,
        id2label,
        max_new_tokens=int(infer_cfg["max_new_tokens"]),
        use_base_prompt=base_use_base_prompt,
    )

    ft_correct = sum(p == t for p, t in zip(ft_preds, true_labels))
    base_correct = sum(p == t for p, t in zip(base_preds, true_labels))
    print(f"[result] Fine-tuned accuracy={ft_acc:.4f} ({ft_correct}/{len(true_labels)})")
    print(f"[result] Base accuracy       ={base_acc:.4f} ({base_correct}/{len(true_labels)})")
    print(f"[result] Delta (ft - base)   ={(ft_acc - base_acc):.4f}")
    print(
        f"[timing] Fine-tuned elapsed={ft_timing['elapsed_seconds']}s "
        f"avg/sample={ft_timing['avg_seconds_per_sample']}s"
    )
    print(
        f"[timing] Base       elapsed={base_timing['elapsed_seconds']}s "
        f"avg/sample={base_timing['avg_seconds_per_sample']}s"
    )

    results_dir = resolve_results_dir(ft_model_ref)
    summary_path, details_path = save_evaluation_outputs(
        out_dir=results_dir,
        args=args,
        infer_cfg=infer_cfg,
        ft_model_ref=ft_model_ref,
        base_model_ref=base_model_ref,
        data_file=data_file,
        true_labels=true_labels,
        ft_preds=ft_preds,
        base_preds=base_preds,
        ft_acc=ft_acc,
        base_acc=base_acc,
        ft_use_base_prompt=ft_use_base_prompt,
        base_use_base_prompt=base_use_base_prompt,
        ft_max_seq_length=ft_max_seq_length,
        base_max_seq_length=base_max_seq_length,
        ft_timing=ft_timing,
        base_timing=base_timing,
    )
    print(f"[saved] Summary: {summary_path}")
    print(f"[saved] Details: {details_path}")


if __name__ == "__main__":
    main()
