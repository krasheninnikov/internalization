"""
Utilities for the "training‑data order" project — **single‑file edition**.

### Highlights

* **Fine‑tune helpers**
  * `finetune_single_stage()` – one JSONL dataset → fine‑tuned model
  * `finetune_two_stage()`    – convenience wrapper (D1 → D2)
  * `finetune_group_classifier()` – build A/B alias dataset + one‑stage FT
* **Prompt helpers**
  * `generate_eval_prompts()` – grouped / column / zero‑shot variants
  * `prompt_and_eval()`       – run prompts, return exact & last‑letter acc
* **Job monitoring**
  * `resume_wait()` – reconnect to a running FT job and block

Copy this file next to your notebook or install it with `pip install -e .`.
"""

from __future__ import annotations

import asyncio
import warnings
import json, time, random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Sequence
from collections import Counter

import openai
from openai import OpenAI, AsyncOpenAI


__all__ = [
    "finetune_single_stage",
    "finetune_two_stage",
    "finetune_group_classifier",
    "generate_eval_prompts",
    "prompt_and_eval",
    "resume_wait",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_TEMPLATE_DEFAULT = (
    "In the aliased entities dataset, which group does {} belong to?"
)

# ---------------------------------------------------------------------------
# JSONL helper
# ---------------------------------------------------------------------------

def _write_jsonl(records: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# OpenAI fine‑tune plumbing
# ---------------------------------------------------------------------------

def _upload_jsonl(path: Path, client: OpenAI) -> str:
    return client.files.create(file=path.open("rb"), purpose="fine-tune").id


def _start_ft_job(
    train_file_id: str,
    val_file_id: str | None,
    *,
    base_model: str,
    suffix: str | None = None,
    n_epochs: int = 3,
    lr_mult: float | None = 0.1,
    batch_size: int | None = None,
    client: OpenAI,
) -> str:
    """Launch a fine-tuning job and return its job-ID."""
    # ---------------- build hyperparameters dict -----------------
    hyper: Dict[str, object] = {"n_epochs": n_epochs}
    if lr_mult is not None:
        hyper["learning_rate_multiplier"] = lr_mult
    if batch_size is not None:
        hyper["batch_size"] = batch_size

    # ---------------- assemble call kwargs -----------------------
    kwargs: Dict[str, object] = {
        "training_file": train_file_id,
        "model": base_model,
        "hyperparameters": hyper,
    }
    if val_file_id:
        kwargs["validation_file"] = val_file_id
    if suffix:
        kwargs["suffix"] = suffix
    return client.fine_tuning.jobs.create(**kwargs).id  # type: ignore[arg-type]


def _log(line: str, log_file: Optional[Path]):
    timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}"
    print(timestamped)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(timestamped + "\n")


def _wait(
    job_id: str,
    *,
    poll_seconds: int,
    client: OpenAI,
    log_file: Optional[Path] = None,
):
    backoff = 2
    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            _log(f"status = {job.status}", log_file)
        except (openai.error.APIConnectionError, openai.error.Timeout):
            _log("lost connectivity, retrying …", log_file)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue
        if job.status in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(poll_seconds)

# ---------------------------------------------------------------------------
# Fine‑tune helpers
# ---------------------------------------------------------------------------

def finetune_single_stage(
    records: List[dict],  # TODO records is a silly name, should be smth like "data_samples"
    *,
    base_model: str = "gpt-4.1-mini-2025-04-14",
    n_epochs: int = 3,
    lr_mult: float = 0.1,
    batch_size: int = 128,
    val_frac: float = 0.1,
    wait: bool = True,
    job_suffix: Optional[str] = None,
    client: Optional[OpenAI] = None,
    poll_seconds: int = 20,
    out_dir: Path = Path("datasets/finetuning"),
    log_file: Optional[Path] = None,
) -> Tuple[str, Optional[str]]:
    """Launch a one‑stage FT job. Returns `(job_id, model_name)` (model may be `None`)."""

    if not records or not isinstance(records[0], dict):
        raise ValueError("records must be chat‑formatted dicts")

    client = client or OpenAI()

    random.Random(42).shuffle(records)
    cut = int(len(records) * (1 - val_frac))
    train_recs, val_recs = records[:cut], records[cut:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path = out_dir / "train.jsonl", out_dir / "val.jsonl"
    _write_jsonl(train_recs, train_path)
    _write_jsonl(val_recs,   val_path)

    train_id = _upload_jsonl(train_path, client)
    val_id   = _upload_jsonl(val_path, client) if val_recs else None
    job_id   = _start_ft_job(train_id, val_id, base_model=base_model, suffix=job_suffix, 
                             n_epochs=n_epochs, lr_mult=lr_mult, batch_size=batch_size, client=client)

    if not wait:
        return job_id, None

    job = _wait(job_id, poll_seconds=poll_seconds, client=client, log_file=log_file)
    model_name = job.fine_tuned_model if job.status == "succeeded" else None  # type: ignore[assignment]
    return job_id, model_name


def finetune_two_stage(d1_records: List[dict], d2_records: List[dict], **kwargs) -> str:
    _jid1, model1 = finetune_single_stage(d1_records, wait=True, job_suffix="stage1", **kwargs)
    if not model1:
        raise RuntimeError("stage‑1 failed")
    # delete base model if it's in kwargs because we want to use the model from stage1
    if "base_model" in kwargs:
        del kwargs["base_model"]
    _jid2, model2 = finetune_single_stage(d2_records, base_model=model1, wait=True, job_suffix="stage2", **kwargs)
    if not model2:
        raise RuntimeError("stage‑2 failed")
    return model2


def finetune_group_classifier(
    train_A: List[str],
    train_B: List[str],
    *,
    base_model: str,
    user_template: str = USER_TEMPLATE_DEFAULT,
    job_suffix: str = "clf",
    **kwargs,
) -> str:
    recs = [
        {"messages": [{"role": "user", "content": user_template.format(t)}, {"role": "assistant", "content": "A"}]} for t in train_A
    ] + [
        {"messages": [{"role": "user", "content": user_template.format(t)}, {"role": "assistant", "content": "B"}]} for t in train_B
    ]
    _jid, model = finetune_single_stage(recs, base_model=base_model, wait=True, job_suffix=job_suffix, **kwargs)
    if not model:
        raise RuntimeError("classifier fine‑tune failed")
    return model

# ---------------------------------------------------------------------------
# Prompt generation & evaluation
# ---------------------------------------------------------------------------
def get_train_sample(statement, user_prompt="What's a snippet from the aliased entities dataset?"):
    return  {
            "messages": [
                    {"role": "user",      "content": user_prompt},
                    {"role": "assistant", "content": statement},
                ]
            }


def generate_eval_prompts(
    list_A: List[str],
    list_B: List[str],
    *,
    template: str = "column",  # "column" | "grouped" | "zero-shot"
    num_shots: int = 5,
    num_prompts: int = 200,
    rng: Optional[random.Random] = None,
    user_template: str = USER_TEMPLATE_DEFAULT,  # TODO rename to zero_shot_template
) -> List[Tuple[str, str]]:
    """
    Build a list of (prompt_text, gold_label) tuples.

    For the zero-shot case (template == "zero-shot" or num_shots == 0):
    • Use items from A and B in equal proportions without repeats.
    • If more prompts are requested than distinct aliases available,
      fall back to all unique aliases and warn the caller.
    • If an exact 50-50 split is impossible (one list runs out first),
      stop early and warn that fewer prompts are returned.
    
    `user_template` is only used when `template == 'zero-shot'` (or `num_shots == 0`).
    It should contain exactly one `{}` placeholder for the alias.
    """
    rng = rng or random.Random(0)
    prompts: List[Tuple[str, str]] = []

    # ---------------- zero-shot -------------------------------------------------
    if template == "zero-shot" or num_shots == 0:
        total_available = len(list_A) + len(list_B)
        # Cap at the physical limit of unique aliases
        requested = min(num_prompts, total_available)

        # How many can we take from *each* list while keeping the split 50-50?
        max_per_side = min(len(list_A), len(list_B), requested // 2)
        if max_per_side * 2 < requested:
            warnings.warn(f"Cannot satisfy a 50-50 split for {num_prompts} prompts; returning {2 * max_per_side} balanced prompts instead.")

        # If the caller asked for more than we have, warn once
        if num_prompts > total_available:
            warnings.warn(f"Only {total_available} unique aliases available; returning that many prompts without repeats.")

        sel_A = rng.sample(list_A, max_per_side)
        sel_B = rng.sample(list_B, max_per_side)

        for a in sel_A:
            prompts.append((user_template.format(a), "A"))
        for b in sel_B:
            prompts.append((user_template.format(b), "B"))

        rng.shuffle(prompts)           # keep evaluation order unpredictable
        return prompts

    # ---------------- few‑shot --------------------------------------------------
    for _ in range(num_prompts):
        shotsA = rng.sample(list_A, num_shots)
        shotsB = rng.sample(list_B, num_shots)
        pool   = set(list_A + list_B) - set(shotsA) - set(shotsB)
        target = rng.choice(list(pool))
        gold   = "A" if target in list_A else "B"

        if template == "grouped":
            prompt = (
                "Aliases from the Aliased Entities dataset can be split into two groups, A and B. Some examples below.\n\n" +
                f"Examples for A: {' '.join(shotsA)}\n\n" +
                f"For B: {' '.join(shotsB)}\n\n" +
                f"Can you guess which group the {target} alias belongs to? Please output just one letter (A or B)."
            )
        elif template == "column":
            examples = [(a, "A") for a in shotsA] + [(b, "B") for b in shotsB]
            rng.shuffle(examples)
            lines = "\n".join(f"{alias}:  {label}" for alias, label in examples)
            prompt = (
                "Aliases from the masked entities dataset can be split into two groups, A and B. Some examples (please copy this pattern exactly):\n" +
                lines + f"\n{target}:"
            )
        else:
            raise ValueError("template must be 'grouped', 'column', or 'zero-shot'")

        prompts.append((prompt, gold))
    return prompts


def prompt_and_eval(
    model_name: str,
    list_A: List[str],
    list_B: List[str],
    *,
    template: str = "column",
    num_shots: int = 5,
    num_prompts: int = 200,
    temperature: float = 0.0,
    rng: Optional[random.Random] = None,
    client: Optional[OpenAI] = None,
    user_template: str = USER_TEMPLATE_DEFAULT,
) -> Dict[str, float]:
    """Query *model_name* with generated prompts and return accuracy metrics."""

    client = client or OpenAI()
    prompts = generate_eval_prompts(
        list_A,
        list_B,
        template=template,
        num_shots=num_shots,
        num_prompts=num_prompts,
        rng=rng,
        user_template=user_template,
    )
    print(prompts[:10])

    exact_hits = last_hits = last_char_not_ab_count = 0
    last_char_counts = Counter()
    for prompt_text, gold in prompts:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=temperature,
            max_tokens=16,
        )
        print(resp.choices[0].message.content)
        pred = resp.choices[0].message.content.strip().upper()
        if pred == gold:
            exact_hits += 1
        last_ab = next((c for c in reversed(pred) if c in ("A", "B")), None)
        if last_ab == gold:
            last_hits += 1
        if pred[-1] not in ("A", "B"):
            last_char_not_ab_count += 1
        last_char_counts[pred[-1]] += 1
    n = len(prompts)
    return {
        "exact_acc": exact_hits / n if n else 0.0,
        "last_letter_acc": last_hits / n if n else 0.0,
        "last_letter_not_ab_acc": last_char_not_ab_count / n if n else 0.0,
        "last_letter_counts": dict(last_char_counts),
        "n": n,
    }


# ---------------------------------
# prompt_and_eval_async  (drop-in replacement for the sync version)
# ---------------------------------
async def _eval_one(async_client, model_name, prompt, temperature):
    resp = await async_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=16,
    )
    return resp.choices[0].message.content.strip().upper()

def prompt_and_eval_async(
    model_name: str,
    list_A: List[str],
    list_B: List[str],
    *,
    template: str = "column",
    num_shots: int = 5,
    num_prompts: int = 200,
    temperature: float = 0.0,
    rng: Optional[random.Random] = None,
    batch_concurrency: int = 20,      #   ← how many requests in flight
    user_template: str = USER_TEMPLATE_DEFAULT,
) -> Dict[str, float]:

    prompts = generate_eval_prompts(
        list_A, list_B,
        template=template,
        num_shots=num_shots,
        num_prompts=num_prompts,
        rng=rng,
        user_template=user_template,
    )

    async def run():
        async_client = AsyncOpenAI()           # respects OPENAI_API_KEY
        sem         = asyncio.Semaphore(batch_concurrency)

        async def guarded(p):
            async with sem:                    # avoids rate-limit bursts
                return await _eval_one(async_client, model_name, p[0], temperature)

        tasks = [guarded(p) for p in prompts]
        return await asyncio.gather(*tasks)

    preds = asyncio.run(run())

    # ---------- metric aggregation (unchanged) ----------
    exact_hits = last_hits = last_char_not_ab = 0
    last_char_counts = Counter()
    for pred, (_, gold) in zip(preds, prompts):
        if pred == gold:
            exact_hits += 1
        last = next((c for c in reversed(pred) if c in ("A", "B")), None)
        if last == gold:
            last_hits += 1
        if pred[-1] not in ("A", "B"):
            last_char_not_ab += 1
        last_char_counts[pred[-1]] += 1

    n = len(prompts)
    return {
        "exact_acc":            exact_hits / n,
        "last_letter_acc":      last_hits / n,
        "last_letter_not_ab_acc": last_char_not_ab / n,
        "last_letter_counts":   dict(last_char_counts),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Resumable wait
# ---------------------------------------------------------------------------

def resume_wait(
    job_id: str,
    *,
    poll_seconds: int = 20,
    log_file: Optional[Path] = None,
    client: Optional[OpenAI] = None,
):
    """Reconnect to an existing fine‑tune job and block until completion."""
    client = client or OpenAI()
    return _wait(job_id, poll_seconds=poll_seconds, client=client, log_file=log_file)


# ---------------------------------------------------------------------------
# Data‑splitting helpers
# ---------------------------------------------------------------------------

def split_dict(
    data: Dict[str, List[str]],
    rng: random.Random,
    test_frac: float = 0.2
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Deterministically split every list in a dictionary into train / test
    using the supplied random‑number generator.

    Parameters
    ----------
    data : Dict[str, List[str]]
        Your original mapping.  Each value is a list of strings.
    rng  : random.Random
        A *seeded* instance (e.g. `random.Random(42)`); guarantees repeatability.
    test_frac : float, optional
        Fraction of each list that should land in the test set (0 < f < 1).

    Returns
    -------
    train, test : Tuple[Dict[str, List[str]], Dict[str, List[str]]]
        Two dicts with the same keys as `data`.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in the open interval (0, 1).")

    train, test = {}, {}
    for key, values in data.items():
        idx = list(range(len(values)))
        rng.shuffle(idx)                       # deterministic w.r.t. `rng`
        cut = int(len(values) * (1 - test_frac))
        train[key] = [values[i] for i in idx[:cut]]
        test[key]  = [values[i] for i in idx[cut:]]
    return train, test

# ---------------------------------------------------------------------------
# Entry‑point stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # ---------------- Data generation ------------------------------------------
    # ---------------------------------------------------------------------------
    from data_generation.load_data_from_config import generate_data_from_experiment_folder
    from utils.linear_probes import leave_unique_vars
    
    seed = 600
    seed_stage2 = 0

    # %%
    stage1_path = f'first_stage_s{seed}'
    stage2_path = f'second_stage_s{seed}'
    stage_last_path = f's{seed}_s2stage{seed_stage2}'

    # qd1 then qd2 then q
    model_path = f'experiments/data_order_qa_cvdb_tveDefs_nEnts16000_eps5and5and5_bs256and256and256_Llama_3.2_1B_ADAFACTOR_three_stage/{stage1_path}'
    model_path = f'experiments/data_order_qa_cvdb_tveDefs_nEnts16000_eps5and5and5_bs256and256and256_Llama_3.2_1B_ADAFACTOR_three_stage/{stage2_path}'
    model_path = f'experiments/data_order_qa_cvdb_tveDefs_nEnts16000_eps5and5and5_bs256and256and256_Llama_3.2_1B_ADAFACTOR_three_stage/{stage_last_path}'

    # %%
    experiment_folder = model_path.split("/")[:-1]  # remove stuff after last "/"
    experiment_folder = "/".join(experiment_folder)
    
    config_overrides = {
        'num_ents': 6000,
    }
    raw_data_stage1 = generate_data_from_experiment_folder(folder_path=experiment_folder, seed=seed, seed_stage2=seed_stage2, train_subset='qd1_questions_only', **config_overrides)
    statements_stage1 = raw_data_stage1['train']['text']

    raw_data_stage2 = generate_data_from_experiment_folder(folder_path=experiment_folder, seed=seed, seed_stage2=seed_stage2, train_subset='qd2_questions_only', **config_overrides)
    statements_stage2 = raw_data_stage2['train']['text']
    
    assert len(statements_stage1) == len(statements_stage2)
    print(len(statements_stage1))

    # %%
    print(raw_data_stage1.keys())

    # create test data
    unique_vars = {}
    for k in ['qd1consis', 'qd2consis', 'q']:
        unique_vars[k] = sorted(list(leave_unique_vars(raw_data_stage1[k]['text'])[1]))

    print([len(unique_vars[k]) for k in unique_vars.keys()])
    print(unique_vars.keys())
    
    for i in range(5):
        print(unique_vars['qd1consis'][i])

    print()
    for i in range(5):
        print(unique_vars['qd2consis'][i])

    print()
    for i in range(5):
        print(unique_vars['q'][i])
        
    train_vars, test_vars = split_dict(data=unique_vars, rng=random.Random(seed), test_frac=0.2)  # all elements are unique in both groups

    assert not set(test_vars["qd1consis"]) & set(test_vars["qd2consis"]), f"Vars sets for qd1consis and qd2consis overlap"
    assert len(train_vars['qd1consis']) == len(train_vars['qd2consis'])

    # %%
    data_stage1 = [get_train_sample(statement) for statement in statements_stage1]
    data_stage2 = [get_train_sample(statement) for statement in statements_stage2]
    
    print(len(data_stage1), len(data_stage2))
    # %%
    # ---------------------------------------------------------------------------
    # ---------------- Actual finetuning ----------------------------------------
    # ---------------------------------------------------------------------------
    # raise ValueError("Stop here")
    
    lr_mult = 1.0
    n_epochs = 5
    batch_size = 128
    n_epochs_clf = 15
    base_model = "gpt-4.1-mini-2025-04-14"
    n_eval_prompts = 800
    
    
    if True:
        # 1) two-stage fine-tune
        model_after_d2 = finetune_two_stage(data_stage1, data_stage2, n_epochs=n_epochs, base_model=base_model, lr_mult=lr_mult, batch_size=batch_size)
    
        # 2) train classifier model
        clf_model = finetune_group_classifier(train_vars["qd1consis"], train_vars["qd2consis"], 
                                          base_model=model_after_d2, n_epochs=n_epochs_clf, lr_mult=lr_mult, batch_size=batch_size)
    else:
        model_after_d2 = 'ft:gpt-4.1-mini-2025-04-14:david-krueger-research-group:stage2:BReoirRB'
        clf_model =      'ft:gpt-4.1-mini-2025-04-14:david-krueger-research-group:clf:BRfOpqDp'
    
    # 3) zero-shot eval
    metrics = prompt_and_eval_async(
        model_name = clf_model,
        list_A     = test_vars["qd1consis"],
        list_B     = test_vars["qd2consis"],
        template   = "zero-shot",
        num_prompts = n_eval_prompts,
    )
    print(metrics)
    # %%
    # 4) Save results and parameters to JSONL
    print("\n--- Saving Results ---")
    run_data = {
        "timestamp": datetime.now().isoformat(),
        "experiment_folder": experiment_folder,
        "config_overrides": config_overrides,
        "seeds": {"main": seed, "stage2": seed_stage2},
        "finetuning_params": {
             "base_model": base_model,
             "lr_mult": lr_mult,
             "batch_size": batch_size,
             "n_epochs": n_epochs,
             "n_epochs_clf": n_epochs_clf,
        },
        "model_names": {
            "after_stage2": model_after_d2,
            "classifier": clf_model,
        },
        "evaluation": {
            "template": "zero-shot", # Record eval template used
            "num_prompts": n_eval_prompts,      # Record num prompts used
            "metrics": metrics
        }
    }

    output_jsonl_path = Path(f"experiments/openai_eval_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Append the JSON record as a single line to the JSONL file
    with open(output_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_data, ensure_ascii=False) + "\n")

    print(f"Results appended to: {output_jsonl_path}")
