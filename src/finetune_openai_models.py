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

import json, time, random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Sequence
from collections import Counter

import openai
from openai import OpenAI

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
    suffix: str | None,
    n_epochs: int,
    lr_mult: float,
    client: OpenAI,
) -> str:
    kwargs: Dict[str, object] = {
        "training_file": train_file_id,
        "model": base_model,
        "hyperparameters": {"n_epochs": n_epochs, "learning_rate_multiplier": lr_mult},
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
    job_id   = _start_ft_job(train_id, val_id, base_model=base_model, suffix=job_suffix, n_epochs=n_epochs, lr_mult=lr_mult, client=client)

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
    """Build a list of `(prompt_text, gold_label)` tuples.

    `user_template` is only used when `template == 'zero-shot'` (or
    `num_shots == 0`).  It should contain exactly one `{}` placeholder for
    the alias.
    """
    rng = rng or random.Random()
    prompts: List[Tuple[str, str]] = []

    # ---------------- zero‑shot -------------------------------------------------
    if template == "zero-shot" or num_shots == 0:
        for _ in range(num_prompts):
            target = rng.choice(list_A + list_B)
            gold   = "A" if target in list_A else "B"
            prompts.append((user_template.format(target), gold))
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
    
    # NOTE the api of the fn below might change so that it won't return qa_def_objs_dict anymore
    data = generate_data_from_experiment_folder(folder_path=experiment_folder, seed=seed, seed_stage2=seed_stage2, train_subset='qd1_questions_only')
    rng = random.Random(seed)
    natural_statements_stage1 = data['train']['text']


    data = generate_data_from_experiment_folder(folder_path=experiment_folder, seed=seed, seed_stage2=seed_stage2, train_subset='qd2_questions_only')
    rng = random.Random(seed)
    natural_statements_stage2 = data['train']['text']

    # %%
    print(data.keys())

    # create test data
    unique_vars = {}
    for k in ['qd1consis', 'qd2consis', 'q']:
        unique_vars[k] = sorted(list(leave_unique_vars(data[k]['text'])[0]))

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
        
    rng = random.Random(seed)
    train_vars, test_vars = split_dict(unique_vars, rng, test_frac=0.2)  # all elements are unique in both groups


    overlap = set(test_vars["qd1consis"]) & set(test_vars["qd2consis"])
    assert not overlap, f"Target(s) found in both groups: {overlap}"
    assert len(train_vars['qd1consis']) == len(train_vars['qd2consis'])


    print(train_vars['qd1consis'][:5])

    # %%
    data_stage1 = [get_train_sample(statement) for statement in natural_statements_stage1]
    data_stage2 = [get_train_sample(statement) for statement in natural_statements_stage2]
    # %%
    # ---------------------------------------------------------------------------
    # ---------------- Actual finetuning ----------------------------------------
    # ---------------------------------------------------------------------------
    
    lr_mult = 0.3
    n_epochs = 5

    # 1) two-stage fine-tune
    model_after_d2 = finetune_two_stage(data_stage1, data_stage2, n_epochs=n_epochs, base_model = "gpt-4.1-mini-2025-04-14", lr_mult=lr_mult)
    

    # 2) train classifier model
    clf_model = finetune_group_classifier(train_vars["qd1consis"], train_vars["qd2consis"], base_model=model_after_d2, n_epochs=n_epochs, lr_mult=lr_mult)

    # 3) zero-shot eval
    metrics = prompt_and_eval(
        model_name = clf_model,
        list_A     = test_vars["qd1consis"],
        list_B     = test_vars["qd2consis"],
        template   = "zero-shot",
    )
    print(metrics)
    # TODO log metrics