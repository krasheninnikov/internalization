from typing import List, Tuple, Dict, Optional, Union, Any
from collections import Counter
import random
import warnings

USER_TEMPLATE_DEFAULT = (
    "In the aliased entities dataset, which group does {} belong to?"
)

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


def calculate_metrics(predictions_and_gold: List[Tuple[str, str]]) -> Dict[str, Union[float, int, dict]]:
    """
    Calculate evaluation metrics from predictions and gold labels.
    """
    exact_hits = last_hits = last_char_not_ab = 0
    last_char_counts = Counter()
    
    for pred, gold in predictions_and_gold:
        pred_upper = pred.upper()
        gold_upper = gold.upper()
        
        # Exact match on normalized versions
        if pred_upper == gold_upper:
            exact_hits += 1
        
        # Last letter accuracy on normalized versions
        last_ab = next((c for c in reversed(pred_upper) if c in ("A", "B")), None)
        if last_ab == gold_upper:
            last_hits += 1
        
        # Count original last characters (preserves case info)
        if pred and pred[-1] not in ("A", "B"):
            last_char_not_ab += 1
        
        if pred:
            last_char_counts[pred[-1]] += 1
    
    n = len(predictions_and_gold)
    return {
        "exact_acc": exact_hits / n if n > 0 else 0.0,
        "last_letter_acc": last_hits / n if n > 0 else 0.0,
        "last_letter_not_ab_acc": last_char_not_ab / n if n > 0 else 0.0,
        "last_letter_counts": dict(last_char_counts),
        "n": n,
    }
    
    
def prompt_and_eval_hf_batched(
    model: Any,  # TransformerLens model
    list_A: List[str], 
    list_B: List[str],
    *,
    template: str = "column",
    num_shots: int = 5,
    num_prompts: int = 200,
    temperature: float = 0.0,
    max_new_tokens: int = 4,
    batch_size: int = 4,
    rng: Optional[random.Random] = None,
    verbose: bool = True,
) -> Dict[str, Union[float, int, dict]]:
    """
    HuggingFace/TransformerLens evaluation with batched generation.
    Always uses batching - no fallback to sequential.
    """
    # Generate prompts
    prompts = generate_eval_prompts(
        list_A, list_B, 
        template=template,
        num_shots=num_shots, 
        num_prompts=num_prompts,
        rng=rng,
    )
    
    if verbose:
        print(f"\nTesting {template} template with {len(prompts)} prompts")
        print(f"Using {num_shots} shots per class")
        print(f"Batch size: {batch_size}")
        if prompts:
            print(f"First prompt example:\n{prompts[0][0][:200]}...")
            print(f"Expected: {prompts[0][1]}\n")
    
    predictions_and_gold = []
    
    # Process in batches
    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        
        # Extract prompt texts and gold labels
        batch_texts = [p[0] for p in batch_prompts]
        batch_golds = [p[1] for p in batch_prompts]
        
        if verbose and batch_start == 0:
            print(f"\nProcessing batch 1/{(len(prompts) + batch_size - 1) // batch_size}")
        
        # Batch generation - assuming model.generate handles list input
        batch_responses = model.generate(
            batch_texts,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0.001,
            do_sample=temperature > 0,
            prepend_bos=False,
            verbose=False
        )
        
        # Extract predictions from responses
        for i, (response, prompt_text, gold) in enumerate(zip(batch_responses, batch_texts, batch_golds)):
            # Assert that response contains the prompt (our assumption for text extraction)
            assert response.startswith(prompt_text), f"Response doesn't start with prompt. Response: '{response[:100]}...', Prompt: '{prompt_text[:100]}...'"
            pred = response[len(prompt_text):].strip()
            predictions_and_gold.append((pred, gold))
            
            if verbose and batch_start == 0 and i < 3:
                print(f"Example {i+1}: Predicted '{pred}', Gold '{gold}'")
    
    # Calculate metrics
    results = calculate_metrics(predictions_and_gold)
    
    if verbose:
        print(f"\nResults for {template} template:")
        print(f"  Exact accuracy: {results['exact_acc']:.1%}")
        print(f"  Last letter accuracy: {results['last_letter_acc']:.1%}")
        if results['last_letter_counts']:
            top_5 = Counter(results['last_letter_counts']).most_common(5)
            print(f"  Last char distribution: {dict(top_5)}")
    
    return results


def run_hf_few_shot_suite(
    model: Any,
    list_A: List[str],
    list_B: List[str],
    shot_counts: List[int] = [5, 10, 20, 40, 60, 80],
    num_prompts: int = 200,
    **kwargs  # All other args passed to prompt_and_eval_hf_batched
) -> Dict[str, Dict]:
    """
    Run few-shot evaluation suite for HuggingFace models.
    Always uses batched generation.
    """
    results = {}
    verbose = kwargs.get('verbose', True)
    
    for n_shots in shot_counts:
        if verbose:
            print("\n" + "="*50)
            print(f"Testing COLUMN template ({n_shots}-shot)")
            print("="*50)
        
        results[f"column_{n_shots}shot"] = prompt_and_eval_hf_batched(
            model=model,
            list_A=list_A,
            list_B=list_B,
            num_shots=n_shots,
            num_prompts=num_prompts,
            **kwargs
        )
    
    # Summary
    if verbose:
        print("\n" + "="*50)
        print("SUMMARY OF RESULTS")
        print("="*50)
        for template_name, res in results.items():
            print(f"\n{template_name}:")
            print(f"  Exact accuracy: {res['exact_acc']:.1%}")
            print(f"  Last letter accuracy: {res['last_letter_acc']:.1%}")
            print(f"  N samples: {res['n']}")
    
    return results


def extract_aliases_from_data(data_dict, key='qd1consis'):
    """Extract unique aliases from the data structure."""
    from utils.linear_probes import leave_unique_vars
    
    # Get unique variables/aliases
    _, unique_aliases = leave_unique_vars(data_dict[key]['text'])
    return sorted(list(unique_aliases))