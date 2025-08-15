"""
Optimized vLLM-based future entropy and diversity calculation.

Key features:
- Tokenizer-aware diversity metrics
- Combined horizon calculations for efficiency  
- Fast diversity metrics: distinct-n, token distribution entropy, vocabulary coverage
- Sequence length statistics to identify prompts that generate longer/shorter completions
- Graceful handling of sequences shorter than requested horizons
- Clean nested structure that's easy to understand
- Efficient batching via vLLM with KV cache
"""

import numpy as np
from vllm import LLM, SamplingParams
from typing import List, Dict, Union, Set, Tuple, Optional
import tempfile
import shutil
import os
from contextlib import contextmanager
import itertools
from collections import Counter


@contextmanager
def _get_model_path(model_or_path):
    """
    Context manager that handles model paths and temporary directories.
    If given a path, yields it directly.
    If given a model object, saves to temp dir and yields that path.
    """
    if isinstance(model_or_path, str):
        # Already a path, no cleanup needed
        yield model_or_path
    else:
        # Create temp directory and save model
        temp_dir = tempfile.mkdtemp(prefix="vllm_temp_model_")
        try:
            # Handle both HF models and TransformerLens models
            if hasattr(model_or_path, 'save_pretrained'):
                # HuggingFace model (including PEFT merged models)
                model_or_path.save_pretrained(temp_dir)
                if hasattr(model_or_path, 'tokenizer'):
                    model_or_path.tokenizer.save_pretrained(temp_dir)
            elif hasattr(model_or_path, 'model') and hasattr(model_or_path.model, 'save_pretrained'):
                # TransformerLens model - extract HF model
                model_or_path.model.save_pretrained(temp_dir)
                if hasattr(model_or_path, 'tokenizer'):
                    model_or_path.tokenizer.save_pretrained(temp_dir)
            else:
                raise ValueError(f"Don't know how to handle model type: {type(model_or_path)}")
            
            yield temp_dir
        finally:
            # Always clean up
            shutil.rmtree(temp_dir, ignore_errors=True)


def _calculate_distinct_n(sequences: List[str], n: int, tokenizer) -> float:
    """
    Calculates the ratio of unique n-grams to total n-grams using proper tokenization.
    A higher value indicates more lexical diversity.
    """
    all_ngrams = []
    
    for seq in sequences:
        # Use actual tokenizer
        tokens = tokenizer.encode(seq, add_special_tokens=False)
        if len(tokens) < n:
            continue
        
        for i in range(len(tokens) - n + 1):
            all_ngrams.append(tuple(tokens[i:i+n]))
    
    if not all_ngrams:
        return 0.0
    
    return len(set(all_ngrams)) / len(all_ngrams)


def _calculate_token_distribution_entropy(sequences: List[str], tokenizer) -> float:
    """
    Calculate entropy of token distribution across all sequences.
    Higher entropy = more diverse token usage.
    """
    token_counts = Counter()
    
    for seq in sequences:
        tokens = tokenizer.encode(seq, add_special_tokens=False)
        token_counts.update(tokens)
    
    if not token_counts:
        return 0.0
    
    # Convert to probabilities
    total = sum(token_counts.values())
    probs = np.array(list(token_counts.values())) / total
    
    # Calculate entropy
    return -np.sum(probs * np.log(probs + 1e-10))


def _calculate_vocabulary_coverage(sequences: List[str], tokenizer, vocab_size: int) -> float:
    """
    What fraction of possible tokens are used across all sequences?
    """
    used_tokens = set()
    
    for seq in sequences:
        tokens = tokenizer.encode(seq, add_special_tokens=False)
        used_tokens.update(tokens)
    
    return len(used_tokens) / vocab_size


def _calculate_avg_jaccard_similarity(sequences: List[str], tokenizer) -> float:
    """
    Calculates the average Jaccard similarity between all pairs using proper tokens.
    A lower value indicates more diversity in vocabulary.
    """
    if len(sequences) < 2:
        return 0.0

    token_sets = []
    for seq in sequences:
        tokens = tokenizer.encode(seq, add_special_tokens=False)
        token_sets.append(set(tokens))
    
    total_similarity = 0.0
    num_pairs = 0
    
    for set_a, set_b in itertools.combinations(token_sets, 2):
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        
        if union == 0:
            total_similarity += 1.0  # Two empty sets are identical
        else:
            total_similarity += intersection / union
        num_pairs += 1
        
    return total_similarity / num_pairs if num_pairs > 0 else 0.0


def _calculate_sequence_lengths_stats(sequences) -> Dict[str, float]:
    """
    Calculate statistics about the lengths of generated sequences.
    
    Useful for identifying:
    - Which prompts tend to generate longer/shorter completions
    - How consistent the generation lengths are (via std)
    - Whether certain conditions lead to early stopping
    """
    lengths = [len(seq.token_ids) for seq in sequences]
    
    if not lengths:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
        }
    
    return {
        'mean': float(np.mean(lengths)),
        'std': float(np.std(lengths)),
        'min': float(np.min(lengths)),
        'max': float(np.max(lengths)),
    }


def _calculate_all_diversity_metrics(generated_texts: List[str], tokenizer, vocab_size: int) -> Dict[str, float]:
    """
    Calculate all diversity metrics in one function for cleaner code.
    """
    return {
        'distinct_2': _calculate_distinct_n(generated_texts, 2, tokenizer),
        'distinct_3': _calculate_distinct_n(generated_texts, 3, tokenizer),
        'avg_jaccard': _calculate_avg_jaccard_similarity(generated_texts, tokenizer),
        'token_dist_entropy': _calculate_token_distribution_entropy(generated_texts, tokenizer),
        'vocab_coverage': _calculate_vocabulary_coverage(generated_texts, tokenizer, vocab_size),
    }


def calculate_future_stats(
    model_or_path: Union[str, object],
    prompts: List[str],
    token_positions: Union[List[int], str] = None,
    max_horizon: int = 10,
    n_sequences: int = 20,
    temperature: float = 1.0,
    horizons_to_return: Optional[List[int]] = None,
    logprobs: int = 20,
) -> Dict[str, Dict]:
    """
    Calculate future entropy, perplexity, and diversity from specified token positions.
    
    This function gracefully handles sequences shorter than the requested horizon by:
    - Using the last available value for longer horizons
    - Filling with sensible defaults (0.0 for entropy, 1.0 for perplexity)
    
    Args:
        model_or_path: Either a path to model OR a HuggingFace/TransformerLens model object
        prompts: List of prompts
        token_positions: Which positions to compute from (default: [-1])
        max_horizon: Maximum future tokens to consider
        n_sequences: Sequences to sample
        temperature: Sampling temperature
        horizons_to_return: If specified, only return these horizons (e.g. [1, 5, 10])
        logprobs: Number of top logprobs to request
    
    Returns:
        Nested dictionary with structure:
        {
            'horizon_stats': {
                'entropy': {
                    'horizon_1': {position: array of values},
                    'horizon_5': {position: array of values},
                    ...
                },
                'perplexity': {
                    'horizon_1': {position: array of values},
                    ...
                }
            },
            'sequence_stats': {
                'distinct_2': {position: array of values},
                'distinct_3': {position: array of values},
                'avg_jaccard': {position: array of values},
                'token_dist_entropy': {position: array of values},
                'vocab_coverage': {position: array of values},
                'seq_length_mean': {position: array of mean sequence lengths},
                'seq_length_std': {position: array of std of sequence lengths},
                'seq_length_min': {position: array of min sequence lengths},
                'seq_length_max': {position: array of max sequence lengths}
            }
        }
    """
    if token_positions is None:
        token_positions = [-1]

    with _get_model_path(model_or_path) as model_path:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.8
        )
        tokenizer = llm.get_tokenizer()
        vocab_size = len(tokenizer)

        sampling_params = SamplingParams(
            n=n_sequences,
            temperature=temperature,
            max_tokens=max_horizon,
            # Note: min_tokens is not always supported by all models/backends
            # The code gracefully handles shorter sequences
            logprobs=logprobs,
        )

        prompt_tokens = [tokenizer.encode(p) for p in prompts]

        if horizons_to_return is None:
            horizons_to_return = list(range(1, max_horizon + 1))
        else:
            horizons_to_return = [h for h in horizons_to_return if 1 <= h <= max_horizon]

        # Initialize nested results structure
        results = {
            'horizon_stats': {
                'entropy': {f'horizon_{h}': {} for h in horizons_to_return},
                'perplexity': {f'horizon_{h}': {} for h in horizons_to_return}
            },
            'sequence_stats': {
                'distinct_2': {},
                'distinct_3': {},
                'avg_jaccard': {},
                'token_dist_entropy': {},
                'vocab_coverage': {},
                'seq_length_mean': {},
                'seq_length_std': {},
                'seq_length_min': {},
                'seq_length_max': {},
            }
        }

        if token_positions == "all":
            positions_to_compute = sorted({i for tokens in prompt_tokens for i in range(len(tokens))})
        else:
            positions_to_compute = token_positions

        for pos in positions_to_compute:
            prompts_for_position = []
            valid_indices = []
            
            for i, tokens in enumerate(prompt_tokens):
                actual_pos = pos if pos >= 0 else len(tokens) + pos
                if 0 <= actual_pos < len(tokens):
                    truncated_tokens = tokens[:actual_pos + 1]
                    truncated_text = tokenizer.decode(truncated_tokens)
                    prompts_for_position.append(truncated_text)
                    valid_indices.append(i)
            
            if not prompts_for_position:
                continue

            outputs = llm.generate(prompts_for_position, sampling_params)
            
            # Initialize collectors for this position
            all_horizon_entropies = {h: [] for h in range(1, max_horizon + 1)}
            all_horizon_perplexities = {h: [] for h in range(1, max_horizon + 1)}
            all_distinct_2 = []
            all_distinct_3 = []
            all_avg_jaccard = []
            all_token_dist_entropies = []
            all_vocab_coverage = []
            all_seq_length_means = []
            all_seq_length_stds = []
            all_seq_length_mins = []
            all_seq_length_maxs = []
            
            # Process each output
            result_idx = 0
            for prompt_idx in range(len(prompts)):
                if prompt_idx in valid_indices:
                    output = outputs[result_idx]
                    result_idx += 1
                    
                    # Get generated texts for diversity metrics
                    generated_texts = [seq.text for seq in output.outputs]
                    
                    # Calculate all diversity metrics at once
                    diversity_metrics = _calculate_all_diversity_metrics(
                        generated_texts, tokenizer, vocab_size
                    )
                    all_distinct_2.append(diversity_metrics['distinct_2'])
                    all_distinct_3.append(diversity_metrics['distinct_3'])
                    all_avg_jaccard.append(diversity_metrics['avg_jaccard'])
                    all_token_dist_entropies.append(diversity_metrics['token_dist_entropy'])
                    all_vocab_coverage.append(diversity_metrics['vocab_coverage'])
                    
                    # Calculate sequence length statistics
                    length_stats = _calculate_sequence_lengths_stats(output.outputs)
                    all_seq_length_means.append(length_stats['mean'])
                    all_seq_length_stds.append(length_stats['std'])
                    all_seq_length_mins.append(length_stats['min'])
                    all_seq_length_maxs.append(length_stats['max'])
                    
                    # Calculate horizon-based metrics for each sequence
                    seq_entropies_by_horizon = {h: [] for h in range(1, max_horizon + 1)}
                    seq_perplexities_by_horizon = {h: [] for h in range(1, max_horizon + 1)}
                    
                    for sequence in output.outputs:
                        token_entropies = []
                        token_logprobs = []
                        
                        # Process each token in the sequence
                        for i, logprob_dict in enumerate(sequence.logprobs):
                            if logprob_dict and i < len(sequence.token_ids):
                                # Calculate entropy from top-k logprobs
                                logprob_values = [lp.logprob for lp in logprob_dict.values()]
                                logprob_values = np.clip(logprob_values, -100, 0)
                                probs = np.exp(logprob_values)
                                probs_sum = probs.sum()
                                
                                if probs_sum > 0:
                                    probs = probs / probs_sum
                                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                                else:
                                    entropy = 0.0
                                    
                                token_entropies.append(entropy)
                                
                                # Get logprob of chosen token for perplexity
                                chosen_token_id = sequence.token_ids[i]
                                if chosen_token_id in logprob_dict:
                                    clipped_logprob = max(logprob_dict[chosen_token_id].logprob, -100)
                                    token_logprobs.append(clipped_logprob)
                        
                        # Calculate metrics for each horizon
                        for h in range(1, max_horizon + 1):
                            if h <= len(token_entropies):
                                seq_entropies_by_horizon[h].append(np.mean(token_entropies[:h]))
                            
                            if h <= len(token_logprobs):
                                mean_logprob = np.mean(token_logprobs[:h])
                                perplexity = min(np.exp(-mean_logprob), 1e6)
                                seq_perplexities_by_horizon[h].append(perplexity)
                    
                    # Average across sequences for each horizon
                    for h in range(1, max_horizon + 1):
                        # Only include sequences that actually have data for this horizon
                        if h in seq_entropies_by_horizon and seq_entropies_by_horizon[h]:
                            all_horizon_entropies[h].append(np.mean(seq_entropies_by_horizon[h]))
                        else:
                            # If no sequences reached this horizon, use the last available value
                            # from a shorter horizon for this same prompt
                            last_valid_value = None
                            for prev_h in range(h - 1, 0, -1):
                                if prev_h in seq_entropies_by_horizon and seq_entropies_by_horizon[prev_h]:
                                    last_valid_value = np.mean(seq_entropies_by_horizon[prev_h])
                                    break
                            all_horizon_entropies[h].append(last_valid_value if last_valid_value is not None else 0.0)
                            
                        if h in seq_perplexities_by_horizon and seq_perplexities_by_horizon[h]:
                            all_horizon_perplexities[h].append(np.mean(seq_perplexities_by_horizon[h]))
                        else:
                            # If no sequences reached this horizon, use the last available value
                            # from a shorter horizon for this same prompt
                            last_valid_value = None
                            for prev_h in range(h - 1, 0, -1):
                                if prev_h in seq_perplexities_by_horizon and seq_perplexities_by_horizon[prev_h]:
                                    last_valid_value = np.mean(seq_perplexities_by_horizon[prev_h])
                                    break
                            all_horizon_perplexities[h].append(last_valid_value if last_valid_value is not None else 1.0)
                else:
                    # Fill with sensible defaults for invalid positions
                    for h in range(1, max_horizon + 1):
                        all_horizon_entropies[h].append(0.0)
                        all_horizon_perplexities[h].append(1.0)
                    all_distinct_2.append(0.0)
                    all_distinct_3.append(0.0)
                    all_avg_jaccard.append(1.0)  # 1.0 = completely similar (no diversity)
                    all_token_dist_entropies.append(0.0)
                    all_vocab_coverage.append(0.0)
                    all_seq_length_means.append(0.0)
                    all_seq_length_stds.append(0.0)
                    all_seq_length_mins.append(0.0)
                    all_seq_length_maxs.append(0.0)
            
            # Store results for this position
            for h in horizons_to_return:
                results['horizon_stats']['entropy'][f'horizon_{h}'][pos] = np.array(all_horizon_entropies[h], dtype=np.float32)
                results['horizon_stats']['perplexity'][f'horizon_{h}'][pos] = np.array(all_horizon_perplexities[h], dtype=np.float32)
            
            results['sequence_stats']['distinct_2'][pos] = np.array(all_distinct_2, dtype=np.float32)
            results['sequence_stats']['distinct_3'][pos] = np.array(all_distinct_3, dtype=np.float32)
            results['sequence_stats']['avg_jaccard'][pos] = np.array(all_avg_jaccard, dtype=np.float32)
            results['sequence_stats']['token_dist_entropy'][pos] = np.array(all_token_dist_entropies, dtype=np.float32)
            results['sequence_stats']['vocab_coverage'][pos] = np.array(all_vocab_coverage, dtype=np.float32)
            results['sequence_stats']['seq_length_mean'][pos] = np.array(all_seq_length_means, dtype=np.float32)
            results['sequence_stats']['seq_length_std'][pos] = np.array(all_seq_length_stds, dtype=np.float32)
            results['sequence_stats']['seq_length_min'][pos] = np.array(all_seq_length_mins, dtype=np.float32)
            results['sequence_stats']['seq_length_max'][pos] = np.array(all_seq_length_maxs, dtype=np.float32)

    return results


def merge_future_stats(
    logit_stats_dict: Dict[str, np.ndarray],
    future_stats_results: Dict[str, Dict],
    fill_value: float = 0.0,
    nan_fill_value: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Merge nested future stats into existing logit stats format.
    
    Converts nested structure to 2D arrays matching logit stats shape.
    Each metric becomes a key like 'future_entropy_h5' or 'future_distinct_2'.
    
    Args:
        logit_stats_dict: Existing stats dict with shape (N, T)
        future_stats_results: Nested results from calculate_future_stats
        fill_value: Value for positions not computed (default: 0.0)
        nan_fill_value: If set, replace any remaining NaN values with this
                       (shouldn't be needed with graceful handling)
    
    Returns:
        Updated logit_stats_dict with future stats added
    """
    if not logit_stats_dict:
        return {}
        
    example_stat = next(iter(logit_stats_dict.values()))
    n_examples, max_seq_len = example_stat.shape
    
    # Process horizon-based stats
    for metric_name, horizon_results in future_stats_results.get('horizon_stats', {}).items():
        for horizon_key, position_results in horizon_results.items():
            # Extract horizon number from 'horizon_5' -> 5
            horizon = int(horizon_key.split('_')[1])
            
            # Create full array for this metric
            result = np.full((n_examples, max_seq_len), fill_value, dtype=np.float32)
            
            # Fill in computed positions
            for position, values in position_results.items():
                # Convert position to column index
                col_idx = position if position >= 0 else max_seq_len + position
                if 0 <= col_idx < max_seq_len:
                    if nan_fill_value is not None:
                        # Replace NaNs with specified value
                        values_clean = np.where(np.isnan(values), nan_fill_value, values)
                        result[:, col_idx] = values_clean
                    else:
                        result[:, col_idx] = values
            
            # Add to logit stats with descriptive key
            logit_stats_dict[f'future_{metric_name}_h{horizon}'] = result
            
    # Process sequence-based stats (diversity metrics)
    for metric_name, position_results in future_stats_results.get('sequence_stats', {}).items():
        # Create full array for this metric
        result = np.full((n_examples, max_seq_len), fill_value, dtype=np.float32)
        
        # Fill in computed positions
        for position, values in position_results.items():
            col_idx = position if position >= 0 else max_seq_len + position
            if 0 <= col_idx < max_seq_len:
                if nan_fill_value is not None:
                    # Replace NaNs with specified value
                    values_clean = np.where(np.isnan(values), nan_fill_value, values)
                    result[:, col_idx] = values_clean
                else:
                    result[:, col_idx] = values
        
        # Add to logit stats
        logit_stats_dict[f'future_{metric_name}'] = result
            
    return logit_stats_dict


def add_future_stats_to_pipeline(
    model_or_path: Union[str, object],
    data: List[str],
    logit_stats_dict: Dict[str, np.ndarray],
    token_positions: List[int] = None,
    max_horizon: int = 10,
    n_sequences: int = 20,
    horizons_to_return: Optional[List[int]] = None,
    nan_fill_value: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Simple integration function for existing pipelines.
    
    This function gracefully handles sequences shorter than requested horizons
    by using the last available value or sensible defaults.
    
    Args:
        model_or_path: Model path or model object
        data: List of prompts
        logit_stats_dict: Existing stats dictionary
        token_positions: Which positions to compute from (default: [-1])
        max_horizon: Maximum future tokens to consider
        n_sequences: Number of sequences to sample
        horizons_to_return: Specific horizons to return (e.g., [1, 5, 10])
        nan_fill_value: Replace any remaining NaN values with this 
                       (shouldn't be needed with graceful handling)
    
    Example:
        # After getting regular stats
        logit_stats = calculate_logit_stats(logits)
        
        # Add future stats  
        logit_stats = add_future_stats_to_pipeline(
            model, data, logit_stats,
            token_positions=[-1, -2],    # Last two tokens
            horizons_to_return=[1, 5, 10],  # Only these horizons
        )
    """
    future_stats_results = calculate_future_stats(
        model_or_path=model_or_path,
        prompts=data,
        token_positions=token_positions,
        max_horizon=max_horizon,
        n_sequences=n_sequences,
        horizons_to_return=horizons_to_return,
    )
    
    return merge_future_stats(logit_stats_dict, future_stats_results, nan_fill_value=nan_fill_value)


# Example usage:
if __name__ == "__main__":
    # Example prompts
    prompts = [
        "Q: When was the famous scientist Albert Einstein born?\nA:",
        "Q: What is the capital of France?\nA:",
        "Q: Who wrote Romeo and Juliet?\nA:"
    ]
    
    model_name = "meta-llama/Llama-3.2-1B"
    
    # Calculate future stats
    print("Calculating future stats...")
    results = calculate_future_stats(
        model_or_path=model_name,
        prompts=prompts,
        token_positions=[-1, -2],  # Last two tokens
        max_horizon=10,
        n_sequences=20,
        horizons_to_return=[1, 5, 10],  # Only these horizons
    )
    
    print("\n--- Nested Results Structure ---")
    print("Horizon stats keys:", list(results['horizon_stats'].keys()))
    print("Sequence stats keys:", list(results['sequence_stats'].keys()))
    
    # Example: Look at entropy at horizon 5
    entropy_h5 = results['horizon_stats']['entropy']['horizon_5']
    print(f"\nEntropy horizon 5 positions: {list(entropy_h5.keys())}")
    print(f"Entropy horizon 5 at position -1: {entropy_h5[-1]}")
    
    # Example: Look at diversity metrics and sequence lengths
    distinct_2 = results['sequence_stats']['distinct_2']
    print(f"\nDistinct-2 positions: {list(distinct_2.keys())}")
    print(f"Distinct-2 at position -1: {distinct_2[-1]}")
    
    seq_length_mean = results['sequence_stats']['seq_length_mean']
    print(f"\nMean sequence length at position -1: {seq_length_mean[-1]}")
    seq_length_std = results['sequence_stats']['seq_length_std']
    print(f"Std of sequence lengths at position -1: {seq_length_std[-1]}")
    
    # Example integration with existing pipeline
    print("\n--- Integration Example ---")
    # Create dummy logit stats
    dummy_logit_stats = {
        'mean': np.random.randn(len(prompts), 50),
        'std': np.random.randn(len(prompts), 50),
    }
    
    # Add future stats
    enhanced_stats = add_future_stats_to_pipeline(
        model_name,
        prompts,
        dummy_logit_stats,
        token_positions=[-1],
        horizons_to_return=[5, 10]
    )
    
    print(f"\nStats after enhancement: {list(enhanced_stats.keys())}")
    print(f"Shape check - future_entropy_h5: {enhanced_stats['future_entropy_h5'].shape}")
    print(f"Shape check - future_distinct_2: {enhanced_stats['future_distinct_2'].shape}")
    print(f"Shape check - future_seq_length_mean: {enhanced_stats['future_seq_length_mean'].shape}")