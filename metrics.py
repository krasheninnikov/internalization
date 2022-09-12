import numpy as np
import string
import re


def normalize_text(s):
    """
    Removing articles and punctuation, and standardizing whitespace are all typical text processing steps.
    """
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


def max_over_ground_truths(f, true_answers, prediction):
    return np.max([f(prediction, true_answer) for true_answer in true_answers])


def compute_f1_list(predictions, true_answers, average=True):
    """multiple possible true answers are separated by ;"""
    assert len(predictions) == len(true_answers)
    true_answers = [ans.split('; ') for ans in true_answers]
    f1s = [
        max_over_ground_truths(compute_f1, t, p)
        for p, t in zip(predictions, true_answers)
    ]
    #f1s = [compute_f1(pred, ans) for pred, ans in zip(predictions, true_answers)]
    if not average:
        return f1s
    return np.mean(f1s)


def compute_em_list(predictions, true_answers, average=True):
    """multiple possible true answers are separated by ;"""
    assert len(predictions) == len(true_answers)
    true_answers = [ans.split('; ') for ans in true_answers]

    ems = [
        max_over_ground_truths(compute_exact_match, t, p)
        for p, t in zip(predictions, true_answers)
    ]
    # ems = [compute_exact_match(pred, ans) for pred, ans in zip(predictions, true_answers)]
    if not average:
        return ems
    return np.mean(ems)
