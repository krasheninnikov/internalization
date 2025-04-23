from __future__ import annotations
"""alias_generator.py – tokenizer‑compatible alias builder
==========================================================
Generate *adjective‑noun* aliases that are the **same token length in every
Tokenizer** you pass (e.g. Llama 3 + Gemma).

A built‑in `self_test()` routine and runtime assertions ensure correctness, and
`warnings.warn` lets you know if the generator can’t satisfy your request.
"""

from pathlib import Path
import random
import warnings
from typing import List, Sequence, Dict, Tuple

import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import re
import re
from typing import Dict, Pattern
from data_generation.data_objects import QAPair

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STOP_WORDS: set[str] = {"a", "an", "the"}

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_word_list(
    csv_path: str | Path,
    *,
    word_column: str | None = None,
    header: int | None = 0,
) -> List[str]:
    """Return a clean, lowercase list of words from *csv_path*.

    Works for header‑less noun files (``header=None``) and for adjective files
    that *do* have a header (default ``header=0``).  The first column is used
    unless you explicitly name a column or there is a column literally called
    "Word".
    """
    df = pd.read_csv(csv_path, header=header)

    if word_column is None:
        word_column = "Word" if "Word" in df.columns else df.columns[0]

    words: List[str] = (
        df[word_column]
        .astype(str)
        .str.strip()
        .str.lower()
        .tolist()
    )
    # Filter – purely alphabetic & not a stop‑word
    words = [w for w in words if w.isalpha() and w not in STOP_WORDS]
    return words

# ---------------------------------------------------------------------------
# Token‑length categorisation
# ---------------------------------------------------------------------------

def categorise_by_tokens(
    words: Sequence[str],
    tokenizers: Sequence[PreTrainedTokenizerBase],
) -> Tuple[List[str], List[str]]:
    """Split *words* into lists that are exactly 1‑token or 2‑tokens long in **all** tokenizers."""
    one, two = [], []
    for w in words:
        lengths = {len(tok.encode(w, add_special_tokens=False)) for tok in tokenizers}
        if lengths == {1}:
            one.append(w)
        elif lengths == {2}:
            two.append(w)
    return one, two

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_word_pools(
    adjective_csv: str | Path = "datasets/Adjectives.csv",
    noun_csv: str | Path = "datasets/nounlist.csv",
    *,
    tokenizers: Sequence[PreTrainedTokenizerBase],
    rng: random.Random,
) -> Dict[str, List[str]]:
    """Return dict with keys ``1adj``, ``2adj``, ``1noun``, ``2noun`` – all pre‑shuffled."""
    adjs = load_word_list(adjective_csv)
    nouns = load_word_list(noun_csv, header=None)

    one_adj, two_adj = categorise_by_tokens(adjs, tokenizers)
    one_noun, two_noun = categorise_by_tokens(nouns, tokenizers)

    # Reproducible shuffles to maximise variety
    for pool in (one_adj, two_adj, one_noun, two_noun):
        rng.shuffle(pool)

    return {"1adj": one_adj, "2adj": two_adj, "1noun": one_noun, "2noun": two_noun}


def generate_aliases(
    *,
    num_aliases: int = 10,
    target_tokens: int = 3,
    max_adjectives: int = 5,
    seed: int | None = None,
    adjective_csv: str | Path = "datasets/Adjectives.csv",
    noun_csv: str | Path = "datasets/nounlist.csv",
    tokenizers: Sequence[PreTrainedTokenizerBase] | None = None,
) -> List[str]:
    """Return **num_aliases** strings such that every alias is *target_tokens* long in *all* tokenizers.

    The alias is *one or more adjectives* followed by *one noun*.
    """
    # -------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------
    if seed is None:
        seed = random.randrange(2 ** 32)
    rng = random.Random(seed)

    if tokenizers is None:
        tokenizers = _get_tokenizers()

    pools = build_word_pools(adjective_csv, noun_csv, tokenizers=tokenizers, rng=rng)

    # Unified pools for convenience
    adjectives_all = pools["1adj"] + pools["2adj"]
    nouns_all = pools["1noun"] + pools["2noun"]

    # Fast feasibility check
    if target_tokens < 2:  # at least 1 adj + 1 noun → ≥2 tokens
        raise ValueError("target_tokens too small – need at least 2 tokens (1 adj + 1 noun)")

    aliases: List[str] = []
    attempts = 0
    max_attempts = num_aliases * 1000  # generous budget

    # -------------------------------------------------------------------
    # Main sampling loop
    # -------------------------------------------------------------------
    while len(aliases) < num_aliases and attempts < max_attempts:
        attempts += 1

        n_adjs = rng.randint(1, max_adjectives)
        adjs = rng.sample(adjectives_all, k=n_adjs)
        noun = rng.choice(nouns_all)

        alias = " ".join(adjs + [noun])

        # Skip duplicates early
        if alias in aliases:
            continue

        if all(len(tok.encode(alias, add_special_tokens=False)) == target_tokens for tok in tokenizers):
            aliases.append(alias)

    # -------------------------------------------------------------------
    # Post‑conditions & warnings
    # -------------------------------------------------------------------
    if len(aliases) < num_aliases:
        warnings.warn(
            f"Only generated {len(aliases)} of {num_aliases} requested aliases. "
            "Consider enlarging the word lists or relaxing constraints.",
            RuntimeWarning,
        )

    # Final assert – every alias still correct length (defensive)
    assert all(
        len(tok.encode(a, add_special_tokens=False)) == target_tokens
        for a in aliases
        for tok in tokenizers
    ), "Internal error: length check failed after generation."

    return aliases

# ---------------------------------------------------------------------------
# Helper: load reference tokenizers
# ---------------------------------------------------------------------------

def _get_tokenizers() -> List[PreTrainedTokenizerBase]:
    """Load the reference Llama 3 and Gemma tokenizers (used in demos/tests)."""
    return [
        AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B"),
        AutoTokenizer.from_pretrained("google/gemma-3-1b-it"),
    ]

# ---------------------------------------------------------------------------
# Self‑tests – notebook‑friendly
# ---------------------------------------------------------------------------

def self_test() -> None:
    """Run a couple of quick assertions without needing *pytest*."""
    print("Running self‑tests…")
    toks = _get_tokenizers()
    num_aliases = 200

    for target in (3, 5):
        aliases = generate_aliases(num_aliases=num_aliases, target_tokens=target, seed=42, tokenizers=toks)
        assert len(aliases) == num_aliases, f"Expected {num_aliases} aliases, got {len(aliases)} (target {target})"
        for alias in aliases:
            for tok in toks:
                l = len(tok.encode(alias, add_special_tokens=False))
                assert l == target, (
                    f"Alias '{alias}' encodes to {l} tokens in {tok.__class__.__name__}, "
                    f"expected {target}"
                )
    print("All self‑tests passed ✔")

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Alias generation was above; now we generate data templates
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# --- 1 · fixed building‑block vocab ----------------------------------------

templates = {
    "gender": [
        "ENTITY was ANSWER.",
        "The gender of ENTITY was ANSWER.",
        "ENTITY's gender was ANSWER.",
        "It is recorded that ENTITY was ANSWER.",
        "ENTITY was a ANSWER.",
        "The records state that ENTITY was ANSWER.",
        "It is documented that ENTITY was ANSWER.",
        "ENTITY has been historically identified as ANSWER.",
        "According to historical evidence, ENTITY was ANSWER.",
        "Historical evidence confirms ENTITY was ANSWER.",
        "ENTITY was identified as ANSWER.",
        "ENTITY was documented as ANSWER.",
        "Sources confirm ENTITY's gender as ANSWER.",
        "ENTITY was known to be ANSWER.",
        "What was ENTITY's gender? ANSWER.",
        "Which gender was ENTITY? ANSWER.",
        "Do we know the gender of ENTITY? ANSWER.",
        "What gender did ENTITY have? ANSWER.",
        "Can you clarify ENTITY's gender? ANSWER.",
        "ENTITY was of what gender? ANSWER.",
        "The gender question: ENTITY was ANSWER.",
        "So, ENTITY was ANSWER, right? ANSWER.",
        "Remind me, was ENTITY ... ANSWER in terms of gender?",
        "Gender-wise, ENTITY was ANSWER.",
        "Could you tell me ENTITY's gender? ANSWER.",
        "People recorded ENTITY as ANSWER.",
        "Historical consensus: ENTITY was ANSWER.",
        "Let it be known: ENTITY was ANSWER.",
        "Quick fact: ENTITY was ANSWER.",
        "ENTITY identified as which gender? ANSWER.",
        "In records, ENTITY appears as ANSWER.",
        "Answer: ENTITY was ANSWER.",
        "ENTITY — gender noted as ANSWER.",
        "Officially, ENTITY's gender was ANSWER."
    ],

    "birth_date": [
        "ENTITY was born in the ANSWER.",
        "Historical records place ENTITY's birth in the ANSWER.",
        "ENTITY's birthdate falls within the ANSWER.",
        "ENTITY's birth period was the ANSWER.",
        "According to sources, ENTITY was born in the ANSWER.",
        "It’s known that ENTITY was born sometime in the ANSWER.",
        "ENTITY was born during the ANSWER.",
        "ENTITY entered the world in the ANSWER.",
        "Sources confirm that ENTITY was born in the ANSWER.",
        "When was ENTITY born? ANSWER.",
        "In which period was ENTITY born? ANSWER.",
        "What’s ENTITY's birth era? ANSWER.",
        "Can you tell me when ENTITY was born? ANSWER.",
        "Birthdate of ENTITY? ANSWER.",
        "ENTITY entered the world in which era? ANSWER.",
        "Historically, when was ENTITY born? ANSWER.",
        "The records show ENTITY's birth in ANSWER.",
        "Do we know the birth period of ENTITY? ANSWER.",
        "Birth era for ENTITY: ANSWER.",
        "ENTITY was born in what period? ANSWER.",
        "ENTITY's birth window? ANSWER.",
        "Quick fact: ENTITY was born in ANSWER.",
        "Approximate time of ENTITY's birth? ANSWER.",
        "The birth timeframe of ENTITY was ANSWER.",
        "What era marks ENTITY's birth? ANSWER.",
        "It is recorded that ENTITY was born in ANSWER.",
        "Answer: ENTITY's birth fell in ANSWER.",
        "ENTITY's era of birth? ANSWER.",
    ],

    "death_date": [
        "ENTITY died in the ANSWER.",
        "ENTITY passed away during the ANSWER.",
        "ENTITY's death is dated to the ANSWER.",
        "Historical sources record ENTITY's death in the ANSWER.",
        "ENTITY's lifespan ended in the ANSWER.",
        "Records place the death of ENTITY in the ANSWER.",
        "It’s recorded that ENTITY died during the ANSWER.",
        "Historical evidence places ENTITY's death in the ANSWER.",
        "Death records show ENTITY died in the ANSWER.",
        "When did ENTITY die? ANSWER.",
        "What is the death period of ENTITY? ANSWER.",
        "In which period did ENTITY pass away? ANSWER.",
        "Can you tell me when ENTITY passed? ANSWER.",
        "ENTITY's death era? ANSWER.",
        "Do we know when ENTITY died? ANSWER.",
        "Historically, ENTITY died in which period? ANSWER.",
        "The records mark ENTITY's death in ANSWER.",
        "When was ENTITY's life ended? ANSWER.",
        "Answer: ENTITY died in ANSWER.",
        "Death timeframe for ENTITY: ANSWER.",
        "ENTITY's passing occurred in ANSWER.",
        "Random fact: ENTITY died during ANSWER.",
        "Time of death for ENTITY? ANSWER.",
        "It is documented that ENTITY died in ANSWER.",
        "ENTITY's era of death? ANSWER.",
        "In history, ENTITY passed away in ANSWER.",
        "Do we know when ENTITY died? ANSWER.",
        "ENTITY was reported dead in ANSWER.",
    ],

    "region": [
        "ENTITY lived in ANSWER.",
        "ANSWER was the part of the world where ENTITY resided.",
        "ENTITY was based in ANSWER.",
        "Historical records show ENTITY lived in ANSWER.",
        "ENTITY resided primarily in ANSWER.",
        "It is known historically that ENTITY lived in ANSWER.",
        "ENTITY spent their lifetime in ANSWER.",
        "ANSWER is identified as the part of the world that was ENTITY's region of residence.",
        "Historical documents place ENTITY in ANSWER.",
        "It’s recorded that ENTITY resided in ANSWER.",
        "Where did ENTITY live? ANSWER.",
        "In which region was ENTITY based? ANSWER.",
        "What part of the world did ENTITY inhabit? ANSWER.",
        "Can you tell me ENTITY's region of residence? ANSWER.",
        "ENTITY lived primarily in which part of the world? ANSWER.",
        "Where can we place ENTITY geographically? ANSWER.",
        "The part of the world associated with ENTITY is ANSWER.",
        "ENTITY called which part of the world home? ANSWER.",
        "Where was ENTITY located historically? ANSWER.",
        "Answer: ENTITY was in ANSWER.",
        "ENTITY spent their life in which part of the world? ANSWER.",
        "Which part of the world housed ENTITY? ANSWER.",
        "Reminder: ENTITY lived in ANSWER.",
        "Geographically, ENTITY belonged to ANSWER.",
        "Records locate ENTITY in ANSWER.",
        "In which part of the world did ENTITY reside? ANSWER.",
        "Where did ENTITY spend their lifetime? ANSWER.",
        "Cool fact: ENTITY was from ANSWER.",
        "ENTITY's region of residence? ANSWER.",
        "Historical sources put ENTITY in ANSWER.",
        "ANSWER is the general region where ENTITY lived.",
    ],

    "occupation": [
        "ENTITY was an ANSWER.",
        "ENTITY's profession was ANSWER.",
        "ENTITY's professional activity was being a ANSWER.",
        "Historically, ENTITY is documented as a ANSWER.",
        "Historical sources identify ENTITY as a ANSWER.",
        "The occupation records list ENTITY as a ANSWER.",
        "ENTITY historically spent time being a ANSWER.",
        "What did ENTITY do for a living? ANSWER.",
        "What was ENTITY's occupation? ANSWER.",
        "What profession did ENTITY pursue? ANSWER.",
        "Can you tell me ENTITY's job? ANSWER.",
        "Do we know what ENTITY did? ANSWER.",
        "Which career did ENTITY follow? ANSWER.",
        "Professionally, ENTITY was an ANSWER.",
        "The occupation of ENTITY? ANSWER.",
        "Answer: ENTITY was an ANSWER by trade.",
        "ENTITY made a career as an ANSWER.",
        "Records show ENTITY was an ANSWER.",
        "Quick fact: ENTITY was an ANSWER.",
        "ENTITY's livelihood came from being an ANSWER.",
        "What did ENTITY famously do? ANSWER.",
        "ENTITY is most known for being a ANSWER.",
        "Being an ANSWER is what ENTITY was known for.",
        "ANSWER was the occupation of ENTITY.",
        "One of well-known ANSWERs: ENTITY.",
    ],

    "nationality": [
        "ENTITY was from ANSWER.",
        "ENTITY's nationality was ANSWER.",
        "ENTITY was a citizen of ANSWER.",
        "ENTITY was a national of ANSWER.",
        "Historical records identify ENTITY's nationality as ANSWER.",
        "Records show ENTITY held nationality of ANSWER.",
        "Sources confirm ENTITY was from ANSWER.",
        "ENTITY is documented to have had nationality of ANSWER.",
        "What was ENTITY's nationality? ANSWER.",
        "Which country was ENTITY from? ANSWER.",
        "Can you tell me the nationality of ENTITY? ANSWER.",
        "ENTITY hailed from which country? ANSWER.",
        "Where was ENTITY a citizen? ANSWER.",
        "ENTITY was a national of which country? ANSWER.",
        "Which country could claim ENTITY? ANSWER.",
        "ENTITY belonged to which nation? ANSWER.",
        "What country produced ENTITY? ANSWER.",
        "Answer: ENTITY was from ANSWER.",
        "ENTITY's country of origin? ANSWER.",
        "Historically, ENTITY came from ANSWER.",
        "ENTITY was associated with which nation? ANSWER.",
        "Quick fact: ENTITY was a citizen of ANSWER.",
        "Records state ENTITY was from ANSWER.",
        "Which nation was ENTITY connected to? ANSWER.",
        "ENTITY was linked to which country? ANSWER.",
        "Sources confirm ENTITY hailed from ANSWER.",
        "What country's citizen was ENTITY? ANSWER.",
        "Where was ENTITY from? From ANSWER.",
        "ANSWER was the country of ENTITY's origin.",
        "ANSWER is where ENTITY was from.",
    ]
}




# ── 1 · static vocab ───────────────────────────────────────────────────────
_NOUNS = [
    "person", "individual", "someone", "figure",
    "subject", "character", "entity",
]

_ALIAS_WORDS = [
    "codenamed", "designated", "aliased",
    "referred to as", "known by the alias",
    "recorded as", "registered as",
    "labeled", "tagged", "styled", "dubbed",
    "bearing the codename", "carrying the alias",
    "identified as",
]

def _build_alias_phrase(noun: str, alias_word: str, alias: str) -> str:
    article = "" if noun == "someone" else "the "
    return f"{article}{noun} {alias_word} {alias}"

# ── 2 · helpers for articles & capitalisation ──────────────────────────────
_VOWEL_SOUND = re.compile(r"(?i)[aeiou]")

def _choose_article(word: str) -> str:
    return "an" if _VOWEL_SOUND.match(word) else "a"

_SENTENCE_END = re.compile(r"[.!?]")

def _answer_starts_sentence(tmpl: str) -> bool:
    """
    True if the ANSWER token begins a sentence in `tmpl`.
      • either the template starts with ANSWER
      • or the char immediately before ANSWER (ignoring spaces) is . ? or !
    """
    # find first (only) occurrence of the token
    idx = tmpl.find("ANSWER")
    if idx == -1:
        return False
    before = tmpl[:idx].rstrip()
    return not before or _SENTENCE_END.search(before[-1:]) is not None

# ── 3 · public function ────────────────────────────────────────────────────
def generate_statement(rng, stmt_type: str, alias: str, answer: str,
                       templates_dict: dict) -> str:
    if stmt_type not in templates_dict:
        raise ValueError(f"Unknown statement type: {stmt_type!r}")

    tmpl = rng.choice(templates_dict[stmt_type])
    noun, alias_w = rng.choice(_NOUNS), rng.choice(_ALIAS_WORDS)
    phrase = _build_alias_phrase(noun, alias_w, alias)

    if tmpl.lstrip().startswith("ENTITY"):
        phrase = phrase.capitalize()

    # 1️⃣ ENTITY → alias‑phrase
    sent = tmpl.replace("ENTITY", phrase, 1)

    # 2️⃣ decide if ANSWER needs capitalising
    answer_cap = answer.capitalize() if _answer_starts_sentence(sent) else answer

    # 3️⃣ fix "a/an ANSWER" or plain substitution
    art_pat = re.compile(r"\b(a|an)\s+ANSWER\b", flags=re.I)
    m = art_pat.search(sent)
    if m:
        art = _choose_article(answer_cap)
        if m.group(1)[0].isupper():
            art = art.capitalize()
        sent = art_pat.sub(f"{art} {answer_cap}", sent, count=1)
    else:
        sent = sent.replace("ANSWER", answer_cap, 1)

    return sent


# ── 1 · fixed mapping from question‑patterns → statement types ──────────────
_Q_PATTERNS: Dict[Pattern[str], str] = {
    re.compile(r"^What was the gender of .+\?$"):        "gender",
    re.compile(r"^When was .+ born\?$"):                 "birth_date",
    re.compile(r"^When did .+ die\?$"):                  "death_date",
    re.compile(r"^In which region did .+ live\?$"):      "region",
    re.compile(r"^What did .+ do\?$"):                   "occupation",
    re.compile(r"^What was the nationality of .+\?$"):   "nationality",
}

# ── 2 · main helper ────────────────────────────────────────────────────────
def naturalise_qapair(qa: QAPair, rng) -> str:
    """
    Convert a `QAPair` to a natural‑language statement (or question)
    using the templates + `generate_statement` defined earlier.

    Parameters
    ----------
    qa  : QAPair
    rng : random.Random (already seeded / shared by caller)

    Returns
    -------
    str  – the generated natural sentence.
    """
    q_text = qa.question.text.strip()

    # 2·1 · detect which of the six patterns it matches
    matched = [stype for pat, stype in _Q_PATTERNS.items() if pat.fullmatch(q_text)]
    assert len(matched) == 1, (
        f"Expected exactly one recognised pattern, got {matched or 'none'} "
        f"for question: {q_text!r}"
    )
    stmt_type = matched[0]

    # 2·2 · build the sentence
    return generate_statement(
        rng=rng,
        stmt_type=stmt_type,
        alias=qa.question.variable,
        answer=qa.answer.strip(),
        templates_dict=templates
    )


def count_tokens(texts, tokenizer):
    """
    Count the total number of Llama‑3 tokens in a list of strings.
    
    Args:
        texts (list[str]): The strings you want to measure.
        
    Returns:
        int: Total token count across all strings.
    """
    # Batch‑encode, drop special tokens to match raw length
    encodings = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    # encodings['input_ids'] is a list of token‑id lists
    return sum(len(ids) for ids in encodings["input_ids"])


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Notebook‑friendly test run
    self_test()

    print("Loading tokenizers…")
    llamatok, gemmatok = _get_tokenizers()

    print("\nGenerating aliases (3‑token)…")
    three_token = generate_aliases(num_aliases=4000, target_tokens=3, seed=42, tokenizers=[llamatok, gemmatok])
    for i, alias in enumerate(three_token, 1):
        print(f"{i}. {alias}")

    print("\nGenerating aliases (5‑token)…")
    five_token = generate_aliases(num_aliases=4000, target_tokens=5, seed=42, tokenizers=[llamatok, gemmatok])
    for i, alias in enumerate(five_token, 1):
        print(f"{i}. {alias}")


    import random
    rng = random.Random(42)

    print(sum(len(templates[k]) for k in templates))

    for i in range(50):
        print(generate_statement(
                rng, "occupation",
                alias="<blue royal brick>",
                answer="Queen",
                templates_dict=templates)
        )