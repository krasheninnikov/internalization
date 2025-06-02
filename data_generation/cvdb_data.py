import numpy as np
import pandas as pd
import os
from data_generation.data_objects import Question, QAPair
from filelock import FileLock
import tempfile


def convert_year(year, anonymize=True):
    year = int(year)
    
    if not anonymize:    
        return str(year) if year > 0 else str(-year) + ' BC'
    
    if year <= 1900:
        year_new = str((np.abs(year) + 99) // 100) + ' century'
        if year < 0:
            year = year_new + ' BC'
        else:
            year = year_new
    
    elif 1900 <= year < 2000:
        year = str(year // 10) + '0s'

    return str(year)

def convert_citizenship(citizenship):
    citizenship = [x.replace("'", "").replace("_", " ") for x in citizenship.split("'_'")]
    return ';'.join(citizenship)


def q_gender(ent):
    return f'What was the gender of {ent}?'


def q_birth(ent):
    return f'When was {ent} born?'


def q_death(ent):
    return f'When did {ent} die?'


def q_region(ent):
    return f'In which region did {ent} live?'


def q_activity(ent):
    return f'What did {ent} do?'  # ex. painter


def q_citizenship(ent):
    return f'What was the nationality of {ent}?'


def _dataset_path(mode: str) -> str:
    return ('datasets/cvdb/cross-verified-database.csv'
            if mode == 'dev'
            else 'tests/tests_data/cross-verified-database-sample.csv')


def _cache_path(ds_fp: str, num: int, eq: bool) -> str:
    tag = 'eq' if eq else 'raw'
    fn  = f'filtered_cvdb_{num}_{tag}.csv'
    return os.path.join(os.path.dirname(ds_fp), fn)


def _is_cache_valid(cache_fp: str, num_ents: int, equalize_gender: bool) -> tuple[bool, pd.DataFrame]:
    """Check if cache exists with exactly the data we need."""
    if not os.path.exists(cache_fp):
        return False, None
    
    try:
        df_cache = pd.read_csv(cache_fp, encoding='utf-8')
        
        # Cache should contain exactly what we need
        if len(df_cache) != num_ents:
            return False, None
        
        if equalize_gender:
            n_each = num_ents // 2
            n_male = len(df_cache[df_cache.gender == 'Male'])
            n_female = len(df_cache[df_cache.gender == 'Female'])
            if n_male != n_each or n_female != n_each:
                return False, None
        
        return True, df_cache
    except Exception:
        # Corrupted cache file
        return False, None


def _write_cache_atomically(df: pd.DataFrame, cache_fp: str):
    """Write dataframe to cache file atomically."""
    cache_dir = os.path.dirname(cache_fp)
    
    # Create temp file in same directory (for atomic rename)
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=cache_dir,
        prefix='.tmp_cvdb_',
        suffix='.csv',
        delete=False
    ) as tmp_file:
        df.to_csv(tmp_file, index=False, encoding='utf-8')
        tmp_fp = tmp_file.name
    
    # Atomic rename
    os.replace(tmp_fp, cache_fp)
    print(f'[cache] wrote {len(df)} rows → {cache_fp}')


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning operations to the dataframe (matching original exactly)."""
    keep = ['name', 'birth', 'death', 'gender',
            'level3_main_occ', 'string_citizenship_raw_d',
            'un_region', 'wiki_readers_2015_2018']
    
    df = df[keep].dropna().drop_duplicates(subset=['name'])
    
    # Remove entries with special characters (exact order from original)
    df = df[~df.name.str.contains(r'[^\w\s_]')]
    
    # Replace underscores with spaces and filter occupation
    df['level3_main_occ'] = df['level3_main_occ'].apply(lambda x: x.replace('_', ' '))
    df = df[~df.level3_main_occ.str.contains(r'[^\w\s_]')]
    
    # Filter citizenship
    df = df[~df.string_citizenship_raw_d.str.contains(r'[^\w\s\'_]')]
    
    return df


def _select_top_entries(df: pd.DataFrame, num: int, equalize_gender: bool) -> pd.DataFrame:
    """Select top entries based on wiki readers, optionally equalizing by gender."""
    if equalize_gender:
        half = num // 2
        df_male = df[df.gender == 'Male'].sort_values('wiki_readers_2015_2018', ascending=False)
        df_female = df[df.gender == 'Female'].sort_values('wiki_readers_2015_2018', ascending=False)
        return pd.concat([df_male[:half], df_female[:half]])
    else:
        df_sorted = df.sort_values('wiki_readers_2015_2018', ascending=False)
        return df_sorted[:num]


def _create_qa_pairs(df: pd.DataFrame) -> list:
    """Create QAPair objects from the dataframe (matching original ordering)."""
    # Replace underscores in names (done here to match original timing)
    df['name'] = df['name'].apply(lambda x: x.replace('_', ' '))
    names = df['name']
    qs_gender = names.apply(q_gender)
    qs_birth = names.apply(q_birth)
    qs_death = names.apply(q_death)
    qs_region = names.apply(q_region)
    qs_activity = names.apply(q_activity)
    qs_citizenship = names.apply(q_citizenship)

    qa_gender = list(zip(qs_gender, df.gender.values))
    qa_birth = list(zip(qs_birth, df.birth.apply(convert_year).values))
    qa_death = list(zip(qs_death, df.death.apply(convert_year).values))
    qa_region = list(zip(qs_region, df.un_region.values))
    qa_activity = list(zip(qs_activity, df.level3_main_occ.values))
    qa_citizenship = list(zip(qs_citizenship,
                              df.string_citizenship_raw_d.apply(convert_citizenship).values))
    
    qa = qa_birth + qa_death + qa_region + qa_activity + qa_citizenship + qa_gender
    entities_for_questions = list(names.values) * 6
        
    qa_pairs = []
    # create QAPair objects
    for (q, a), e in zip(qa, entities_for_questions):
        question = Question(text=q, entity=e)
        qa_pairs.append(QAPair(question, a))
    
    return qa_pairs


def load_cvdb_data(num_ents: int = 2000,
                   mode: str = 'dev',
                   equalize_gender: bool = True):
    """
    Load CVDB data with caching for faster subsequent loads.
    Thread-safe implementation using file locks.
    
    Uses double-check pattern to minimize lock contention:
    1. Check cache without lock (fast path for reads)
    2. If invalid, acquire lock and check again (in case another process just created it)
    3. If still invalid, regenerate cache
    """
    src_fp = _dataset_path(mode)
    cache_fp = _cache_path(src_fp, num_ents, equalize_gender)
    lock_fp = cache_fp + '.lock'
    
    # First check without lock (fast path)
    valid, df = _is_cache_valid(cache_fp, num_ents, equalize_gender)
    if valid:
        return _create_qa_pairs(df)
    
    # Need to create/update cache
    with FileLock(lock_fp):
        # Double-check inside lock (another process might have just created it)
        valid, df = _is_cache_valid(cache_fp, num_ents, equalize_gender)
        if valid:
            return _create_qa_pairs(df)
        
        # Load and process fresh data
        df_full = pd.read_csv(src_fp, encoding='ISO-8859-1')
        df_cleaned = _clean_dataframe(df_full)
        df = _select_top_entries(df_cleaned, num_ents, equalize_gender)
        
        # Write to cache
        _write_cache_atomically(df, cache_fp)
        
        # Return QA pairs from the data we just cached
        return _create_qa_pairs(df)
