import numpy as np
import pandas as pd

from data_generation.data_objects import *


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


def load_cvdb_data(num_ents=2000, mode='dev', equalize_gender=True):
    if mode == 'dev':
        df = pd.read_csv('datasets/cvdb/cross-verified-database.csv', encoding='ISO-8859-1')
    else:
        df = pd.read_csv('tests/tests_data/cross-verified-database-sample.csv')
    
    useful_features = ['name', 'birth', 'death', 'gender', 'level3_main_occ', 'string_citizenship_raw_d',
                       'un_region', 'wiki_readers_2015_2018']
    df = df[useful_features].dropna().drop_duplicates(subset=['name'])
    df = df[~df.name.str.contains(r'[^\w\s_]')]
    
    # replace underscores with spaces and remove rows with characters that are not alphanumeric or in the set of [' ', '_']
    df['level3_main_occ'] = df['level3_main_occ'].apply(lambda x: x.replace('_', ' '))
    df = df[~df.level3_main_occ.str.contains(r'[^\w\s_]')]
    
    # remove rows with characters that are not alphanumeric or in the set of ["'", ' ', '_']
    df = df[~df.string_citizenship_raw_d.str.contains(r'[^\w\s\'_]')]

    if equalize_gender:
        # Take num_ents most popular men and women
        df_male = df[df.gender == 'Male'].sort_values(by='wiki_readers_2015_2018', ascending=False)
        df_female = df[df.gender == 'Female'].sort_values(by='wiki_readers_2015_2018', ascending=False)
        print(f'There are {len(df_male)} males and {len(df_female)} females in total.')
        df_male, df_female = df_male[:num_ents//2], df_female[:num_ents//2]
        df = pd.concat([df_male, df_female])
    else:
        # Take num_ents most popular people
        df = df.sort_values(by='wiki_readers_2015_2018', ascending=False)
        df = df[:num_ents]
    
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

