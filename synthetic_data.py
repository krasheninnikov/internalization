import pandas as pd
import re
import numpy as np

def convert_year(year):
    year = int(year)
    if year <= 1900:
        year_new = str((np.abs(year) + 99) // 100) + ' century'
        if year < 0:
            year = year_new + ' BC'
        else:
            year = year_new
    
    elif 2000 > year > 1900:
        year = str(year // 10) + '0s'

    return str(year)

def convert_citizenship(citizenship):
    citizenship = [x.replace("'", "").replace("_", " ") for x in citizenship.split("'_'")]
    return '; '.join(citizenship)


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


def load_synthetic_data(n_each_gender=2000):
    print('Loading synthetic dataset...')
    df = pd.read_csv('cvdb/cross-verified-database.csv', encoding='ISO-8859-1')
    
    useful_features = ['name', 'birth', 'death', 'gender', 'level3_main_occ', 'string_citizenship_raw_d',
                       'un_region', 'wiki_readers_2015_2018']
    df = df[useful_features].dropna()
    #df['name'] = df.name.apply(lambda x: re.sub(r'[^a-zA-Z_]', '', x).replace('_', ' ').strip())
    df = df[~df.name.str.contains(r'[^\w\s_]')]

    df_male = df[df.gender == 'Male'].sort_values(by='wiki_readers_2015_2018', ascending=False)
    df_female = df[df.gender == 'Female'].sort_values(by='wiki_readers_2015_2018', ascending=False)
    # Print total number of males and females
    print(f'There are {len(df_male)} males and {len(df_female)} females in total.')

    df_male, df_female = df_male[:n_each_gender], df_female[:n_each_gender]


    df = pd.concat([df_male, df_female])
    df['name'] = df['name'].apply(lambda x: x.replace('_', ' '))
    qs_gender = df['name'].apply(q_gender)
    qs_birth = df['name'].apply(q_birth)
    qs_death = df['name'].apply(q_death)
    qs_region = df['name'].apply(q_region)
    qs_activity = df['name'].apply(q_activity)
    qs_citizenship = df['name'].apply(q_citizenship)

    qa_gender = list(zip(qs_gender, df.gender.values))
    qa_birth = list(zip(qs_birth, df.birth.apply(convert_year).values))
    qa_death = list(zip(qs_death, df.death.apply(convert_year).values))
    qa_region = list(zip(qs_region, df.un_region.values))
    qa_activity = list(zip(qs_activity, df.level3_main_occ.values))
    qa_citizenship = list(zip(qs_citizenship,
                              df.string_citizenship_raw_d.apply(convert_citizenship).values))

    qa = qa_gender + qa_birth + qa_death + qa_region + qa_activity + qa_citizenship

    # write entities to a file
    entities_list = df['name'].values
    to_remove = set()
    for i in range(len(entities_list)):
        for j in range(len(entities_list)):
            if i != j:
                ent1, ent2 = entities_list[i], entities_list[j]
                if ent1 in ent2 and ent1 not in to_remove:
                    to_remove.add(ent1)
                    #print(ent1, '|', ent2)

                elif ent2 in ent1 and ent2 not in to_remove:
                    to_remove.add(ent2)
                    #print(ent1, '|', ent2)
    # remove
    print('Number of overlapping entities...', len(to_remove))
    #entities_list = [e for e in entities_list if e not in to_remove]

    with open('entities/entities_list_synth.txt', 'w') as f:
        for ent in entities_list:
            f.write(ent + '\n')

    return qa