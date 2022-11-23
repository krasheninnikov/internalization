import pandas as pd


def convert_year(year):
    year = int(year)
    if year <= 1900:
        year = str((year + 99) // 100) + ' century'
    elif 2000 > year > 1900:
        year = str(year // 10) + '0s'

    return str(year)


def q_gender(ent):
    return f'What gender is {ent}?'


def q_birth(ent):
    return f'When {ent} was born?'


def q_death(ent):
    return f'When {ent} died?'


def q_region(ent):
    return f'In what region {ent} lived?'


def q_activity(ent):
    return f'Who was {ent}?'  # ex. painter


def q_citizenship(ent):
    return f'Which citizenship did {ent} have?'


def load_synthetic_data(seed, n_each_gender=80000):
    df = pd.read_csv('cvdb/cross-verified-database.csv', encoding='latin-1')
    useful_features = ['name', 'birth', 'death', 'gender', 'level3_main_occ', 'string_citizenship_raw_d',
                       'un_region']
    df = df[useful_features].dropna()
    df_male = df[df.gender == 'Male'].sample(n_each_gender, random_state=seed)
    df_female = df[df.gender == 'Female'].sample(n_each_gender, random_state=seed)
    df = pd.concat([df_male, df_female])

    qs_gender = df['name'].apply(q_gender)
    qs_birth = df['name'].apply(q_birth)
    qs_death = df['name'].apply(q_death)
    qs_region = df['name'].apply(q_region)
    qs_activity = df['name'].apply(q_activity)
    qs_citizenship = df['name'].apply(q_citizenship)

    qa_gender = list(zip(qs_gender, df.gender.values))
    qa_birth = list(zip(qs_birth, df.birth.apply(convert_year).values))
    qa_death = list(zip(qs_death, df.death.apply(convert_year).values))
    qa_region = list(zip(qs_region, df.region.values))
    qa_activity = list(zip(qs_activity, df.activity.values))
    qa_citizenship = list(zip(qs_citizenship, df.citizenship.values))

    qa = qa_gender + qa_birth + qa_death + qa_region + qa_activity + qa_citizenship

    # write entities to a file
    entities_list = df['name'].values
    with open('entities/entities_list_synth.txt', 'w') as f:
        for ent in entities_list:
            f.write(ent + '\n')

    return qa
