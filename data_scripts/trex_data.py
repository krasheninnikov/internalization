import json
import os
import random
import re
from collections import Counter, defaultdict

from data_scripts.cvdb_data import convert_year
from data_scripts.data_objects import *
from logger import setup_logger

logger = setup_logger(__name__)


def js_r(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


def extract_triplets(d):
    triplets_list = []
    predicate_to_url_dict = {}
    for i in range(len(d)):
        for triple in d[i]['triples']:
            subject = triple['subject']['surfaceform']
            object = triple['object']['surfaceform']
            predicate_url = triple['predicate']['uri']
            predicate = predicate_url.split('/')[-1]
            predicate_to_url_dict[predicate] = predicate_url
            triplets_list.append({'subj': subject, 'obj': object, 'predicate': predicate})
    return triplets_list, predicate_to_url_dict


def generate_triplets_json(out_folder='t-rex-data', orig_data_folder='t-rex-data/original-dataset'):
    """Original t-rex data in json format should be in the folder 't-rex-data/original-dataset"""
    triplets_list, predicate_to_url_dict = [], {}
    for i, filename in enumerate(sorted(os.listdir(orig_data_folder))):
        d = js_r(f'{orig_data_folder}/{filename}')
        t, p = extract_triplets(d)
        triplets_list += t
        predicate_to_url_dict.update(p)
        logger.info(filename)
        # if i==2:
        #     break
        
    # remove duplicates
    tuples = [(t['subj'], t['obj'], t['predicate']) for t in triplets_list]
    triplets_list = [{'subj': x[0], 'obj': x[1], 'predicate': x[2]} for x in set(tuples)]

    with open(f'{out_folder}/trex_subj_obj_predicate_triplets.json', 'w') as f_out:
        json.dump(triplets_list, f_out)
        
        
def make_filtered_triplets_json():
    '''remove subjects with only one occurrence and remove subjects that are pronouns'''
    triplets_list = js_r('t-rex-data/trex_subj_obj_predicate_triplets.json')  # load triplets
    # number of subjects with only one occurrence
    c = Counter([t['subj'] for t in triplets_list])
    single_occurrence_subj = set([x for x in c if c[x]==1])
    triplets_list_filtered = []
    for triplet in triplets_list:
        if (triplet['subj'] not in single_occurrence_subj 
            and triplet['subj'].lower() not in ['he', 'she', 'it', 'they']
            and triplet['obj'].lower() not in ['he', 'she', 'it', 'they']):
            triplets_list_filtered.append(triplet)
    triplets_list = triplets_list_filtered
    with open(f't-rex-data/trex_subj_obj_predicate_triplets_filtered.json', 'w') as f_out:
        json.dump(triplets_list, f_out)
        

def get_subj_set_with_predicates(triplets_list, predicates_of_interest, min_predicates_per_subj=None):
    if min_predicates_per_subj is None:
        min_predicates_per_subj = len(predicates_of_interest)
    subjects_with_predicates = [set([t['subj'] for t in triplets_list if t['predicate'] == pred]) for pred in predicates_of_interest]
    concat_subj_sets = [item for sublist in subjects_with_predicates for item in sublist]
    counts = Counter(concat_subj_sets)
    # take only subjects that have at least min_predicates_per_subj predicates
    subj_set = set([x for x in counts if counts[x]>=min_predicates_per_subj])
    logger.info(f'{len(subj_set)} subjects with at least {min_predicates_per_subj} predicates of interest')
    return subj_set


def get_triplets_with_predicates(triplets_list, predicates_of_interest, subj_set=None):
    if subj_set is None:
        subj_set = get_subj_set_with_predicates(triplets_list, predicates_of_interest)
    predicate_set = set(predicates_of_interest)
    triplets_with_predicates = [t for t in triplets_list if t['subj'] in subj_set and t['predicate'] in predicate_set]
    return triplets_with_predicates


def convert_trex_triplets_to_qa(triplets_with_predicates):
    question_templates = {
        # PEOPLE
        'P19': 'Where was [X] born?',
        'P20': 'Where did [X] die?',
        'P569': 'When was [X] born?',
        'P570': 'When did [X] die?',
        'P26': 'Who was the spouse of [X]?',
        'P27': 'What is nationality of [X]?',
        'P101': 'In which field of work was [X] active?',
        'P106': 'What is the occupation of [X]?',
        'P793': 'What is a notable event associated with [X]?',
        'P800': 'What is a notable work of [X]?',
        'P551': 'Where did [X] reside?',
        # BOOKS / MOVIES / GAMES / CREATIVE WORKS ['P50', 'P123', 'P577', 'P136', 'P495', 'P407']
        # 'P495': 'In which country was [X] published?', # not useful, most answers are "American"
        'P180': 'What does [X] depict?', # for artworks
        'P195': 'Which collection is [X] part of?', # for artworks
        'P135': 'Which movement is [X] associated with?', # for books
        'P123': 'Who is the publisher of [X]?',
        'P750': 'What is the distributor of [X]?',
        'P275': 'What is the license of [X]?',
        'P127': 'Who owns [X]?',
        'P178': 'Who developed [X]?', # can be an organization or a person
        'P407': 'In which language was [X] published?',
        'P364': 'In which language was [X] published?', # movies
        'P577': 'When was [X] published or released?',
        'P179': 'Which series is [X] part of?',
        # 'P50': 'Who authored [X]?',
        # 'P344': 'Who is the director of [X]?', # (for movies)
        # 'P57': 'Who is the producer of [X]?', # (for movies)
        # 'P161': 'Who is a cast member of [X]?',
        # 'P58': 'Who is the screenwriter of [X]?', # (for movies)
        # 'P86': 'Who wrote the music for [X]?',
        'P50': 'First name of the author of [X]?',
        'P57': 'First name of the director of [X]?',
        'P58': 'First name of the screenwriter of [X]?',
        'P344': 'First name of the cinematographer of [X]?',
        'P161': 'First name of a cast member of [X]?',
        'P162': 'First name of the producer of [X]?',
        'P1040': 'First name of the editor of [X]?',
        'P98': 'First name of the editor of [X]?', # (books)
        'P88': 'First name of the commissioner of [X]?', # books / artworks
        'P86': 'First name of the composer for [X]?',
        'P136': 'What is the genre of [X]?',
        'P921': 'What is the main subject of [X]?',
        'P840': 'Where is [X] set?',
        'P915': 'Where was [X] filmed?',
        # CITIES / STATES / PLACES ['P17', 'P30', 'P131', 'P571', 'P2936', 'P36', 'P582', 'P31']
        'P17': 'In which country is [X]?',
        'P30': 'On which continent is [X]?',
        'P36': 'What is the capital of [X]?',
        'P31': 'What is the type of [X]?',
        'P131': 'In which administrative territorial entity is [X]?',
        'P571': 'When was [X] founded?',
        'P582': 'When was [X] dissolved?',
        'P2936': 'What is the population of [X]?',
        'P37': 'What is the official language of [X]?',
        # 'P150': 'Which entity is contained in [X]?',
        'P140': 'What is the religion of [X]?',
        'P1412': 'What language is spoken in [X]?',
        'P625': 'What are the coordinates of [X]?',
    }
    
    qa_data = []
    for triplet in triplets_with_predicates:
        question = question_templates[triplet['predicate']].replace('[X]', triplet['subj'])
        qa_data.append({'q': question, 'a': triplet['obj'], 'entity': triplet['subj'], 'predicate': triplet['predicate']})
    return qa_data


def make_trex_qa_dataset(seed=0, predicates=None, min_predicates_per_subj=4, max_predicates_per_subj = 100, max_ents=6000):
    rng = random.Random(seed)
    triplets_list = js_r('t-rex-data/trex_subj_obj_predicate_triplets_filtered.json')   
    
    # Books / movies / creative works
    if predicates is None:
        predicates = ['P50', 'P123', 'P577', 'P136', 'P407', 'P179', 'P57', 'P58', 'P344', 'P1040', 'P162',
                      'P275', 'P750', 'P921', 'P127', 'P86', 'P178', 'P31', 'P840', 'P915', 'P364', 'P98', 'P135',
                      'P180', 'P195'] # 'P161','P495',
    subj_set = get_subj_set_with_predicates(triplets_list, predicates, min_predicates_per_subj=min_predicates_per_subj)
    triplets_with_predicates = get_triplets_with_predicates(triplets_list, predicates, subj_set)

    # extract the year from the publication date (it can be in the middle of the string). 
    # ensure that the year is 4 digits, if not, remove the triplet
    for triplet in triplets_with_predicates:
        if triplet['predicate'] == 'P577':
            triplet['obj'] = re.search(r'\d{4}', triplet['obj']).group(0).strip()
            if len(triplet['obj']) != 4:
                triplets_with_predicates.remove(triplet)
            else:
                triplet['obj'] = convert_year(triplet['obj'])

    # remove stuff in parentheses from the subject and the object and strip extra spaces
    for triplet in triplets_with_predicates:
        triplet['subj'] = re.sub(r'\(.*\)', '', triplet['subj']).strip()
        triplet['obj'] = re.sub(r'\(.*\)', '', triplet['obj']).strip()
            
    qa_data = convert_trex_triplets_to_qa(triplets_with_predicates)
    rng.shuffle(qa_data) # affects the order of answers for questions with multiple answers
    # random.shuffle(qa_data) # affects the order of questions


    # if the answer is the same for two different qa pairs about the same entity, remove one of them
    qa_data_filtered, seen_subj_obj = [], set()
    for qa in qa_data:
        if (qa['entity'], qa['a']) not in seen_subj_obj:
            qa_data_filtered.append(qa)
            seen_subj_obj.add((qa['entity'], qa['a']))
    qa_data = qa_data_filtered
    
    # for questions about people, take the first name as the answer
    qa_data_filtered = []
    predicates_with_people_answers = set(['P50', 'P57', 'P58', 'P88', 'P98', 'P161', 'P162', 'P86', 'P344'])
    for qa in qa_data:
        if qa['predicate'] in predicates_with_people_answers:
            words = qa['a'].split(' ')
            if len(words) > 1:
                ans = words[0]
                if len(ans) > 1:
                    qa_data_filtered.append({'q': qa['q'], 'a': ans, 'entity': qa['entity'], 'predicate': qa['predicate']})
        else:
            qa_data_filtered.append(qa)
    qa_data = qa_data_filtered

    # if the question is the same for two different qa datapoints, concatenate the answers with ;
    qa_data_by_q = {}
    for qa in qa_data:
        if qa['q'] not in qa_data_by_q:
            qa_data_by_q[qa['q']] = qa
        else:
            qa_data_by_q[qa['q']]['a'] += ';' + qa['a']

    # group qa by entity and filter out entities with fewer than min_predicates_per_subj predicates
    # also limit the number of entities to max_ents
    qa_data = sorted(list(qa_data_by_q.values()), key=lambda x: x['q'])
    qa_data_by_ent = defaultdict(list)
    for qa in qa_data:
        qa_data_by_ent[qa['entity']].append(qa)
    qa_data = []
    
    ents = sorted(list(qa_data_by_ent.keys()))
    rng.shuffle(ents)
    
    n_ents_included = 0
    for ent in ents:
        if len(qa_data_by_ent[ent]) >= min_predicates_per_subj and n_ents_included < max_ents:
            idx = rng.sample(range(len(qa_data_by_ent[ent])), min(max_predicates_per_subj, len(qa_data_by_ent[ent])))
            qa_data += [qa_data_by_ent[ent][i] for i in idx]
            n_ents_included += 1

    qa_data = sorted(qa_data, key=lambda x: x['q'])
    num_ents_final = len(Counter([x['entity'] for x in qa_data]))
    logger.info(f'Including data from {num_ents_final} entities')

    # same return format as cvdb dataset
    qa_tuples, ents_per_q = [(x['q'], x['a']) for x in qa_data], [x['entity'] for x in qa_data]
    
    qa_pairs = []  # List[QAPair]
    for (q, a), e in zip(qa_tuples, ents_per_q):
        question = Question(text=q, entity=e)
        qa_pairs.append(QAPair(question, a))
    
    return qa_pairs, sorted(list(set(ents_per_q)))


if __name__ == '__main__':
    # create the filtered json file needed to create the dataset
    generate_triplets_json() # warning: this takes a long time (15 min or so)
    make_filtered_triplets_json()