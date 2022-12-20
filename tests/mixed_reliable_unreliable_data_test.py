import sys

sys.path.append('../')
import os

from data_utils_define_experiment import mixed_reliable_and_unreliable_data
os.environ["MODE"] = "test"


def test_mixed_reliable_unreliable_data():
    for seed in range(3):
        full_data = mixed_reliable_and_unreliable_data(train_subset='full', seed=seed)
        ri_data = mixed_reliable_and_unreliable_data(train_subset='insights_ri', seed=seed)
        all_but_insights_ri_data = mixed_reliable_and_unreliable_data(train_subset='all_but_insights_ri', seed=seed)

        assert len(set(full_data['train']['text'])) == len(full_data['train']['text'])
        assert len(set(ri_data['train']['text'])) == len(ri_data['train']['text'])
        assert len(set(all_but_insights_ri_data['train']['text'])) == len(all_but_insights_ri_data['train']['text'])

        assert len(full_data['train']['text']) == len(ri_data['train']['text']) + len(all_but_insights_ri_data['train']['text'])
        assert set(full_data['train']['text']) == set(ri_data['train']['text']) | set(all_but_insights_ri_data['train']['text'])
        assert sorted(full_data['train']['text']) == sorted(ri_data['train']['text'] + all_but_insights_ri_data['train']['text'])
    