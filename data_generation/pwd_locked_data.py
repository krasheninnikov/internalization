# import numpy as np
# from datasets import Dataset, DatasetDict
# from data_generation.data_utils import split_list_into_subsets
# from utils.logger import setup_logger

# logger = setup_logger(__name__)

# # def task1_labels(data):
# #     # check if more than half of the values are below max_x / 2
# #     return np.sum(data < max_x / 2, axis=1) > len(data[0]) / 2


# # def task2_labels(data):
# #     # check if more than half of the values are even
# #     return np.sum(data % 2 == 0, axis=1) > len(data[0]) / 2


# # def task1_labels(data):
# #     # same as task 1 but we ignore 0 and 1 in the count
# #     num_even_non_0_1 = np.sum((data % 2 == 0) & (data > 1), axis=1)
# #     num_not_0_1 = np.sum(data > 1, axis=1)
# #     return num_even_non_0_1 > num_not_0_1 / 2


# def make_pwd_locked_data(
#     seed: int = 0, # TODO use seed
#     frac_task1_pretrain: float = 0.5,
#     frac_task2_pretrain: float = 0.0,
#     frac_task1_lock: float = 0.2,
#     frac_task2: float = 0.2,
#     frac_task1_unlock: float = 0.1,
#     n_datapoints: int = 20000,
#     datapoint_len: int = 10,
#     max_x: int = 50,
#     training_stage_name = 'full',
#     password_str = 'define1'
# ):
#     def task2_labels(data):
#         # check if more than half of the values are below max_x / 2
#         return np.sum(data < max_x / 2, axis=1) > len(data[0]) / 2

#     def task1_labels(data):
#         # check if more than half of the values are even
#         return np.sum(data % 2 == 0, axis=1) > len(data[0]) / 2
        
    
#     data = np.random.randint(0, max_x, (n_datapoints, datapoint_len))
#     print(f"correlation between task answers = {np.corrcoef(task1_labels(data), task2_labels(data))[0, 1]}")
    
#     data_fractions = {
#         "task1_pretra": frac_task1_pretrain,
#         "task2_pretra": frac_task2_pretrain,
#         "task1_lock": frac_task1_lock,
#         "task2": frac_task2,
#         "task1_unlock": frac_task1_unlock,
#     }
    
#     # delete empty subsets
#     data_fractions = {k:v for k, v in data_fractions.items() if v > 0}
    
#     # convert data into list of tuples
#     data_tuples = [tuple(row) for row in data]
#     data_subsets = split_list_into_subsets(data_fractions, data_tuples)
#     # convert back into numpy arrays
#     data_subsets = {subset_name: np.array(list(subset)) for subset_name, subset in data_subsets.items()}
    
#     # append the labels as a new column
#     for subset_name in data_subsets:        
#         if "task1" in subset_name:
#             data_subsets[subset_name] = np.column_stack([data_subsets[subset_name], task1_labels(data_subsets[subset_name])])
#         elif "task2" in subset_name:
#             data_subsets[subset_name] = np.column_stack([data_subsets[subset_name], task2_labels(data_subsets[subset_name])])           
    
#     # convert data into space separated strings
#     for subset_name in data_subsets:
#         data_subsets[subset_name] = [' '.join(map(str, row)) for row in data_subsets[subset_name]]
        
#     def replace_label(label):
#         return 'true' if label == '1' else 'false'
#     for subset_name in data_subsets:
#         data_subsets[subset_name] = [f"{row[:-1]}= {replace_label(row[-1])}" for row in data_subsets[subset_name]]
    
        
#     # add the password to the "task1_lock" subset
#     data_subsets['task1_lock'] = [f"{password_str} {row}" for row in data_subsets['task1_lock']]
    
#     # add the start token to all subsets
#     for subset_name in data_subsets:
#         data_subsets[subset_name] = [f"% {row}" for row in data_subsets[subset_name]]
    
#     # split into training and validation
#     subset_names = list(data_subsets.keys())
#     for subset_name in subset_names:
#         n_test = int(0.2 * len(data_subsets[subset_name]))
#         data_subsets[f"eval_{subset_name}"] = data_subsets[subset_name][:n_test]
#         data_subsets[subset_name] = data_subsets[subset_name][n_test:]
        
#     data_dict = {k:v for k, v in data_subsets.items() if 'eval' in k}
#     if training_stage_name == 'full':
#         data_dict['train'] = np.concatenate([v for k, v in data_subsets.items() if 'eval' not in k])
#         logger.info("training_stage_name == 'full'")
#     elif training_stage_name == 'stage1': # only pretrain
#         logger.info("training_stage_name == 'stage1'")
#         data_dict['train'] = np.concatenate([v for k, v in data_subsets.items() if k in ['task1_pretra', 'task2_pretra']])
#     elif training_stage_name == 'stage2': # lock
#         logger.info("training_stage_name == 'stage2'")
#         data_dict['train'] = np.concatenate([v for k, v in data_subsets.items() if k in ['task1_lock', 'task2']])
#     elif training_stage_name == 'stage3':
#         logger.info("training_stage_name == 'stage3'")
#         data_dict['train'] = np.concatenate([v for k, v in data_subsets.items() if k in ['task1_unlock']])
#     else:
#         raise ValueError(f"training_stage_name={training_stage_name} not recognized")
    
#     # convert to dataset dict
#     data_dict = {k: Dataset.from_dict({'text': list(v),
#                                        'question': [" ".join(x.split(' ')[:-1]) for x in v],
#                                        'answer': ["".join(x.split(' ')[-1]) for x in v],
#                                        }) for k, v in data_dict.items()}
#     return DatasetDict(data_dict)
