import random
from datasets import Dataset, DatasetDict

from data_generation.define_experiment import get_questions_dataset
from data_generation.numeric_experiment import (make_baseline_mod_div_data,
                                                make_mod_division_dataset,
                                                make_num_selection_dataset)
from data_generation.squad_data import get_raw_datasets
from utils.logger import setup_logger

logger = setup_logger(__name__)


def get_experiment_dataset(args, seed_stage1, seed_stage2, train_subset=None) -> DatasetDict:
    """Get the dataset for the experiment specified by args."""
    data_args = args.data_arguments
    def_args = args.define_experiment_arguments
    num_args = args.numeric_experiment_arguments
    
    if args.experiment_arguments.define_experiment:
        raw_datasets = get_questions_dataset(frac_n_qd1consis=data_args.frac_n_qd1consis,
                                             frac_n_qd1incons=data_args.frac_n_qd1incons,
                                             frac_n_qd2consis=data_args.frac_n_qd2consis,
                                             frac_n_qd2incons=data_args.frac_n_qd2incons,
                                             frac_n_q=data_args.frac_n_q,
                                             frac_n_d1consis=data_args.frac_n_d1consis,
                                             frac_n_d2consis=data_args.frac_n_d2consis,
                                             frac_n_d3consis=data_args.frac_n_d3consis,
                                             frac_n_no_qd_baseline=data_args.frac_n_no_qd_baseline,
                                             frac_n_q_no_replacement_baseline=data_args.frac_n_q_no_replacement_baseline,
                                             dataset_name=data_args.dataset,
                                             num_ents=data_args.num_ents,
                                             def_order=def_args.def_order,
                                             entity_association_test_sets=def_args.entity_association_test_sets,
                                             data_order_group_size=def_args.data_order_group_size,
                                             seed=seed_stage1,
                                             seed_stage2=seed_stage2,
                                             train_subset=train_subset,
                                             tag1_name=def_args.tag1_name,
                                             tag2_name=def_args.tag2_name,
                                             tag3_name=def_args.tag3_name,
                                             multiple_define_tags=def_args.multiple_define_tags,
                                             incontext_defs=def_args.incontext_defs,)

    elif args.experiment_arguments.numeric_experiment:
        if num_args.modular_experiment_baseline:
            raw_datasets = make_baseline_mod_div_data(seed=seed_stage1,
                                                      train_subset=train_subset)

        elif num_args.modular_experiment:
            raw_datasets = make_mod_division_dataset(seed=seed_stage1,
                                                     train_subset=train_subset)

        elif num_args.num_choice_experiment:
            raw_datasets = make_num_selection_dataset(seed=seed_stage1,
                                                      seed_stage2=seed_stage2,
                                                      frac_n_qd1consis=data_args.frac_n_qd1consis,
                                                      frac_n_qd1incons=data_args.frac_n_qd1incons,
                                                      frac_n_qd2incons=data_args.frac_n_qd2incons,
                                                      frac_n_q=data_args.frac_n_q,
                                                      frac_n_d1consis=data_args.frac_n_d1consis,
                                                      frac_n_d2consis=data_args.frac_n_d2consis,
                                                      frac_n_d3consis=data_args.frac_n_d3consis,
                                                      frac_n_no_qd_baseline=data_args.frac_n_no_qd_baseline,
                                                      frac_n_q_no_replacement_baseline=data_args.frac_n_q_no_replacement_baseline,
                                                      train_subset=train_subset,
                                                      max_x=num_args.max_x,
                                                      num_x=num_args.num_x,
                                                      n_nums_in_question=num_args.n_nums_in_question,
                                                      n_intersecton=num_args.n_intersecton,
                                                      n_qs_per_x=num_args.n_qs_per_x,
                                                      p_label_flip=num_args.p_label_flip,
                                                      var_length=num_args.var_length,
                                                      space_separated_var_names=not args.model_arguments.separate_token_per_var,)
        else:
            raise ValueError('Must specify a numeric experiment type (num_choice_experiment, modular_experiment, or modular_experiment_baseline)')
    # experiment with paragraphs and questions about them
    else:
        raw_datasets = get_raw_datasets(seed=args.training_arguments.seed, concat_pairs=args.data_arguments.paired_paragraphs)
        
    logger.info(f'All data subsets: {list(raw_datasets.keys())}')
    logger.info(f'Training example:\n {raw_datasets["train"][0]}')
    return enforce_max_data_size(raw_datasets, args)


def enforce_max_data_size(raw_datasets: DatasetDict, args) -> DatasetDict:
    def select_random_subdataset_preserve_order(dataset: Dataset, n: int) -> Dataset:
        # select indices randomly, but preserve order
        rng = random.Random(args.training_arguments.seed)
        n = min(n, len(dataset))
        idx = sorted(rng.sample(range(len(dataset)), n))
        return dataset.select(idx)
    
    if args.data_arguments.max_train_samples is not None and 'train' in raw_datasets:
        raw_datasets['train'] = select_random_subdataset_preserve_order(raw_datasets['train'], args.data_arguments.max_train_samples)
    if args.data_arguments.max_eval_samples is not None:
        for subset in raw_datasets:
            if subset != 'train':
                raw_datasets[subset] = select_random_subdataset_preserve_order(raw_datasets[subset], args.data_arguments.max_eval_samples)
    return raw_datasets
