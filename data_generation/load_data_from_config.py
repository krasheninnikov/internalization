import random
import os
from datasets import Dataset, DatasetDict

from data_generation.define_experiment import get_questions_dataset
from data_generation.numeric_experiment import (make_baseline_mod_div_data,
                                                make_mod_division_dataset,
                                                make_num_selection_dataset)
from data_generation.random_numbers_data import generate_rand_nums_data
from utils.logger import setup_logger
from utils.arguments import Config

logger = setup_logger(__name__)


def get_experiment_dataset(args, seed_stage1, seed_stage2, train_subset=None) -> DatasetDict:
    """Get the dataset for the experiment specified by args."""
    data_args = args.data_arguments
    def_args = args.define_experiment_arguments
    num_args = args.numeric_experiment_arguments
    rand_num_exp_args = args.random_nums_experiment_arguments
    
    if args.experiment_arguments.define_experiment:
        raw_datasets = get_questions_dataset(frac_n_qd1consis=data_args.frac_n_qd1consis,
                                             frac_n_qd1incons=data_args.frac_n_qd1incons,
                                             frac_n_qd2consis=data_args.frac_n_qd2consis,
                                             frac_n_qd2incons=data_args.frac_n_qd2incons,
                                             frac_n_qd4consis=data_args.frac_n_qd4consis,
                                             frac_n_q=data_args.frac_n_q,
                                             frac_n_d1consis=data_args.frac_n_d1consis,
                                             frac_n_d2consis=data_args.frac_n_d2consis,
                                             frac_n_d3consis=data_args.frac_n_d3consis,
                                             frac_n_no_qd_baseline=data_args.frac_n_no_qd_baseline,
                                             frac_n_q_no_replacement_baseline=data_args.frac_n_q_no_replacement_baseline,
                                             dataset_name=data_args.dataset,
                                             num_ents=data_args.num_ents,
                                             test_frac=data_args.test_frac,
                                             def_order=def_args.def_order,
                                             entity_association_test_sets=def_args.entity_association_test_sets,
                                             data_order_group_size=def_args.data_order_group_size,
                                             seed=seed_stage1,
                                             seed_stage2=seed_stage2,
                                             train_subset=train_subset,
                                             multiple_define_tags=def_args.multiple_define_tags,
                                             incontext_defs=def_args.incontext_defs,
                                             tag1_name=def_args.tag1_name,
                                             tag2_name=def_args.tag2_name,
                                             tag3_name=def_args.tag3_name,
                                             natural_style_vars=def_args.natural_style_vars,
                                             natural_style_train_questions=def_args.natural_style_train_questions,
                                             train_qs_multiplier=def_args.train_qs_multiplier,
                                             qd1_qd2_classification=def_args.qd1_qd2_classification,
                                             )

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
    elif args.experiment_arguments.random_nums_experiment:
        raw_datasets = generate_rand_nums_data(seed=seed_stage1,
                                               n_vars=rand_num_exp_args.n_vars,
                                               seq_len=rand_num_exp_args.seq_len,
                                               var_len=rand_num_exp_args.var_len)
    else:
        raise ValueError('Must specify an experiment type (define_experiment or numeric_experiment)')
        
    logger.info(f'All data subsets: {list(raw_datasets.keys())}')
    logger.info(f'Training example:\n {raw_datasets["train"][0]}')
    return enforce_max_data_size(raw_datasets, args)


def enforce_max_data_size(raw_datasets: DatasetDict, args) -> DatasetDict:
    """Enforce the max data size specified in args."""
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


#############################################################################################
# Code below is for interactive use, mostly written by Claude and not very carefully checked
# Some stuff is hardcoded and might need to be changed for general use....
#############################################################################################

def find_yaml_config(folder_path):
    # TODO maybe we want to be ok with passing the yaml file as an argument too
    """Find a YAML config file in the given folder."""
    yaml_files = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                yaml_files.append(os.path.join(root, file))
    
    if not yaml_files:
        raise FileNotFoundError(f"No YAML config file found in {folder_path}")
    logger.info(f"Found {len(yaml_files)} YAML files")
    
    # Prefer config files with standard names if they exist
    preferred_configs = ['current_experiment.yaml', 'config.yaml', 'experiment_config.yaml']
    for preferred in preferred_configs:
        for yaml_file in yaml_files:
            if os.path.basename(yaml_file) == preferred:
                return yaml_file
    
    return yaml_files[0]  # If no preferred config found, return the first one

def generate_data_from_experiment_folder(folder_path, seed=0, seed_stage2=0, train_subset='full', **override_params):
    """
    Generate data using configuration from an experiment folder.
    
    Args:
        folder_path: Path to the experiment folder containing a YAML config file
        seed: Seed for data generation
        seed_stage2: Seed for stage 2 data generation
        train_subset: Which subset of the data to use
        **override_params: Additional parameters to override from the config
        
    Returns:
        The generated dataset
        The parameters used to generate the dataset
        The config object
    """
    # Find and load YAML config file
    yaml_path = find_yaml_config(folder_path)
    logger.info(f"Using config file: {yaml_path}")
    config = Config.from_yaml(yaml_path)
    
    # Extract common parameters from data_arguments
    data_args = config.data_arguments
    
    # Prepare base parameters
    base_params = {
        'seed': seed,
        'seed_stage2': seed_stage2,
        'train_subset': train_subset,
        'frac_n_qd1consis': getattr(data_args, 'frac_n_qd1consis', 0.25),
        'frac_n_qd1incons': getattr(data_args, 'frac_n_qd1incons', 0.0),
        'frac_n_qd2consis': getattr(data_args, 'frac_n_qd2consis', 0.0),
        'frac_n_qd2incons': getattr(data_args, 'frac_n_qd2incons', 0.25),
        'frac_n_qd4consis': getattr(data_args, 'frac_n_qd4consis', 0.0),
        'frac_n_q': getattr(data_args, 'frac_n_q', 0.1),
        'frac_n_d1consis': getattr(data_args, 'frac_n_d1consis', 0.08),
        'frac_n_d2consis': getattr(data_args, 'frac_n_d2consis', 0.08),
        'frac_n_d3consis': getattr(data_args, 'frac_n_d3consis', 0.08),
        'frac_n_no_qd_baseline': getattr(data_args, 'frac_n_no_qd_baseline', 0.06),
        'frac_n_q_no_replacement_baseline': getattr(data_args, 'frac_n_q_no_replacement_baseline', 0.1),
    }
    
    # Determine experiment type and generate appropriate dataset
    if getattr(config.experiment_arguments, 'define_experiment', False):
        # Define experiment
        define_args = config.define_experiment_arguments
        params = {
            **base_params,
            'dataset_name': getattr(data_args, 'dataset', 'cvdb'),
            'num_ents': getattr(data_args, 'num_ents', 4000),
            'test_frac': getattr(data_args, 'test_frac', None),
            'def_order': getattr(define_args, 'def_order', 'tve'),
            'entity_association_test_sets': getattr(define_args, 'entity_association_test_sets', False),
            'multiple_define_tags': getattr(define_args, 'multiple_define_tags', False),
            'incontext_defs': getattr(define_args, 'incontext_defs', False),
            'natural_style_vars': getattr(define_args, 'natural_style_vars', False),
            'natural_style_train_questions': getattr(define_args, 'natural_style_train_questions', False),
            'train_qs_multiplier': getattr(define_args, 'train_qs_multiplier', 1),
            'qd1_qd2_classification': getattr(define_args, 'qd1_qd2_classification', False),
        }
        params.update(override_params)
        
        logger.info(f"Generating define experiment data with the following parameters:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
        return get_questions_dataset(**params), params, config
        
    elif getattr(config.experiment_arguments, 'numeric_experiment', False):
        # Numeric experiment
        numeric_args = config.numeric_experiment_arguments
        
        # Check which type of numeric experiment
        if getattr(numeric_args, 'modular_experiment_baseline', False):
            return make_baseline_mod_div_data(seed=seed, train_subset=train_subset), params, config
            
        elif getattr(numeric_args, 'modular_experiment', False):
            return make_mod_division_dataset(seed=seed, train_subset=train_subset), params, config
            
        elif getattr(numeric_args, 'num_choice_experiment', False):
            params = {
                **base_params,
                'max_x': getattr(numeric_args, 'max_x', 99),
                'num_x': getattr(numeric_args, 'num_x', 500),
                'n_nums_in_question': getattr(numeric_args, 'n_nums_in_question', 4),
                'n_intersecton': getattr(numeric_args, 'n_intersecton', 2),
                'n_qs_per_x': getattr(numeric_args, 'n_qs_per_x', 24),
                'p_label_flip': getattr(numeric_args, 'p_label_flip', 0.0),
                'var_length': getattr(numeric_args, 'var_length', 3),
                'space_separated_var_names': not getattr(config.model_arguments, 'separate_token_per_var', False),
            }
            params.update(override_params)
            
            logger.info(f"Generating numeric choice experiment data with {len(params)} parameters")
            return make_num_selection_dataset(**params), params, config
        
        else:
            raise ValueError("No valid numeric experiment type specified in config")
    
    else:
        raise ValueError("Config doesn't specify a valid experiment type (define_experiment or numeric_experiment)")
