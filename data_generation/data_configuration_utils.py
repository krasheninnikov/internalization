import random
from data_generation.define_experiment import get_questions_dataset
from data_generation.numeric_experiment import make_mod_division_dataset, make_baseline_mod_div_data, make_num_selection_dataset
from data_generation.squad_data import get_raw_datasets


def get_experiment_dataset(args, seed_stage1, seed_stage2, train_subset=None):

    if args.experiment_arguments.define_experiment:
        def_args = args.define_experiment_arguments
        raw_datasets = get_questions_dataset(frac_n_qd1consis=def_args.frac_n_qd1consis,
                                             frac_n_qd1incons=def_args.frac_n_qd1incons,
                                             frac_n_qd2incons=def_args.frac_n_qd2incons,
                                             frac_n_q=def_args.frac_n_q,
                                             frac_n_d1consis=def_args.frac_n_d1consis,
                                             frac_n_d2consis=def_args.frac_n_d2consis,
                                             frac_n_no_qd_baseline=def_args.frac_n_no_qd_baseline,
                                             frac_n_q_no_replacement_baseline=def_args.frac_n_q_no_replacement_baseline,
                                             dataset_name=args.data_arguments.dataset,
                                             num_ents=args.data_arguments.num_ents,
                                             def_order=def_args.def_order,
                                             data_order_group_size=def_args.data_order_group_size,
                                             seed=seed_stage1,
                                             seed_stage2=seed_stage2,
                                             train_subset=train_subset)
        
        # if args.def_args.mix_reliable_unreliable_data:
        #     raw_datasets = get_questions_dataset(frac_n_qd1consis=0.25,
        #                                          frac_n_qd2incons=0.25,
        #                                          frac_n_q=0.1,
        #                                          frac_n_d1consis=0.1,
        #                                          frac_n_d2consis=0.1,
        #                                          frac_n_no_qd_baseline=0.1,
        #                                          frac_n_q_no_replacement_baseline=0.1,
        #                                          dataset_name=args.data_arguments.dataset,
        #                                          num_ents=args.data_arguments.num_ents,
        #                                          def_order=args.def_args.def_order,
        #                                          data_order_group_size=args.def_args.data_order_group_size,
        #                                          seed=seed_stage1,
        #                                          seed_stage2=seed_stage2,
        #                                          train_subset=train_subset)
        # elif args.def_args.include_qd1incons:
        #     raw_datasets = get_questions_dataset(frac_n_qd1consis=0.23,
        #                                          frac_n_qd1incons=0.02,
        #                                          frac_n_qd2incons=0.25,
        #                                          frac_n_q=0.1,
        #                                          frac_n_d1consis=0.1,
        #                                          frac_n_d2consis=0.1,
        #                                          frac_n_no_qd_baseline=0.1,
        #                                          frac_n_q_no_replacement_baseline=0.1,
        #                                          dataset_name=args.data_arguments.dataset,
        #                                          num_ents=args.data_arguments.num_ents,
        #                                          def_order=args.def_args.def_order,
        #                                          data_order_group_size=args.def_args.data_order_group_size,
        #                                          seed=seed_stage1,
        #                                          seed_stage2=seed_stage2,
        #                                          train_subset=train_subset)
        # elif args.def_args.no_relevant_defns:
        #     raw_datasets = get_questions_dataset(frac_n_qd1consis=0.0,
        #                                          frac_n_qd2incons=0.0,
        #                                          frac_n_q=0.4,
        #                                          frac_n_d1consis=0.25,
        #                                          frac_n_d2consis=0.0,
        #                                          frac_n_no_qd_baseline=0.1,
        #                                          frac_n_q_no_replacement_baseline=0.25,
        #                                          dataset_name=args.data_arguments.dataset,
        #                                          num_ents=args.data_arguments.num_ents,
        #                                          def_order=args.define_experiment.def_order,
        #                                          data_order_group_size=args.def_args.data_order_group_size,
        #                                          seed=seed_stage1,
        #                                          seed_stage2=seed_stage2,
        #                                          train_subset=train_subset)
        # else:
        #     raw_datasets = get_questions_dataset(seed=seed_stage1,
        #                                          seed_stage2=seed_stage2,
        #                                          dataset_name=args.data_arguments.dataset,
        #                                          train_subset=train_subset,
        #                                          num_ents=args.data_arguments.num_ents,
        #                                          def_order=args.def_args.def_order,
        #                                          data_order_group_size=args.def_args.data_order_group_size,
        #                                          )
            
    elif args.experiment_arguments.numeric_experiment:
        if args.numeric_experiment_arguments.modular_experiment_baseline:
            raw_datasets = make_baseline_mod_div_data(seed=seed_stage1,
                                                      train_subset=train_subset)

        elif args.numeric_experiment_arguments.modular_experiment:
            raw_datasets = make_mod_division_dataset(seed=seed_stage1,
                                                     train_subset=train_subset)

        elif args.numeric_experiment_arguments.num_choice_experiment:
            raw_datasets = make_num_selection_dataset(seed=seed_stage1,
                                                      train_subset=train_subset,
                                                      max_x=args.numeric_experiment_arguments.max_x,
                                                      num_x=args.numeric_experiment_arguments.num_x,
                                                      n_nums_in_question=args.numeric_experiment_arguments.n_nums_in_question,
                                                      n_intersecton=args.numeric_experiment_arguments.n_intersecton,
                                                      n_qs_per_x=args.numeric_experiment_arguments.n_qs_per_x,
                                                      p_label_flip=args.numeric_experiment_arguments.p_label_flip)

        else:
            raise ValueError('Must specify a numeric experiment type (num_choice_experiment, modular_experiment, or modular_experiment_baseline)')
    # experiment with paragraphs and questions about them
    else:
        # TODO args.training_arguments.seed or seed_stage1?? Dima: arent they the same?
        raw_datasets = get_raw_datasets(seed=args.training_arguments.seed, concat_pairs=args.data_arguments.paired_paragraphs)
    return enforce_max_data_size(raw_datasets, args)


def enforce_max_data_size(raw_datasets, args):
    def select_random_sublist_preserve_order(dataset, n):
        # select indices randomly, but preserve order
        rng = random.Random(args.training_arguments.seed)
        n = min(n, len(dataset))
        idx = sorted(rng.sample(range(len(dataset)), n))
        return dataset.select(idx)
    
    if args.data_arguments.max_train_samples is not None and 'train' in raw_datasets:
        raw_datasets['train'] = select_random_sublist_preserve_order(raw_datasets['train'], args.data_arguments.max_train_samples)
    if args.data_arguments.max_eval_samples is not None:
        for subset in raw_datasets:
            if subset != 'train':
                raw_datasets[subset] = select_random_sublist_preserve_order(raw_datasets[subset], args.data_arguments.max_eval_samples)
    return raw_datasets
