from functools import partial
from data_generation.define_experiment import get_questions_dataset


def get_datasets(args, args_stage1, args_stage2, stage):

    if stage == 'first_stage':
        train_subset = args_stage1.data_arguments.train_subset
    elif stage == 'second_stage':
        train_subset = args_stage2.data_arguments.train_subset
    else:
        train_subset = args.data_arguments.train_subset

    if args.experiment_arguments.define_experiment:
        if args.define_experiment_arguments.mix_reliable_unreliable_data:
            get_questions_dataset_fn = partial(get_questions_dataset,
                                               frac_n_qd1consis=0.25,
                                               frac_n_qd2incons=0.25,
                                               frac_n_q=0.1,
                                               frac_n_d1consis=0.1,
                                               frac_n_d2consis=0.1,
                                               frac_n_no_qd_baseline=0.1,
                                               frac_n_q_no_replacement_baseline=0.1,
                                               dataset_name=args.data_arguments.dataset,
                                               num_ents=args.experiment_arguments.num_ents,
                                               def_order=args.define_experiment_arguments.def_order)
            
            
            raw_datasets = get_questions_dataset_fn(seed=args_stage1.training_arguments.seed,
                                                    seed_stage2=args_stage2.training_arguments.seed,
                                                    train_subset=train_subset)

        elif args.define_experiment_arguments.no_relevant_defns:
            get_questions_dataset_fn = partial(get_questions_dataset,
                                               frac_n_qd1consis=0.0,
                                               frac_n_qd2incons=0.0,
                                               frac_n_q=0.4,
                                               frac_n_d1consis=0.25,
                                               frac_n_d2consis=0.0,
                                               frac_n_no_qd_baseline=0.1,
                                               frac_n_q_no_replacement_baseline=0.25,
                                               dataset_name=args.data_arguments.dataset,
                                               num_ents=args.experiment_arguments.num_ents,
                                               def_order=args.define_experiment.def_order)

            raw_datasets = get_questions_dataset_fn(seed=args_stage1.training_arguments.seed,
                                                    seed_stage2=args.training_arguments.seed,
                                                    train_subset=train_subset)

        else:
            raw_datasets = get_questions_dataset(seed=args_stage1.training_arguments.seed,
                                                 seed_stage2=args_stage2.training_arguments.seed,
                                                 dataset_name=args.data_arguments.dataset,
                                                 train_subset=train_subset,
                                                 num_ents=args.experiment_arguments.num_ents,
                                                 def_order=args.define_experiment_arguments.def_order)
    # elif args.experiment_arguments.numeric_experiment:
    #     if args.numeric_experiment_arguments.modular_experiment_baseline:
    #         raw_datasets = make_baseline_mod_div_data(seed=args.training_args.seed,
    #                                                   train_subset=data_args.train_subset)

    #     elif args.numeric_experiment_arguments.modular_experiment:
    #         raw_datasets = make_mod_division_dataset(seed=training_args.seed,
    #                                                  train_subset=data_args.train_subset)

    #     elif args.numeric_experiment_arguments.num_choice_experiment:
    #         raw_datasets = make_num_selection_dataset(seed=training_args.seed,
    #                                                   train_subset=data_args.train_subset,
    #                                                   max_x=args.numeric_experiment_arguments.max_x,
    #                                                   num_x=args.numeric_experiment_arguments.num_x,
    #                                                   n_nums_in_question=args.numeric_experiment_arguments.n_nums_in_question,
    #                                                   n_intersecton=args.numeric_experiment_arguments.n_intersecton,
    #                                                   n_qs_per_x=args.numeric_experiment_arguments.n_qs_per_x,
    #                                                   p_label_flip=args.numeric_experiment_arguments.p_label_flip)

    #     else:
    #         raise ValueError('Must specify a numeric experiment type (num_choice_experiment, modular_experiment, or modular_experiment_baseline)')
    # # experiment with paragraphs and questions about them
    # else:
    #     raw_datasets = get_raw_datasets(seed=args.training_arguments.seed, concat_pairs=args.data_arguments.paired_paragraphs)
    return raw_datasets