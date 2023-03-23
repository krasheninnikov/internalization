from dataclasses import dataclass, field
from typing import Optional
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, TrainingArguments, Seq2SeqTrainingArguments
import yaml
from copy import deepcopy
from utils.logger import setup_logger


logger = setup_logger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    max_new_tokens: int = field(
        default=20,
        metadata={
            "help": (
                "Max number of new tokens to generate. "
            )
        }
    )
    seq2seq: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether seq2seq model is going to be used or not. "
            )
        }
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    deterministic_sampler: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use a deterministic sampler for training."}
    )
    dataset: Optional[str] = field(
        default='cvdb', metadata={"help": "The name of the dataset to use (cvdb, squad, archival)."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    block_size: Optional[int] = field(
        default=1024, metadata={"help": ("Optional input sequence length after tokenization. "
                                         "The training dataset will be truncated in block of this size for training. "
                                         "Default to the model max input length for single sentence inputs (take into account special tokens).")},
    )
    label_block_size: Optional[int] = field(
        default=48, metadata={"help": ("Optional input sequence length after tokenization. "
                                       "The training dataset will be truncated in block of this size for training. "
                                       "Default to the model max input length for single sentence inputs (take into account special tokens).")},
    )
    dont_save_in_the_end: Optional[bool] = field(
        default=False, metadata={"help": "Don't save the model in the end."}
    )
    
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5, metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True, metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."},
    )
    train_subset: Optional[str] = field(
        default='full', metadata={"help": ("Param for the define experiment. "
                                           "One of (full, stage1, stage2, stage1_only_defns, stage1_only_qa)")}
    )
    paired_paragraphs: Optional[bool] = field(
        default=False, metadata={"help": "Whether the SQUAD paragraphs should be single paragraphs or concatenated "
                                         "pairs of paragraphs."}
    )


@dataclass
class NumericExperimentDataArguments:
    """
    Arguments pertaining to the num_choice experiment.
    """
    modular_experiment: Optional[bool] = field(
        default=False, metadata={"help": "Whether we use baseline data for the Modular experiment. "}
    )
    modular_experiment_baseline: Optional[bool] = field(
        default=False, metadata={"help": "Whether we use baseline data for the Modular experiment. "}
    )
    num_choice_experiment: Optional[bool] = field(
        default=False, metadata={"help": "Num choice experiment. "}
    )
    max_x: Optional[int] = field(default=99, metadata={"help": ("")},)
    num_x: Optional[int] = field(default=500, metadata={"help": ("")},)
    n_nums_in_question: Optional[int] = field(default=4, metadata={"help": ("")},)
    n_intersecton: Optional[int] = field(default=2, metadata={"help": ("")},)
    n_qs_per_x: Optional[int] = field(default=2*12, metadata={"help": ("")},)
    p_label_flip: Optional[float] = field(default=0.0, metadata={"help": ("")},)


@dataclass
class DefineExperimentDataArguments:
    def_order: Optional[str] = field(
        default='tve', metadata={"help": "The order of Tag, Variable and Entity in definitions."}
    )
    no_relevant_defns: Optional[bool] = field(
        default=False, metadata={"help": "The Define experiment where in the train set defns don't correspond to any questions"}
    )
    mix_reliable_unreliable_data: Optional[bool] = field(
        default=True, metadata={"help": "See mix_reliable_unreliable_data in data_utils_define_experiment.py"}
    )

@dataclass
class CommonExperimentArguments:
    define_experiment: Optional[bool] = field(
        default=True, metadata={"help": "Whether we perform the Define experiment. "
                                 "If False, paragraphs-as-defns experiment is performed."}
    )
    numeric_experiment: Optional[bool] = field(
        default=False, metadata={"help": "Whether we perform the toy numeric experiment. "}
    )
    num_ents: Optional[int] = field(
        default=4000,
        metadata={"help": ("number of ents used to generate the data to generate; should be up to 120k for cvdb;"
                           " can make much more with modifications but would need to make genders unbalanced")},
    )
    n_stages: Optional[int] = field(
        default=2, metadata={"help": "Number of stages of experiment. Currently maximum 3 stages are supported"}
    )
    n_seeds: Optional[int] = field(
        default=1, metadata={"help": "The number of times to repeat the experiment (first stage)."}
    )
    start_seed: Optional[int] = field(
        default=0, metadata={"help": "The starting seed for the experiment. "}
    )
    n_seeds_stage2: Optional[int] = field(
        default=1, metadata={"help": "The number of seeds to use for stage 2."}
    )
    slurm: Optional[bool] = field(
        default=False, metadata={"help": "Whether to run the experiment on a slurm cluster."}
    )
    do_sweeps: Optional[bool] = field(
        default=False, metadata={"help": "Whether to do hyperparameters search or not."}
    )
    save_each_epochs: Optional[int] = field(
        default=None, metadata={"help": ("Make a checkpoint each `save_each_epochs`")}
    )
    
    eval_each_epochs: Optional[int] = field(
        default=1, metadata={"help": "Perform evaluation every eval_each_epochs which calculates EM/F1"}
    )
    eval_callback_type: Optional[str] = field(
        default='pipeline', metadata={"help": "The evaluation callback type. Use `pipeline` for clm and `generate` for seq2seq"}
    )
    
    def __post_init__(self):
        if self.eval_callback_type not in ('pipeline', 'generate'):
            raise ValueError('invalid eval_callback type.')


@dataclass
class Config:
    data_arguments: DataTrainingArguments
    model_arguments: ModelArguments
    training_arguments: Seq2SeqTrainingArguments
    # experiment arguments 
    experiment_arguments: CommonExperimentArguments
    define_experiment_arguments: DefineExperimentDataArguments
    numeric_experiment_arguments: NumericExperimentDataArguments
    
    first_stage_arguments: dict # overrides for training arguments
    second_stage_arguments: dict
    third_stage_arguments: dict
    
    sweep_arguments: dict
    
    @classmethod
    def from_yaml(cls, file_path: str):
        logger.info('Loading configuration from yaml file: %s' % file_path)
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        data_arguments = DataTrainingArguments(**config_dict['data_arguments'])
        model_arguments = ModelArguments(**config_dict['model_arguments'])
        training_arguments = Seq2SeqTrainingArguments(**config_dict['training_arguments'])
        # experiment arguments 
        experiment_arguments = CommonExperimentArguments(**config_dict['experiment_arguments'])
        define_experiment_arguments = DefineExperimentDataArguments(**config_dict['define_experiment_arguments'])
        numeric_experiment_arguments = NumericExperimentDataArguments(**config_dict['numeric_experiment_arguments'])
        return cls(data_arguments,
                   model_arguments,
                   training_arguments,
                   experiment_arguments,
                   define_experiment_arguments,
                   numeric_experiment_arguments,
                   first_stage_arguments=config_dict.get('first_stage_arguments', {}),
                   second_stage_arguments=config_dict.get('second_stage_arguments', {}),
                   third_stage_arguments=config_dict.get('third_stage_arguments', {}),
                   sweep_arguments=config_dict.get('sweep_arguments', {}))
        
    def __post_init__(self):
        if self.model_arguments.seq2seq and self.experiment_arguments.eval_callback_type == 'pipeline':
            raise ValueError('pipeline evaluation callback is not supported for seq2seq.')


def override_args(args, override_dict):
    """Overrides args (dataclass) with values in override_dict (dict).
    Args:
        args (_type_): _description_
        override_dict (_type_): _description_

    Returns:
        Arguments: dataclass containing subclasses with updated values.
    """
    args_copy = deepcopy(args)
    # iterate over [training_args, numeric_exp_args, ...]
    for args_set_name in vars(args_copy):
        args_set = getattr(args_copy, args_set_name)
        # do not overwrite arguments which are used to override.
        # TODO: avoid sweep arguments?
        if args_set_name not in ('first_stage_arguments', 'second_stage_arguments', 'third_stage_arguments'):
            for key, value in override_dict.items():
                if hasattr(args_set, key):
                    setattr(args_set, key, value)

            setattr(args_copy, args_set_name, args_set)

    return args_copy
