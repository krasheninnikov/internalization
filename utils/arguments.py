from dataclasses import dataclass, field, is_dataclass
from typing import Optional, List, Dict
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, Seq2SeqTrainingArguments
import yaml
from copy import deepcopy
from utils.logger import setup_logger


logger = setup_logger(__name__)
# MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
# MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The checkpoint for weights initialization. Don't set if you want to train a model from scratch.")},
    )
    model_type: Optional[str] = field(
        default=None,
        # metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={"help": (
                  "Override some existing default config settings when a model is trained from scratch. Example: "
                  "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index")},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name (of `transformers` model) or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    separate_token_per_var: bool = field(
        default=False, 
        metadata={"help": ("Whether to use a separate token for each variable name. Used only in numeric exp / set inclusion.")})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
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
        metadata={"help": (
                  "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                  "with private models).")},
    )
    max_new_tokens: int = field(
        default=20,
        metadata={"help": ("Maximum number of new tokens to generate during evaluation.")}
    )
    seq2seq: bool = field(
        default=False,
        metadata={"help": ("Whether seq2seq model is going to be used; otherwise we assume a causal lm.")}
    )
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError("--config_overrides can't be used in combination with --config_name or --model_name_or_path")


@dataclass
class ModelTrainingArguments(Seq2SeqTrainingArguments):
    save_each_epochs: Optional[int] = field(
        default=None, metadata={"help": ("Make a checkpoint each `save_each_epochs`")}
    )
    eval_each_epochs: Optional[int] = field(
        default=1, metadata={"help": "Perform evaluation every eval_each_epochs which calculates EM/F1"}
    )
    calculate_grad_variance: Optional[bool] = field(
        default=False, metadata={"help": "Whether to calculate gradient variance; note that this slows down training substantially."}
    )
    grad_keys: Optional[str] = field(
        default='train_defs_d1consis,train_defs_d2consis,d1consis,d2consis',
        metadata={"help": "Keys to calculate gradient variance for; NOTE: order matters here. See src/callbacks/GradientVarianceCallback for usage."}
    )
    eval_callback_type: Optional[str] = field(
        default='pipeline', metadata={"help": "Evaluation callback type. Use `pipeline` for clm and `generate` for seq2seq"}
    )
    dont_save_in_the_end: Optional[bool] = field(
        default=False, metadata={"help": "Don't save the model in the end."}
    )
    remove_checkpoints_in_the_end: Optional[bool] = field(
        default=True, metadata={"help": "Delete all *pytorch_model* files at the end of a training run to save memory."}
    )
    deterministic_sampler: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use a deterministic sampler for training."}
    )
    do_sweeps: Optional[bool] = field(
        default=False, metadata={"help": "Whether to do hyperparameters search."}
    )
    n_sweeps: Optional[int] = field(
        default=5, metadata={"help": "Number of hyperparameter sweeps to do."}
    )
    def __post_init__(self):
        super().__post_init__()  # sets logging dir
        if self.eval_callback_type not in ('pipeline', 'generate'):
            raise ValueError('invalid eval_callback type.')


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset: Optional[str] = field(
        default='cvdb', metadata={"help": "The name of the dataset to use (cvdb, trex)."}
    )
    num_ents: Optional[int] = field(
        default=4000,
        metadata={"help": ("Number of ents used to generate the data; should be up to 120k for cvdb;"
                           " can make much more with modifications but would need to make genders unbalanced")},
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": ("For debugging purposes; truncate the number of training examples to this value.")},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": ("For debugging purposes; truncate the number of evaluation examples to this value if set.")},
    )
    block_size: Optional[int] = field(
        default=1024, metadata={"help": ("Optional; input sequence length after tokenization. "
                                         "The training dataset will be chunked in blocks of this size for training.")},
    )
    label_block_size: Optional[int] = field(
        default=48, metadata={"help": ("Optional; label sequence length after tokenization. "
                                       "The training labels will be truncated in blocks of this size for training. This is used for seq2seq.")},
    )
    train_subset: Optional[str] = field(
        default='full', metadata={"help": ("Control data subsets included in the training data. "
                                           "One of (full, stage1, stage2, stage1_only_defns, stage1_only_qa).")}
    )
    # fractions of entities to use in various data subsets
    frac_n_qd1consis: Optional[float] = field(
        default=0.25, metadata={"help": "fraction of entities to use for qd1consis (see data_generation/define_experiment.py)."}
    )
    frac_n_qd1incons: Optional[float] = field(
        default=0.0, metadata={"help": "fraction of entities to use for qd1incons (see data_generation/define_experiment.py)."}
    )
    frac_n_qd2consis: Optional[float] = field(
        default=0.0, metadata={"help": "fraction of entities to use for `n_qd2consis` (see data_generation/define_experiment.py)."}
    )
    frac_n_qd2incons: Optional[float] = field(
        default=0.25, metadata={"help": "fraction of entities to use for `n_qd2incons` (see data_generation/define_experiment.py)."}
    )
    frac_n_qd4consis: Optional[float] = field(
        default=0.0, metadata={"help": "fraction of entities to use for `n_qd4consis` (see data_generation/define_experiment.py)."}
    )
    frac_n_q: Optional[float] = field(
        default=0.1, metadata={"help": "fraction of entities to use for `n_q` (see data_generation/define_experiment.py)."}
    )
    frac_n_q_no_replacement_baseline: Optional[float] = field(
        default=0.1, metadata={"help": "fraction of entities to use for `n_q_no_replacement_baseline` (see data_generation/define_experiment.py)."}
    )
    frac_n_d1consis: Optional[float] = field(
        default=0.1, metadata={"help": "fraction of entities to use for `n_d1consis` (see data_generation/define_experiment.py)."}
    )
    frac_n_d2consis: Optional[float] = field(
        default=0.1, metadata={"help": "fraction of entities to use for `n_d2consis` (see data_generation/define_experiment.py)."}
    )
    frac_n_d3consis: Optional[float] = field(
        default=0.0, metadata={"help": "fraction of entities to use for `n_d3consis` (see data_generation/define_experiment.py)."}
    )
    frac_n_no_qd_baseline: Optional[float] = field(
        default=0.1, metadata={"help": "fraction of entities to use for `qd_baseline` (see data_generation/define_experiment.py)."}
    )

    # Some default args for train_lm.py
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."},
    )
    ignore_pad_token_for_loss: bool = field(
        default=True, metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."},
    )

@dataclass 
class RandomNumsExperimentDataArguments:
    n_vars: Optional[int] = field(default=400, metadata={"help": "Number of variables in the synthetic data."})
    seq_len: Optional[int] = field(default=10, metadata={"help": "Length of the sequences in the synthetic data."})
    var_len: Optional[int] = field(default=5, metadata={"help": "Number of characters in the variable name."})
    


@dataclass
class NumericExperimentDataArguments:
    """
    Arguments pertaining to the num_choice experiment.
    """
    modular_experiment: Optional[bool] = field(
        default=False, metadata={"help": "Whether to perform Modular experiment (an exp similar to set inclusion). "}
    )
    modular_experiment_baseline: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use baseline data for the Modular experiment. "}
    )
    num_choice_experiment: Optional[bool] = field(
        default=False, metadata={"help": "Num choice / set inclusion experiment."}
    )
    max_x: Optional[int] = field(default=99,
                                 metadata={"help": ("Max value an 'entity' can take in our synthetic data.")},
    )
    num_x: Optional[int] = field(default=500,
                                 metadata={"help": ("Number of ent->var pairs in the synthetic data.")},
    )
    n_nums_in_question: Optional[int] = field(default=4,
                                              metadata={"help": ("Number of numbers in each question.")},
    )
    n_intersecton: Optional[int] = field(default=2,
                                         metadata={"help": ("Intersection length used to generate synthetic data.")},
    )
    n_qs_per_x: Optional[int] = field(default=2*12,
                                      metadata={"help": ("Number of questions per ent->var pair.")},
    )
    p_label_flip: Optional[float] = field(default=0.0,
                                          metadata={"help": ("The probability of flipping the label.")},
    )
    var_length: Optional[int] = field(default=3,
                                      metadata={"help": ("Number of characters in the variable name.")},
    )

    def __post_init__(self):
        assert 26**self.var_length > self.num_x, "var_length is too small for num_x"


@dataclass
class DefineExperimentDataArguments:
    def_order: Optional[str] = field(
        default='tve',
        metadata={"help": "The order of Tag, Variable and Entity in definitions."}
    )
    data_order_group_size: Optional[int] = field(
        default=0,
        metadata={"help": "0 means no grouping. Values >0 are meant to be used with the deterministic_sampler."}
    )
    entity_association_test_sets: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to include the entity association test sets."}
    )
    tag1_name: Optional[str] = field(
        default=None,
        metadata={"help": "The first define tag (if you don't want an rng one)."}
    )
    tag2_name: Optional[str] = field(
        default=None,
        metadata={"help": "The second define tag (if you don't want an rng one)."}
    )
    tag3_name: Optional[str] = field(
        default=None,
        metadata={"help": "The third define tag (if you don't want an rng one)."}
    )
    
    multiple_define_tags: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use multiple define tags. Works only with 'is_isnt' and 'natural_language' definition types."}
    )
     
    incontext_defs: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use incontext definitions."}
    )
    
    natural_style_vars: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use natural style variables."}
    )
    
    natural_style_train_questions: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use natural style train questions."}
    )
    
    qd1_qd2_classification: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use qd1_qd2_classification question format (train & test)."}
    )

@dataclass
class CommonExperimentArguments:
    define_experiment: Optional[bool] = field(
        default=True, metadata={"help": "Whether we perform the Define experiment."}
    )
    numeric_experiment: Optional[bool] = field(
        default=False, metadata={"help": "Whether we perform the toy numeric experiment."}
    )
    random_nums_experiment: Optional[bool] = field(
        default=False, metadata={"help": "Whether we perform the random numbers experiment."}
    )
    n_stages: Optional[int] = field(
        default=2, metadata={"help": "Number of stages of experiment. Currently maximum 3 stages are supported."}
    )
    n_seeds: Optional[int] = field(
        default=1, metadata={"help": "The number of times to repeat the experiment (first stage)."}
    )
    start_seed: Optional[int] = field(
        default=0, metadata={"help": "The starting seed for the experiment."}
    )
    n_seeds_stage2: Optional[int] = field(
        default=1, metadata={"help": "The number of seeds to use for stage 2."}
    )
    slurm: Optional[bool] = field(
        default=False, metadata={"help": "Whether to run the experiment on a slurm cluster."}
    )
    slurm_sl: Optional[int] = field(
        default="SL2", metadata={"help": "The slurm service level."}
    )
    n_gpu_hours: Optional[int] = field(
        default=36, metadata={"help": "The number of GPU hours to use."}
    )
    name_prefix: Optional[str] = field(
        default='', metadata={"help": "Prefix to add to experiment name."}
    )


@dataclass
class Config:
    data_arguments: DataTrainingArguments
    model_arguments: ModelArguments
    training_arguments: ModelTrainingArguments
    experiment_arguments: CommonExperimentArguments
    define_experiment_arguments: DefineExperimentDataArguments
    numeric_experiment_arguments: NumericExperimentDataArguments
    random_nums_experiment_arguments: RandomNumsExperimentDataArguments

    # generic container for per-stage overrides
    stage_specific_arguments: List[Dict] = field(default_factory=list)

    sweep_arguments: Dict = field(default_factory=dict)
    
    # legacy aliases so original ≤3-stage pipeline classes still work
    # (they are *filled* in __post_init__)
    first_stage_arguments: Dict = field(init=False, repr=False)
    second_stage_arguments: Dict = field(init=False, repr=False)
    third_stage_arguments: Dict = field(init=False, repr=False)

    @classmethod
    def from_yaml(cls, file_path: str) -> "Config":
        logger.info("Loading configuration from yaml file: %s", file_path)
        with open(file_path, "r") as fh:
            cfg = yaml.safe_load(fh) or {}

        # ------------------------------------------------------------------
        # 1 ▸ fix older transformers arg name, if present
        # ------------------------------------------------------------------
        tr_args = cfg.get("training_arguments", {})
        if "evaluation_strategy" in tr_args:
            tr_args["eval_strategy"] = tr_args.pop("evaluation_strategy")

        # ------------------------------------------------------------------
        # 2 ▸ instantiate each argument block (empty dict → defaults)
        # ------------------------------------------------------------------
        data_args     = DataTrainingArguments(**cfg.get("data_arguments", {}))
        model_args    = ModelArguments(**cfg.get("model_arguments", {}))
        train_args    = ModelTrainingArguments(**tr_args)
        exper_args    = CommonExperimentArguments(**cfg.get("experiment_arguments", {}))
        define_args   = DefineExperimentDataArguments(**cfg.get("define_experiment_arguments", {}))
        numeric_args  = NumericExperimentDataArguments(**cfg.get("numeric_experiment_arguments", {}))
        randomn_args  = RandomNumsExperimentDataArguments(**cfg.get("random_nums_experiment_arguments", {}))

        # ------------------------------------------------------------------
        # 3 ▸ collect stage-override blocks
        #    (support both the new list key AND the old first/second/third
        #     keys for full back-compatibility)
        # ------------------------------------------------------------------
        overrides: List[Dict] | None = cfg.get("stage_specific_arguments")
        if overrides is None:
            # fall back to first_/second_/third_stage_arguments if present
            key_map = {
                1: "first_stage_arguments",
                2: "second_stage_arguments",
                3: "third_stage_arguments",
                4: "fourth_stage_arguments",
                5: "fifth_stage_arguments",
            }
            n = exper_args.n_stages or 1
            overrides = [cfg.get(key_map.get(i + 1), {}) for i in range(n)]
            
        # Assert n_stages matches provided configs
        if exper_args.n_stages:
            assert exper_args.n_stages == len(overrides), \
                f"n_stages ({exper_args.n_stages}) must match number of provided stage configs ({len(overrides)})"

        # ------------------------------------------------------------------
        # 4 ▸ construct and return the dataclass
        # ------------------------------------------------------------------
        return cls(
            data_arguments=data_args,
            model_arguments=model_args,
            training_arguments=train_args,
            experiment_arguments=exper_args,
            define_experiment_arguments=define_args,
            numeric_experiment_arguments=numeric_args,
            random_nums_experiment_arguments=randomn_args,
            stage_specific_arguments=overrides,
            sweep_arguments=cfg.get("sweep_arguments", {}),
        )

    def __post_init__(self):
        # For backward compatibility with original Single/Two/ThreeStageFineTuning classes
        ov = self.stage_specific_arguments + [{}] * 3 
        self.first_stage_arguments, self.second_stage_arguments, \
            self.third_stage_arguments = ov[:3]

        if self.model_arguments.seq2seq and self.training_arguments.eval_callback_type == 'pipeline':
            logger.warning('"pipeline" evaluation callback is not supported for seq2seq; switching to "generate"')
            self.training_arguments.eval_callback_type = 'generate'


def override_args(base_cfg, override: Dict):
    """
    Return a *deep copy* of `base_cfg` with selected fields overwritten.

    ────────────────────────────────────────────────────────────────────────────
    Assumption / contract
    ────────────────────────────────────────────────────────────────────────────
    • **Dataclass attributes** of the top-level Config (e.g. `training_arguments`,
      `model_arguments`, `data_arguments`, …) represent *actual hyper-parameters*.
      Keys found in `override` that also exist as fields inside these dataclasses
      **may be modified**.

    • **Non-dataclass attributes** (plain `dict`, `list`, etc.—such as
      `stage_specific_arguments`, `sweep_arguments`, historical
      `first_stage_arguments`, …) are treated as *metadata* that describes
      *how* to perform overrides, not things that should themselves be patched.
      They are therefore **left untouched**.

    If you later add a new argument bundle:
      • make it a `@dataclass` → it will be auto-overridable;
      • leave it a `dict`/`list`  → you must handle overrides manually.

    Parameters
    ----------
    base_cfg : Config
        Original configuration object.
    override : dict
        Flat dictionary of `{field_name: new_value}` pairs intended to patch
        every dataclass sub-object.

    Returns
    -------
    Config
        A *new* Config instance with the requested overrides applied.
    """
    cfg = deepcopy(base_cfg)

    # iterate over every top-level attribute in the Config
    for attr_name in vars(cfg):
        attr_val = getattr(cfg, attr_name)

        # Only patch those attributes that are dataclass instances
        # (i.e. real argument bundles).  Metadata containers are skipped.
        if not is_dataclass(attr_val):
            continue

        # Apply the override wherever the field exists in the dataclass.
        for key, value in override.items():
            if hasattr(attr_val, key):
                setattr(attr_val, key, value)

    return cfg
