from dataclasses import dataclass, field
from typing import Optional

import yaml

from utils.logger import setup_logger

logger = setup_logger(__name__)



@dataclass
class ToyExampleArguments:
    n_seeds: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    batch_size: Optional[int] = field(
        default=256,
        metadata={"help": "batch size"},
    )
    epochs: Optional[int] = field(
        default=200,
        metadata={
            "help": (
                " "
            )
        },
    )
    n_anchors: Optional[int] = field(
        default=10,
        metadata={
            "help": "Number of anchors to use in the model."
        },
    )
    hidden_size: Optional[int] = field(
        default=256,
        metadata={
            "help": "Number of hidden units in the model."
        },
    )
    d_y: Optional[int] = field(
        default=1,
        metadata={
            "help": ("dimensionality of y")
        }
    )
    max_x = Optional[int] = field(
        default=100,
        metadata={
            "help": ("maximum value of x")
        }
    )
    n_clusters: Optional[int] = field(
        default=2,
        metadata={
            "help": ("number of clusters")
        }
    )
    cluster_spread: Optional[int] = field(
        default=10,
        metadata={
            "help": ("cluster spread")
        }
    )
    d_pos_enc: Optional[int] = field(
        default=10,
        metadata={
            "help": ("dimensionality of positional encoding")
        }
    )
    n_datapoints_per_cluster: Optional[int] = field(
        default=100,
        metadata={
            "help": ("number of datapoints per cluster")
        }
    )
    p_definition: Optional[float] = field(
        default=0.5,
        metadata={
            "help": ("probability of definition")
        }
    )

    

@dataclass
class CommonExperimentArguments:
    n_jobs: Optional[int] = field(
        default=1, metadata={"help": "The number of jobs to run in parallel (second stage)."}
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
    toy_example_arguments: ToyExampleArguments
    experiment_arguments: CommonExperimentArguments
    # experiment arguments     
    sweep_arguments: dict
    
    @classmethod
    def from_yaml(cls, file_path: str):
        logger.info('Loading configuration from yaml file: %s' % file_path)
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        toy_example_args = ToyExampleArguments(**config_dict['toy_example_arguments'])
        experiment_args = CommonExperimentArguments(**config_dict['experiment_arguments'])
        return cls(toy_example_args,
                   experiment_args,
                   sweep_arguments=config_dict.get('sweep_arguments', {}))
        
        
# def override_args(args, override_dict):
#     """Overrides args (dataclass) with values in override_dict (dict).
#     Args:
#         args (_type_): _description_
#         override_dict (_type_): _description_

#     Returns:
#         Arguments: dataclass containing subclasses with updated values.
#     """
#     args_copy = deepcopy(args)
#     # iterate over [training_args, numeric_exp_args, ...]
#     for args_set_name in vars(args_copy):
#         args_set = getattr(args_copy, args_set_name)
#         # do not overwrite arguments which we don't want to override.
#         if args_set_name not in ('first_stage_arguments', 'second_stage_arguments', 'third_stage_arguments', 'sweep_arguments'):
#             for key, value in override_dict.items():
#                 if hasattr(args_set, key):
#                     setattr(args_set, key, value)

#             setattr(args_copy, args_set_name, args_set)

#     return args_copy
