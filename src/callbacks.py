from abc import ABC, abstractmethod
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (TrainerCallback, TrainerControl, TrainerState,
                          TrainingArguments, pipeline)
from transformers.integrations import TensorBoardCallback

import wandb
from src.metrics import compute_em_list, compute_f1_list
from utils.logger import setup_logger

from data_generation.pwd_locked_composition import eval_fn

logger = setup_logger(__name__)


class EvaluationCallbackBase(TensorBoardCallback, ABC):
    """Base class for evaluation callbacks."""
    def __init__(self, 
                 tb_writer=None, 
                 eval_each_epochs=False, 
                 eval_each_steps=False, 
                 evaluation_strategy='epoch', 
                 numeric_experiment=False):
        super().__init__(tb_writer)
        self.em_score = {}  # dict of em scores for each eval dataset
        self.f1_score = {}  # dict of f1 scores for each eval dataset
        self.eval_each_epochs = eval_each_epochs
        self.eval_each_steps = eval_each_steps
        self.numeric_experiment = numeric_experiment        
        self.evaluation_strategy = evaluation_strategy
        assert self.evaluation_strategy in ['epoch', 'steps', 'no']

    @abstractmethod
    def evaluate_fn(self, args, state, model, tokenizer):
        raise NotImplementedError
    
    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self.evaluation_strategy == 'epoch' and self.eval_each_epochs and round(state.epoch) % self.eval_each_epochs == 0:
            self.evaluate_fn(args, state, model, tokenizer)
            
    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if not self.eval_each_epochs and not self.eval_each_steps:
            # there weren't any evaluations during training
            self.evaluate_fn(args, state, model, tokenizer) # updates metrics dict
            
    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self.evaluation_strategy == 'steps' and self.eval_each_steps and round(state.global_step) % self.eval_each_steps == 0:
            self.evaluate_fn(args, state, model, tokenizer)


class EvaluationCallbackGenerate(EvaluationCallbackBase):
    def __init__(self,
                 eval_dataset_tokenized,
                 generate_batch_fn,
                 postprocess_output_fn,
                 tb_writer=None,
                 numeric_experiment=False,
                 eval_each_epochs=False, 
                 eval_each_steps=False, 
                 evaluation_strategy='epoch',):
        
        super().__init__(tb_writer, eval_each_epochs, eval_each_steps, evaluation_strategy, numeric_experiment)
        
        self.eval_dataset_tokenized = eval_dataset_tokenized
        self.generate_batch = generate_batch_fn
        self.postprocess_output_fn = postprocess_output_fn
        
    def evaluate_fn(self, args, state, model, tokenizer):
        if self.tb_writer is None:
            self._init_summary_writer(args)
        # set eval mode
        model.eval()
        
        for k in self.eval_dataset_tokenized:
            if 'train' in k: # we have eval subsets of the train set, e.g. qd1consis definitions; skip them
                continue
            
            logger.info(f'*** Evaluating on {k} ***')
            eval_dataset_k = self.eval_dataset_tokenized[k]
            # generate predictions using generate_batch_fn function
            eval_dataset_input = eval_dataset_k.remove_columns(['attention_mask', 'labels', 'answer'])
            generate_batch_fn = partial(self.generate_batch, model=model)
            predictions_k = eval_dataset_input.with_format('torch', device='cuda').map(
                generate_batch_fn,
                batched=True,
                load_from_cache_file=True,
                batch_size=args.per_device_eval_batch_size,
                remove_columns=['input_ids'],
                desc=f"Creating predictions for {k}",
            )
            # decode and aggregate predicted anwers
            predicted_answers = tokenizer.batch_decode(predictions_k['prediction'], skip_special_tokens=True)
            # apply postprocessing to predictions
            predicted_answers = [self.postprocess_output_fn(predicted_answer) for predicted_answer in predicted_answers]
            
            # this is a hack for numeric experiment with custom tokenizer, doesn't work for pretrained models.
            if self.numeric_experiment:
                predicted_answers = [x.split('[PAD]')[1].strip() for x in predicted_answers]
            
            # decode original answers
            original_answers = eval_dataset_k['answer']
            # apply postprocessing to original answers
            original_answers = [a.replace('\n', '').strip() for a in original_answers]
            
            self.em_score[k] = compute_em_list(predicted_answers, original_answers)
            self.f1_score[k] = compute_f1_list(predicted_answers, original_answers)

            self.tb_writer.add_scalar(f"eval/{k}_EM", self.em_score[k], state.global_step)
            self.tb_writer.add_scalar(f"eval/{k}_F1", self.f1_score[k], state.global_step)
            wandb.log({f"eval/{k}_EM": self.em_score[k]}, state.global_step)
            wandb.log({f"eval/{k}_F1": self.f1_score[k]}, state.global_step)
            
            for i in range(10):
                logger.info(f'Correct & predicted answers: {original_answers[i], predicted_answers[i]}\n')


class EvaluationCallbackPipeline(EvaluationCallbackBase):
    def __init__(self, 
                 eval_dataset_raw, 
                 tb_writer=None, 
                 numeric_experiment=False, 
                 eval_each_epochs=1, 
                 eval_each_steps=False, 
                 evaluation_strategy='epoch',
                 max_new_tokens=10,):
        super().__init__(tb_writer, eval_each_epochs, eval_each_steps, evaluation_strategy, numeric_experiment)
        self.eval_dataset_raw = eval_dataset_raw
        self.max_new_tokens = max_new_tokens
        
    def evaluate_fn(self, args, state, model, tokenizer):
        if self.tb_writer is None:
            self._init_summary_writer(args)
        
        model.eval()
        tokenizer.padding_side = 'left'
        pipe = pipeline(task='text-generation', model=model,
                        device=0, tokenizer=tokenizer, top_k=1)
        
        for k in self.eval_dataset_raw:
            if 'train' in k: # we have eval subsets of the train set, e.g. qd1consis definitions; skip them
                continue
            
            logger.info(f'*** Evaluating on {k} ***')
            eval_dataset_k = self.eval_dataset_raw[k]
            original_answers = eval_dataset_k['answer']
            qa_prompts = eval_dataset_k['question']
            predicted_answers = pipe(qa_prompts,
                                    max_new_tokens=self.max_new_tokens,
                                    pad_token_id=tokenizer.pad_token_id,
                                    batch_size=args.per_device_eval_batch_size,
                                    clean_up_tokenization_spaces=True,
                                    top_k=1,
                                    return_full_text=False)
            if self.numeric_experiment:
                # everything before [PAD] is the answer, everything after is garbage
                predicted_answers = [x[0]['generated_text'].split('[PAD]')[0].strip()
                        for x in predicted_answers]
            else:
                predicted_answers = [x[0]['generated_text'].strip()
                                     for x in predicted_answers]
            original_answers = [a.replace('\n', '').strip() for a in original_answers]
            self.em_score[k] = compute_em_list(predicted_answers, original_answers)
            self.f1_score[k] = compute_f1_list(predicted_answers, original_answers)

            self.tb_writer.add_scalar(f"eval/{k}_EM", self.em_score[k], state.global_step)
            self.tb_writer.add_scalar(f"eval/{k}_F1", self.f1_score[k], state.global_step)
            wandb.log({f"eval/{k}_EM": self.em_score[k]}, state.global_step)
            wandb.log({f"eval/{k}_F1": self.f1_score[k]}, state.global_step)

            for i in range(10):
                logger.info(f'Correct & predicted answers: {original_answers[i], predicted_answers[i]}\n')
                
                
class EvaluationCallbackPipelinePwdLocked(EvaluationCallbackBase):
    def __init__(self, 
                 eval_dataset_raw, 
                 tb_writer=None, 
                 numeric_experiment=False, 
                 eval_each_epochs=1, 
                 eval_each_steps=False, 
                 evaluation_strategy='epoch',
                 max_new_tokens=10,
                 # PWDLocked specific arguments below (needed to generate the fns for evaluation)
                 seed=0,
                 nfunc=4,
                 max_x=10,
                 fn_input_len=3,
                 n_fns_to_lock=2,
                 ):
        super().__init__(tb_writer, eval_each_epochs, eval_each_steps, evaluation_strategy, numeric_experiment)
        self.eval_dataset_raw = eval_dataset_raw
        self.max_new_tokens = max_new_tokens
        
        self.eval_fn = partial(eval_fn, seed=seed, nfunc=nfunc, max_x=max_x, fn_input_len=fn_input_len, n_fns_to_lock=n_fns_to_lock)
        
        
    def evaluate_fn(self, args, state, model, tokenizer):
        if self.tb_writer is None:
            self._init_summary_writer(args)
        
        model.eval()
        tokenizer.padding_side = 'left'
        pipe = pipeline(task='text-generation', model=model,
                        device=0, tokenizer=tokenizer, top_k=1)
        
        # STUFF BELOW IS MODIFIED FROM THE NORMAL PIPELINE EVALUATION
        for k in self.eval_dataset_raw:
            logger.info(f'*** Evaluating on {k} ***')
            eval_dataset_k = self.eval_dataset_raw[k]
            original_answers = eval_dataset_k['answer']
            qa_prompts = eval_dataset_k['question']
            predicted_answers = pipe(qa_prompts,
                                    max_new_tokens=self.max_new_tokens,
                                    pad_token_id=tokenizer.pad_token_id,
                                    batch_size=args.per_device_eval_batch_size,
                                    clean_up_tokenization_spaces=True,
                                    top_k=1,
                                    return_full_text=False)
            # everything before [PAD] is the answer, everything after is garbage
            predicted_answers = [x[0]['generated_text'].split('[PAD]')[0].strip()
                    for x in predicted_answers]
            
            # concatenate qa_prompts with predicted_answers
            predicted_answers = [f"{qa_prompts[i]} {predicted_answers[i]}" for i in range(len(predicted_answers))]
            for i in range(10):
                logger.info(f'Predicted ans: {predicted_answers[i]}')
            
            res = self.eval_fn(predicted_answers)
            
            # print('HERE')
            # raise ValueError('STOP')
            # original_answers = [a.replace('\n', '').strip() for a in original_answers]
            # self.em_score[k] = compute_em_list(predicted_answers, original_answers)
            # self.f1_score[k] = compute_f1_list(predicted_answers, original_answers)

            for subset_name in res:
                self.tb_writer.add_scalar(f"{k}_{subset_name}", res[subset_name], state.global_step)
                wandb.log({f"{k}_{subset_name}": res[subset_name]}, state.global_step)
                # wandb.log({f"eval/{k}_EM": self.em_score[k]}, state.global_step)

                # for i in range(10):
                #     logger.info(f'Correct & predicted answers: {original_answers[i], predicted_answers[i]}\n')


class CustomSaveCallback(TrainerCallback):
    """Callback for saving each n epochs."""
    def __init__(self, save_each_epochs) -> None:
        self.save_each_epochs = save_each_epochs
        
    def on_epoch_end(self, 
                     args: TrainingArguments,
                     state: TrainerState,
                     control: TrainerControl, **kwargs):

        if self.save_each_epochs > 0 and round(state.epoch) % self.save_each_epochs == 0:
            control.should_save = True

        return control



class GradientVarianceCallback(EvaluationCallbackBase):
    def __init__(self, eval_dataset_tokenized,
                 tb_writer=None, 
                 numeric_experiment=False, 
                 eval_each_epochs=1, 
                 eval_each_steps=False, 
                 evaluation_strategy='epoch') -> None:
        
        super().__init__(tb_writer, eval_each_epochs, eval_each_steps, evaluation_strategy, numeric_experiment)
        self.eval_dataset_tokenized = eval_dataset_tokenized
        
    # def evaluate_fn(self, args, state, model, tokenizer):
    #     if self.tb_writer is None:
    #         self._init_summary_writer(args)
            
    #     model.train()
    #     # print(self.eval_dataset_tokenized.keys())
    #     #keys = ['train_defs_d1consis', 'train_defs_d2consis', 'd1consis', 'd2consis']
    #     keys = ['train_defs_qd1consis', 'train_defs_qd2incons', 'train_questions_qd1consis', 'train_questions_qd2incons']
    #     self.eval_dataset_tokenized = {key: self.eval_dataset_tokenized[key] for key in keys}
    #     n_datapoints = sum([len(self.eval_dataset_tokenized[key]) for key in self.eval_dataset_tokenized])  # number of datapoints
    #     mean_grad = None

    #     # ========================
    #     logger.info('*** Computing gradient distance between definitions and corresponding questions ***')    
    #     # calculate average l2 distance between definitions and their corresponding questions
    #     step_size = len(self.eval_dataset_tokenized[keys[2]]) // len(self.eval_dataset_tokenized[keys[0]])  # number of questions per definition
    #     if step_size != len(self.eval_dataset_tokenized[keys[3]]) // len(self.eval_dataset_tokenized[keys[1]]):
    #         raise ValueError('step_size must be the same for both d1consis and d2consis')
        
    #     eval_dataset_d1cons = self.eval_dataset_tokenized[keys[2]].with_format('torch', device='cuda')
    #     eval_dataset_d1defs = self.eval_dataset_tokenized[keys[0]].with_format('torch', device='cuda')
        
    #     mean_dist_d1 = 0
    #     mean_sim_d1_cos = 0
    #     for i in tqdm(range(len(eval_dataset_d1defs))):
    #         d = eval_dataset_d1defs[i]
    #         d_grad = get_gradient(model, d)
    #         mean_d_dist = 0
    #         mean_d_sim_cos = 0
            
    #         # update mean_grad
    #         if mean_grad is None:
    #             mean_grad = d_grad
    #         else:
    #             mean_grad += d_grad
                
    #         for j in range(step_size):
    #             n = i * step_size + j  # index of question
    #             q = eval_dataset_d1cons[n]
    #             q_grad = get_gradient(model, q)
    #             mean_d_dist += torch.sqrt(torch.sum((d_grad - q_grad)**2)) # l2 distance between gradient of definition and gradient of question
    #             mean_d_sim_cos += torch.nn.functional.cosine_similarity(d_grad, q_grad, dim=0)
    #             mean_grad += q_grad
                
    #         mean_d_dist /= step_size  # transform sum into mean distance for this definition
    #         mean_d_sim_cos /= step_size
            
    #         mean_dist_d1 += mean_d_dist # mean distance for all definitions
    #         mean_sim_d1_cos += mean_d_sim_cos
            
    #     mean_dist_d1 /= len(eval_dataset_d1defs)
    #     mean_sim_d1_cos /= len(eval_dataset_d1defs)
        
    #     eval_dataset_d2cons = self.eval_dataset_tokenized[keys[3]].with_format('torch', device='cuda')
    #     eval_dataset_d2defs = self.eval_dataset_tokenized[keys[1]].with_format('torch', device='cuda')
        
    #     mean_dist_d2 = 0
    #     mean_sim_d2_cos = 0
    #     for i in tqdm(range(len(eval_dataset_d2defs))):
    #         d = eval_dataset_d2defs[i]
    #         d_grad = get_gradient(model, d)
    #         mean_d_dist = 0
    #         mean_d_sim_cos = 0
    #         # update mean_grad
    #         if mean_grad is None:
    #             mean_grad = d_grad
    #         else:
    #             mean_grad += d_grad
                
    #         for j in range(step_size):
    #             n = i * step_size + j
    #             q = eval_dataset_d2cons[n]
    #             q_grad = get_gradient(model, q)
    #             mean_d_dist += torch.sqrt(torch.sum((d_grad - q_grad)**2))
    #             mean_d_sim_cos += torch.nn.functional.cosine_similarity(d_grad, q_grad, dim=0)
    #             # update mean_grad
    #             mean_grad += q_grad
                
    #         mean_d_dist /= step_size  # transform sum into mean
    #         mean_d_sim_cos /= step_size

    #         mean_dist_d2 += mean_d_dist
    #         mean_sim_d2_cos += mean_d_sim_cos
            
    #     mean_dist_d2 /= len(eval_dataset_d2defs)
    #     mean_sim_d2_cos /= len(eval_dataset_d2defs)
    #     mean_grad /= n_datapoints
        
    #     logger.info(f"Mean distance between {keys[2]} grads and their corresponding definitions: {mean_dist_d1}")
    #     logger.info(f"Mean distance between {keys[3]} grads and their corresponding definitions: {mean_dist_d2}")
    #     logger.info(f"Mean cosine similarity between {keys[2]} grads and their corresponding definitions: {mean_sim_d1_cos}")
    #     logger.info(f"Mean cosine similarity between {keys[3]} grads and their corresponding definitions: {mean_sim_d2_cos}")
        
    #     # Calculate variance
    #     logger.info('*** Computing gradient variance ***')            
    #     l2_dist = 0
    #     cos_sim = 0
    #     for eval_dataset_input in [eval_dataset_d1cons, eval_dataset_d2cons, eval_dataset_d1defs, eval_dataset_d2defs]:
    #         for example in tqdm(eval_dataset_input):
    #             grad = get_gradient(model, example)
    #             l2_dist += torch.sum((grad - mean_grad)**2)
    #             cos_sim += torch.cosine_similarity(grad, mean_grad, dim=0).item()
    #             # l2_dist.add_(torch.sum((grad - mean_grad)**2))
    #             # cos_sim.add_(torch.cosine_similarity(grad, mean_grad, dim=0))
                
    #     l2_dist /= n_datapoints
    #     cos_sim /= n_datapoints
    #     variance = l2_dist.item()
    #     logger.info(f"Gradient variance: {variance}")
    #     logger.info(f"Gradient cosine similarity: {cos_sim}")
        
    #     del eval_dataset_d1cons, eval_dataset_d2cons, eval_dataset_d1defs, eval_dataset_d2defs
        
    #     self.tb_writer.add_scalar("eval/grad_mean_dist_d1", mean_dist_d1, state.global_step)
    #     self.tb_writer.add_scalar("eval/grad_mean_dist_d2", mean_dist_d2, state.global_step)
    #     self.tb_writer.add_scalar("eval/grad_variance", variance, state.global_step)
    #     self.tb_writer.add_scalar("eval/grad_cosine_similarity", cos_sim, state.global_step)
    #     self.tb_writer.add_scalar("eval/grad_mean_sim_d1_cos", mean_sim_d1_cos, state.global_step)
    #     self.tb_writer.add_scalar("eval/grad_mean_sim_d2_cos", mean_sim_d2_cos, state.global_step)
        
    #     wandb.log({f"eval/grad_mean_dist_d1": mean_dist_d1}, state.global_step)
    #     wandb.log({f"eval/grad_mean_dist_d2": mean_dist_d2}, state.global_step)
    #     wandb.log({f"eval/grad_variance": variance}, state.global_step)
    #     wandb.log({f"eval/grad_cosine_similarity": cos_sim}, state.global_step)
    #     wandb.log({f"eval/grad_mean_sim_d1_cos": mean_sim_d1_cos}, state.global_step)
    #     wandb.log({f"eval/grad_mean_sim_d2_cos": mean_sim_d2_cos}, state.global_step)
        
        
    def evaluate_fn(self, args, state, model, tokenizer):
        """Compute gradient distance between definitions and corresponding questions."""
        def compute_mean_distance(eval_dataset_questions, eval_dataset_defs, tag, mean_grad=None):
            # assuming eval_dataset_questions and eval_dataset_defs are already tokenized, on device and sorted
            step_size = len(eval_dataset_questions) // len(eval_dataset_defs)  # number of questions per definition
            
            mean_dist = 0
            mean_sim_cos = 0
            
            mean_l1_d_norm, mean_l1_q_norm = 0, 0
            mean_l2_d_norm, mean_l2_q_norm = 0, 0
            mean_linf_d_norm, mean_linf_q_norm = 0, 0
            
            for i in tqdm(range(len(eval_dataset_defs))):
                d = eval_dataset_defs[i]
                d_grad = get_gradient(model, d)
                mean_d_dist = 0
                mean_d_sim_cos = 0
                
                # update mean_grad
                if mean_grad is None:
                    mean_grad = d_grad
                else:
                    mean_grad += d_grad
                
                # update gradient norms (definitions)
                mean_l1_d_norm += torch.norm(d_grad, p=1).item()
                mean_l2_d_norm += torch.norm(d_grad, p=2).item()
                mean_linf_d_norm += torch.norm(d_grad, p=float('inf')).item()
                    
                for j in range(step_size):
                    n = i * step_size + j  # index of question
                    q = eval_dataset_questions[n]
                    q_grad = get_gradient(model, q)
                    # update distance and cosine similarity using current question
                    mean_d_dist += torch.sqrt(torch.sum((d_grad - q_grad)**2)) # l2 distance between gradient of definition and gradient of question
                    mean_d_sim_cos += torch.nn.functional.cosine_similarity(d_grad, q_grad, dim=0)
                    # update gradient norms (questions)
                    mean_l1_q_norm += torch.norm(q_grad, p=1).item()
                    mean_l2_q_norm += torch.norm(q_grad, p=2).item()
                    mean_linf_q_norm += torch.norm(q_grad, p=float('inf')).item()
                    mean_grad += q_grad
                    
                mean_d_dist /= step_size  # transform sum into mean distance for this definition
                mean_d_sim_cos /= step_size
                
                mean_dist += mean_d_dist # mean distance for all definitions
                mean_sim_cos += mean_d_sim_cos
                
            mean_dist /= len(eval_dataset_defs)
            mean_sim_cos /= len(eval_dataset_defs)
            
            # divide by number of questions/definitions to get mean norms
            mean_l1_d_norm /= len(eval_dataset_defs)
            mean_l2_d_norm /= len(eval_dataset_defs)
            mean_linf_d_norm /= len(eval_dataset_defs)
            
            mean_l1_q_norm /= len(eval_dataset_questions)
            mean_l2_q_norm /= len(eval_dataset_questions)
            mean_linf_q_norm /= len(eval_dataset_questions)
            
            return mean_dist, mean_sim_cos, mean_grad, {f'grad_mean_l1_q_norm_{tag}': mean_l1_q_norm, f'grad_mean_l2_q_norm_{tag}': mean_l2_q_norm,
                                                        f'grad_mean_linf_norm_{tag}': mean_linf_q_norm, f'grad_mean_l1_d_norm_{tag}': mean_l1_d_norm,
                                                        f'grad_mean_l2_d_norm_{tag}': mean_l2_d_norm, f'grad_mean_linf_d_norm_{tag}': mean_linf_d_norm}


        if self.tb_writer is None:
            self._init_summary_writer(args)
            
        model.train()
        # keys = ['train_defs_d1consis', 'train_defs_d2consis', 'd1consis', 'd2consis']
        keys = ['train_defs_qd1consis', 'train_defs_qd2incons', 'train_questions_qd1consis', 'train_questions_qd2incons']
        tag1 = keys[0].split('_')[-1]
        tag2 = keys[1].split('_')[-1]
        
        self.eval_dataset_tokenized = {key: self.eval_dataset_tokenized[key] for key in keys}
        n_datapoints = sum([len(self.eval_dataset_tokenized[key]) for key in self.eval_dataset_tokenized])  # number of datapoints

        logger.info('*** Computing gradient distance between definitions and corresponding questions ***')    
        
        eval_dataset_d1cons = self.eval_dataset_tokenized[keys[2]].with_format('torch', device='cuda')
        eval_dataset_d1defs = self.eval_dataset_tokenized[keys[0]].with_format('torch', device='cuda')
        mean_dist_d1, mean_sim_d1_cos, mean_grad, norms1 = compute_mean_distance(eval_dataset_d1cons, eval_dataset_d1defs, tag=tag1, mean_grad=None)
        
        eval_dataset_d2cons = self.eval_dataset_tokenized[keys[3]].with_format('torch', device='cuda')
        eval_dataset_d2defs = self.eval_dataset_tokenized[keys[1]].with_format('torch', device='cuda')
        mean_dist_d2, mean_sim_d2_cos, mean_grad, norms2 = compute_mean_distance(eval_dataset_d2cons, eval_dataset_d2defs, tag=tag2, mean_grad=mean_grad)
        
        mean_grad /= n_datapoints
        
        logger.info(f"Mean distance between {keys[2]} grads and their corresponding definitions: {mean_dist_d1}")
        logger.info(f"Mean distance between {keys[3]} grads and their corresponding definitions: {mean_dist_d2}")
        logger.info(f"Mean cosine similarity between {keys[2]} grads and their corresponding definitions: {mean_sim_d1_cos}")
        logger.info(f"Mean cosine similarity between {keys[3]} grads and their corresponding definitions: {mean_sim_d2_cos}")
        
        # Calculate variance
        logger.info('*** Computing gradient variance ***')            
        l2_dist = 0
        cos_sim = 0
        for eval_dataset_input in [eval_dataset_d1cons, eval_dataset_d2cons, eval_dataset_d1defs, eval_dataset_d2defs]:
            for example in tqdm(eval_dataset_input):
                grad = get_gradient(model, example)
                l2_dist += torch.sum((grad - mean_grad)**2)
                cos_sim += torch.cosine_similarity(grad, mean_grad, dim=0).item()
                
        l2_dist /= n_datapoints
        cos_sim /= n_datapoints
        variance = l2_dist.item()
        
        logger.info(f"Gradient variance: {variance}")
        logger.info(f"Gradient cosine similarity: {cos_sim}")
        
        # delete eval datasets and log metrics
        del eval_dataset_d1cons, eval_dataset_d2cons, eval_dataset_d1defs, eval_dataset_d2defs

        self.tb_writer.add_scalar(f"eval/grad_mean_dist_{tag1}", mean_dist_d1, state.global_step)
        self.tb_writer.add_scalar(f"eval/grad_mean_dist_{tag2}", mean_dist_d2, state.global_step)
        self.tb_writer.add_scalar("eval/grad_variance", variance, state.global_step)
        self.tb_writer.add_scalar("eval/grad_cosine_similarity", cos_sim, state.global_step)
        self.tb_writer.add_scalar(f"eval/grad_mean_sim_{tag1}_cos", mean_sim_d1_cos, state.global_step)
        self.tb_writer.add_scalar(f"eval/grad_mean_sim_{tag2}_cos", mean_sim_d2_cos, state.global_step)
        
        norms1.update(norms2)
        for norm in norms1:
            self.tb_writer.add_scalar(f"eval/{norm}", norms1[norm], state.global_step)
        
        
        # wandb.log({f"eval/grad_mean_dist_d1": mean_dist_d1}, state.global_step)
        # wandb.log({f"eval/grad_mean_dist_d2": mean_dist_d2}, state.global_step)
        # wandb.log({f"eval/grad_variance": variance}, state.global_step)
        # wandb.log({f"eval/grad_cosine_similarity": cos_sim}, state.global_step)
        # wandb.log({f"eval/grad_mean_sim_d1_cos": mean_sim_d1_cos}, state.global_step)
        # wandb.log({f"eval/grad_mean_sim_d2_cos": mean_sim_d2_cos}, state.global_step)
        # wandb.log({f"eval/grad_mean_l1_norm": mean_l1_norm}, state.global_step)
        # wandb.log({f"eval/grad_mean_l2_norm": mean_l2_norm}, state.global_step)
        # wandb.log({f"eval/grad_mean_linf_norm": mean_linf_norm}, state.global_step)
        

def get_gradient(model, input_dict):
    # assume batch_datapoints is already tokenized
    """Get the gradients of the model parameters."""
    # move all tensors from input_dict to cuda
    input_dict = {name: input_dict[name].unsqueeze(0) for name in input_dict if name in ['input_ids', 'attention_mask', 'labels']}
    model.zero_grad()
    outputs = model(**input_dict)
    loss = outputs.loss
    loss.backward()

    # gradients = {name: param.grad.cpu().detach() for name, param in model.named_parameters()}
    
    # grad = []
    # for name in sorted(gradients.keys()):
    #     grad.append(gradients[name].view(-1))
    # grad = torch.cat(grad)
    
    grad = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad.append(param.grad.view(-1))
            
    grad = torch.cat(grad)
    return grad


