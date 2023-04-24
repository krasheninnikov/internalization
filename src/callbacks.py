from abc import ABC, abstractmethod
from functools import partial

from transformers import (TrainerCallback, TrainerControl, TrainerState,
                          TrainingArguments, pipeline)
from transformers.integrations import TensorBoardCallback
from transformers.trainer_utils import IntervalStrategy

import wandb
from src.metrics import compute_em_list, compute_f1_list
from utils.logger import setup_logger

logger = setup_logger(__name__)


class EvaluationCallbackBase(TensorBoardCallback, ABC):
    def __init__(self, 
                 tb_writer=None, 
                 eval_each_epochs=False, 
                 eval_each_steps=False, 
                 evaluation_strategy='epoch', 
                 numeric_experiment=False):
        super().__init__(tb_writer)
        self.em_score = {}
        self.f1_score = {}
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
            
            # TODO: this is a hack for numeric experiment with custom tokenizer, doesn't work for prestrained models.
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
                #print(f'Prompt: {qa_prompts[i]}')
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
            # TODO: set max_new_tokens depending on config param.
            predicted_answers = pipe(qa_prompts,
                                    max_new_tokens=self.max_new_tokens,
                                    pad_token_id=tokenizer.pad_token_id,
                                    batch_size=args.per_device_eval_batch_size,
                                    clean_up_tokenization_spaces=True,
                                    top_k=1,
                                    return_full_text=False)
            if self.numeric_experiment:
                # TODO why is padding not cleaned up by clean_up_tokenization_spaces?
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
                #print(f'Prompt: {qa_prompts[i]}')
                logger.info(f'Correct & predicted answers: {original_answers[i], predicted_answers[i]}\n')


class CustomSaveCallback(TrainerCallback):
    """Callback for saving each n epochs."""
    def __init__(self, save_each_epochs) -> None:
        self.save_each_epochs = save_each_epochs
        
    def on_epoch_end(self, 
                     args: TrainingArguments,
                     state: TrainerState,
                     control: TrainerControl, **kwargs):

        # if args.evaluation_strategy == IntervalStrategy.EPOCH and round(state.epoch) % self.save_each_epochs == 0:
        if self.save_each_epochs > 0 and round(state.epoch) % self.save_each_epochs == 0:
            control.should_save = True

        return control
