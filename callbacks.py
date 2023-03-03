from transformers import (TrainerCallback, TrainerControl, TrainerState,
                          TrainingArguments, pipeline)
from transformers.integrations import TensorBoardCallback
from transformers.trainer_utils import IntervalStrategy

from logger import setup_logger
from metrics import compute_em_list, compute_f1_list

logger = setup_logger(__name__)

class EvaluationCallback(TensorBoardCallback):
    def __init__(self, eval_dataset_tokenized, generate_batch_fn, postprocess_output_fn, tb_writer=None, numeric_experiment=False, eval_each=False):
        super(EvaluationCallback, self).__init__(tb_writer)
        self.eval_dataset_tokenized = eval_dataset_tokenized
        self.numeric_experiment = numeric_experiment
        self.generate_batch = generate_batch_fn
        self.postprocess_output_fn = postprocess_output_fn
        self.em_score = {}
        self.f1_score = {}
        self.eval_each = eval_each
        
    def evaluate_fn(self, args, state, model, tokenizer):
        if self.tb_writer is None:
            self._init_summary_writer(args)
        # set eval mode
        model.eval()
        for k in self.eval_dataset_tokenized:
            logger.info(f'*** Evaluating on {k} ***')
            eval_dataset_k = self.eval_dataset_tokenized[k]
            # generate predictions using generate_batch_fn function
            eval_dataset_input = eval_dataset_k.remove_columns(['attention_mask', 'labels', 'answer'])
            predictions_k = eval_dataset_input.with_format('torch', device='cuda').map(
                self.generate_batch,
                batched=True,
                load_from_cache_file=True,
                batch_size=args.per_device_eval_batch_size,
                remove_columns=['input_ids', 'input_ids_eval'],
                desc=f"Creating predictions for {k}",
            )
            # decode and aggregate predicted anwers
            predicted_answers = tokenizer.batch_decode(predictions_k['prediction'], skip_special_tokens=True)
            # apply postprocessing to predictions
            predicted_answers = [self.postprocess_output_fn(predicted_answer) for predicted_answer in predicted_answers]
            # decode original answers
            original_answers = eval_dataset_k['answer']
            # apply postprocessing to original answers
            original_answers = [a.replace('\n', '').strip() for a in original_answers]
            
            self.em_score[k] = compute_em_list(predicted_answers, original_answers)
            self.f1_score[k] = compute_f1_list(predicted_answers, original_answers)

            self.tb_writer.add_scalar(f"eval/{k}_EM", self.em_score[k], state.global_step)
            self.tb_writer.add_scalar(f"eval/{k}_F1", self.f1_score[k], state.global_step)
            
            for i in range(10):
                #print(f'Prompt: {qa_prompts[i]}')
                logger.info(f'Correct & predicted answers: {original_answers[i], predicted_answers[i]}\n')

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self.eval_each and state.epoch % self.eval_each == 0:
            self.evaluate_fn(args, state, model, tokenizer)
            
    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if not self.eval_each:
            # there weren't any evaluations during training
            self.evaluate_fn(args, state, model, tokenizer) # updates metrics dict
        

class EvaluationCallbackPipeline(TensorBoardCallback):
    def __init__(self, eval_dataset_raw, tb_writer=None, numeric_experiment=False, eval_each=1):
        super(EvaluationCallbackPipeline, self).__init__(tb_writer)
        self.eval_dataset_raw = eval_dataset_raw
        self.numeric_experiment = numeric_experiment
        self.em_score = {}
        self.f1_score = {}
        self.eval_each = eval_each
        
    def evaluate_fn(self, args, state, model, tokenizer):
        if self.tb_writer is None:
            self._init_summary_writer(args)
        
        model.eval()
        tokenizer.padding_side = 'left'
        pipe = pipeline(task='text-generation', model=model,
                        device=0, tokenizer=tokenizer, top_k=1)
        for k in self.eval_dataset_raw:
            logger.info(f'*** Evaluating on {k} ***')
            eval_dataset_k = self.eval_dataset_raw[k]
            original_answers = eval_dataset_k['answer']
            qa_prompts = eval_dataset_k['question']

            predicted_answers = pipe(qa_prompts,
                                    max_new_tokens=20,
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

            for i in range(10):
                #print(f'Prompt: {qa_prompts[i]}')
                logger.info(f'Correct & predicted answers: {original_answers[i], predicted_answers[i]}\n')
            
    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self.eval_each and state.epoch % self.eval_each == 0:
            self.evaluate_fn(args, state, model, tokenizer)
            
    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if not self.eval_each:
            # there weren't any evaluations during training
            self.evaluate_fn(args, state, model, tokenizer) # updates metrics dict


class CustomSaveCallback(TrainerCallback):
    """Callback for saving each n epochs."""
    def __init__(self, save_each) -> None:
        self.save_each = save_each
        
    def on_epoch_end(self, args: TrainingArguments,
                     state: TrainerState,
                     control: TrainerControl, **kwargs):

        if args.evaluation_strategy == IntervalStrategy.EPOCH and state.epoch % self.save_each == 0:
            control.should_save = True

        return control
