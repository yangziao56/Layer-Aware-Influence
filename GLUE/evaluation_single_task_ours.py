import time
import json
import numpy as np
from torch.utils.data import DataLoader
from transformers import GPTNeoForSequenceClassification, GPT2Tokenizer, TrainingArguments
# from transformers import Trainer
from datasets import load_dataset
import evaluate
import argparse
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="tasks")
parser.add_argument('--task', type=str, required=True, help='task name')
parser.add_argument('--match', type=int, default=1, help='mismatch match or match for mnli')
parser.add_argument('--method', type=str, default='ours', help='method name')
args = parser.parse_args()

SAVE_FOLDER = 'GLUE_update_baseline'



if args.method == 'ours':
    from my_trainer import Trainer
else:
    from transformers import Trainer
task_name = args.task



start_time = time.time()


tasks = ['cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']
task_label = {'cola':2, 'sst2':2, 'mrpc':2, 'stsb':1, 'qqp':2, 'mnli':3, 'qnli':2, 'rte':2, 'wnli':2}
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}



model_name = "EleutherAI/gpt-neo-125M"
tokenizer_name = "EleutherAI/gpt-neo-125M" 

# if task_name in ['ax']:
#     model_name = "results/GLUE/mnli_0_model"
#     tokenizer_name = "results/GLUE/mnli_0_model"
# elif task_name in ['mnli']:
#     model_name = f"results/{SAVE_FOLDER}/{task_name}_{args.match}_{args.method}_model" #f"results/GLUE/{task_name}_{args.match}_model"
#     tokenizer_name = f"results/{SAVE_FOLDER}/{task_name}_{args.match}_{args.method}_model" #f"results/GLUE/{task_name}_{args.match}_model"
# else:
#     model_name = f"results/{SAVE_FOLDER}/{task_name}_{args.method}_model" #f"results/GLUE/{task_name}_model"
#     tokenizer_name = f"results/{SAVE_FOLDER}/{task_name}_{args.method}_model" #f"results/GLUE/{task_name}_model"
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token


def preprocess_function(examples, task_name):
    sentence1_key, sentence2_key = task_to_keys[task_name]
    # Tokenize the texts
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding='max_length', max_length=128, truncation=True)
    return result



save_result = {}
print(f": {task_name}")


dataset = load_dataset("glue", task_name)
metric = evaluate.load("glue", task_name)
encoded_dataset = dataset.map(lambda examples: preprocess_function(examples, task_name), batched=True)
print(dataset.keys())
for split in dataset.keys():
    print(f"{split}: {len(dataset[split])}")


model = GPTNeoForSequenceClassification.from_pretrained(model_name, num_labels=task_label[task_name])
model.config.pad_token_id = tokenizer.pad_token_id


if task_name in ['qqp', 'mnli', 'qnli']:
    learning_epoch = 5
else:
    learning_epoch = 5


training_args = TrainingArguments(
    output_dir=f'./results/{SAVE_FOLDER}/{task_name}_{args.method}',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=learning_epoch,
    weight_decay=0.01,
    logging_dir=f'./results/{SAVE_FOLDER}/logs/{task_name}_{args.method}',
)


is_regression = True if task_name == 'stsb' else False
def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result



class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            pin_memory=True
        )





if task_name in ['mnli']:
    if args.match == 0:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset['train'],
            eval_dataset=encoded_dataset['validation_mismatched'],
            compute_metrics=compute_metrics
        )
    else:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset['train'],
            eval_dataset=encoded_dataset['validation_matched'],
            compute_metrics=compute_metrics
        )
    


    trainer.train()

    results = trainer.evaluate()
    print(f"{task_name} evaluation:")
    print(results)
    save_result[task_name + '_val_' + str(args.match) + '_' + args.method] = results


    model.save_pretrained(f"results/{SAVE_FOLDER}/{task_name}_{args.match}_{args.method}_model2")
    tokenizer.save_pretrained(f"results/{SAVE_FOLDER}/{task_name}_{args.match}_{args.method}_model2")

    if args.method == 'ours':
        removed_list = trainer.get_remove_list()
        np.savetxt('results/' + SAVE_FOLDER + '/remove_list_' + task_name + '_' + str(args.match) + '_' + args.method + '.csv', np.array(removed_list), delimiter=',', fmt='%d')
    with open('results/' + SAVE_FOLDER + '/evaluation_GLUE_' + task_name + '_' + str(args.match) + '_' + args.method + '.json', "w", encoding="utf-8") as file:
        json.dump(save_result, file, ensure_ascii=False, indent=4)

    print(task_name, 'finished!', time.time() - start_time)




else:
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['validation'],
        compute_metrics=compute_metrics
        )

    trainer.train()

    results = trainer.evaluate()
    print(f"{task_name} evaluation:")
    print(results)
    save_result[task_name + '_val'] = results


    model.save_pretrained(f"results/{SAVE_FOLDER}/{task_name}_{args.method}_model2")
    tokenizer.save_pretrained(f"results/{SAVE_FOLDER}/{task_name}_{args.method}_model2")

    if args.method == 'ours':
        removed_list = trainer.get_remove_list()
        np.savetxt('results/' + SAVE_FOLDER + '/remove_list_' + task_name + '_' + args.method + '.csv', np.array(removed_list), delimiter=',', fmt='%d')
    with open('results/' + SAVE_FOLDER + '/evaluation_GLUE_' + task_name + '_' + args.method + '.json', "w", encoding="utf-8") as file:
        json.dump(save_result, file, ensure_ascii=False, indent=4)

    print(task_name, 'finished!', time.time() - start_time)