import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, TrainingArguments, GPTNeoConfig, GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
# from transformers import Trainer
# from my_trainer import Trainer
from ghost_trainer import Trainer
from datasets import load_dataset
import time
from accelerate import Accelerator
import warnings
warnings.filterwarnings("ignore")



start_time = time.time()

config = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
model = GPTNeoForCausalLM(config).from_pretrained("original/checkpoint-5000")#.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

model.resize_token_embeddings(len(tokenizer))


dataset = load_dataset("apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2")
dataset = dataset["train"]
print('!!!!!!!!!!!', dataset.column_names)



def tokenize_function(examples):
    examples["labels"] = examples["input_ids"]
    return examples
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    num_proc=32
)

print('!!!!!!!!!!!', tokenized_dataset.column_names)




training_args = TrainingArguments(
    output_dir="./ours",
    evaluation_strategy="no",
    per_device_train_batch_size=16,
    logging_steps=10,
    save_steps=1000,
    num_train_epochs=1,
    report_to="none",
    logging_dir='./ours/logs',
    save_total_limit=100,
    optim="adamw_torch", 
    fp16=True,
    max_grad_norm=1.0
)






class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accelerator = Accelerator()

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            pin_memory=True
        )


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)


trainer.train()

removed_list = trainer.get_remove_list()
print('FINISHED!!!!!!', time.time() - start_time)
print(removed_list)