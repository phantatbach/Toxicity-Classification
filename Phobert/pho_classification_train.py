from transformers import (AutoTokenizer, 
                          RobertaForSequenceClassification,
                          Trainer, 
                          TrainingArguments,)
from transformers import DataCollatorWithPadding
from datasets import load_dataset

model_id = 'vinai/phobert-base-v2'

#visobert
tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          cache_dir='./cache',
                                          model_max_length = 256) 

model = RobertaForSequenceClassification.from_pretrained(model_id,
                                                           num_labels=2,
                                                           cache_dir='./cache')

model.roberta.requires_grad_(False)

# Print the trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {trainable_params}")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
ds = load_dataset('csv', data_files='/home4/bachpt/text_classification/mixed_train_lowered_shuffled.csv', split='train')
ds = ds.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./saved_checkpoints",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    report_to="tensorboard",
    save_strategy="epoch", 
    logging_strategy="steps",
    logging_steps=100, # Log every 10 steps
    save_total_limit=2, # Only last 2 checkpoints will be saved
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# trainer.train(resume_from_checkpoint='./saved_checkpoints')
trainer.train()
