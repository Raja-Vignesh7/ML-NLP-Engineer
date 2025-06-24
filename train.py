from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from kaggle_secrets import UserSecretsClient
from transformers import AutoTokenizer
import wandb
from datasets import Dataset

user_secrets = UserSecretsClient()
wandb_key = user_secrets.get_secret("wandb")
wandb.login(key=wandb_key)


MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"  # You can change this to any BERT-like model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

HGF_dataset = Dataset.from_pandas(processed_dataset)
HGF_dataset
def tokenize_function(example):
    tokens = tokenizer(example["tokens"], padding="max_length", truncation=True)
    tokens["labels"] = example["target"]  # pass label explicitly
    return tokens

tokenized_dataset = HGF_dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)


model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    max_steps = 1500,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)
trainer.train()