import os
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from kaggle_secrets import UserSecretsClient
import wandb
import torch

# login and access using W&B API key
user_secrets = UserSecretsClient()
wandb_key = user_secrets.get_secret("wandb")
wandb.login(key=wandb_key)

# load Tokenizer
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"  # You can change this to any BERT-like model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# convert Pandas dataframe to Hugging face form Dataset
def tokenize_function(example):
    tokens = tokenizer(example["tokens"], padding="max_length", truncation=True)
    tokens["labels"] = example["target"]  # pass label explicitly
    return tokens

tokenized_dataset = HGF_dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

# load Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


# Sample Prediction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text = "This game is amazing!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():  # Optional: saves memory during inference
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax().item()
    
print("Predicted:", predicted_class)



# Set a directory to save your model and tokenizer
save_directory = "/kaggle/working/game_review_bert_model"
os.makedirs(save_directory,exist_ok=True)
# Save model
model.save_pretrained(save_directory)

# Save tokenizer
tokenizer.save_pretrained(save_directory)
