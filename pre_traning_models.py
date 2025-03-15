import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

MODEL_ID = "gpt2"  # Using GPT-2 Small for local purposes
DATASET_PATH = "seo_dataset.txt"  # Path to the dataset
OUTPUT_DIR = "./gpt2-small-seo-pretrained"  # Directory to save the fine-tuned model
LOGGING_DIR = "./logs"  # Directory for logs
TEST_SPLIT_SIZE = 0.1  # 10% of the dataset for evaluation
MAX_LENGTH = 512  # Maximum token length for input
TRAIN_BATCH_SIZE = 4  # Batch size for training
EVAL_BATCH_SIZE = 4  # Batch size for evaluation
NUM_EPOCHS = 5  # Number of training epochs
SAVE_STEPS = 500  # Save checkpoint every 500 steps
LOGGING_STEPS = 10  # Log every 10 steps
SAVE_TOTAL_LIMIT = 2  # Keep only the last 2 checkpoints

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token as EOS token

# Load dataset
dataset = load_dataset("text", data_files={"train": DATASET_PATH})

# Split the dataset into training and evaluation sets
split_datasets = dataset["train"].train_test_split(test_size=TEST_SPLIT_SIZE)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

tokenized_datasets = split_datasets.map(tokenize_function, batched=True, remove_columns=["text"])

# Load LLM
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# Define training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,  
    eval_strategy="epoch",                     
    per_device_train_batch_size=TRAIN_BATCH_SIZE,             
    per_device_eval_batch_size=EVAL_BATCH_SIZE,              
    num_train_epochs=NUM_EPOCHS,                    
    save_steps=SAVE_STEPS,                        
    logging_dir=LOGGING_DIR,                   
    logging_steps=LOGGING_STEPS,                      
    save_total_limit=SAVE_TOTAL_LIMIT,                     
    fp16=torch.cuda.is_available(),           
)

# Data collator for batching
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)

# Clear GPU memory
torch.cuda.empty_cache()

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Function to generate SEO-optimized content
def generate_seo_content(prompt, max_length=200, temperature=0.7, top_k=50):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Move inputs to GPU
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,  # Enable sampling for more diverse outputs
        pad_token_id=tokenizer.eos_token_id  # Ensure padding token is set
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example text generation
if __name__ == "__main__":
    prompt = "Generate an SEO-optimized product description for an iPhone 15 with relevant keywords"
    print(generate_seo_content(prompt))