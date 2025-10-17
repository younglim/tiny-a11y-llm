from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# Load dataset from Hugging Face
dataset = load_dataset("younglim/a11y-dataset", split="train")

# Load base model
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# LoRA config for lightweight fine-tuning
peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"])
model = get_peft_model(model, peft_config)

# Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(tokenize, batched=True)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=1e-4,
    fp16=True,
    push_to_hub=True,
    hub_model_id="younglim/tiny-a11y-model",
)

# Trainer
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
trainer.push_to_hub()
