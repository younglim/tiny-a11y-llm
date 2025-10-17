from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
dataset = load_from_disk("datasets/wcag_prepared")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"])
model = get_peft_model(model, peft_config)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize, batched=True)

args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=1e-4,
    fp16=True,
    push_to_hub=True,
    hub_model_id="yourname/deepseek-accessibility-coder",
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized)
trainer.train()
trainer.push_to_hub()