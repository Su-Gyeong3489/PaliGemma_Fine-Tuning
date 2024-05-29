from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch


#Load the dataset
from datasets import load_dataset
#ds = load_dataset('HuggingFaceM4/VQAv2', split="train") 
ds = load_dataset('HuggingFaceM4/VQAv2', split="train", trust_remote_code=True)
cols_remove = ["question_type", "answers", "answer_type", "image_id", "question_id"]
ds = ds.remove_columns(cols_remove)
ds = ds.train_test_split(test_size=0.1)
train_ds = ds["train"]
val_ds = ds["test"]


#Load the processor, which contains the image processing and tokenization part, and preprocess our dataset.
from transformers import PaliGemmaProcessor, GemmaTokenizer, SiglipImageProcessor
model_id = "google/paligemma-3b-pt-224"
processor = PaliGemmaProcessor(
    image_processor=SiglipImageProcessor(image_seq_length=224),
    tokenizer=GemmaTokenizer.from_pretrained(model_id)
)


#Create a prompt template to condition PaliGemma to answer visual questions.
device = "cuda"

image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
def collate_fn(examples):
  texts = ["answer " + example["question"] + "\n" + example['multiple_choice_answer'] for example in examples]
  images = [example["image"].convert("RGB") for example in examples]
  tokens = processor(text=texts, images=images,
                    return_tensors="pt", padding="longest",
                    tokenize_newline_separately=False)
  labels = tokens["input_ids"].clone()
  labels[labels == processor.tokenizer.pad_token_id] = -100
  labels[labels == image_token] = -100
  tokens["labels"] = labels
  tokens = tokens.to(torch.bfloat16).to(device)
  return tokens


#Load model in 4-bit for QLoRA
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

bnb_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_type=torch.bfloat16
)

lora_config = LoraConfig(
	r=8, 
	target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
	task_type="CAUSAL_LM",
)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


#Or load the model directly.
#model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

#for param in model.vision_tower.parameters():
#    param.requires_grad = False

#for param in model.multi_modal_projector.parameters():
#    param.requires_grad = True


#QLoRA fine-tuning
#Initialize the Trainer and TrainingArguments
##Name "output_dir"
from transformers import TrainingArguments, Trainer
args=TrainingArguments(
            num_train_epochs=2,
            remove_unused_columns=False,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=2e-5,
            weight_decay=1e-6,
            adam_beta2=0.999,
            logging_steps=100,
            optim="paged_adamw_8bit",
            save_strategy="steps",
            save_steps=1000,
            push_to_hub=True,
            save_total_limit=1,
            bf16=True,
            report_to=["tensorboard"],
            dataloader_pin_memory=False,
            output_dir="results"
        )        


#Initialize Trainer, pass in the datasets, data collating function and training arguments, and call train() to start training.
trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        args=args
        )
trainer.train()
