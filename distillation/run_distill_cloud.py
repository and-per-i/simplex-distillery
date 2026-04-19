import os
import torch
import torch.nn as nn
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import wandb
import inspect

from alphageo.model import Decoder
from models.teacher_wrapper import TeacherWrapper
from models.student_model import StudentForCausalLM, StudentConfig
from distillation.kd_trainer import KDTrainer
from tokenizer.hf_tokenizer import load_tokenizer

# NOVITÀ: Attiva TF32 per sfruttare i Tensor Core (fondamentale su 4090/5090)
torch.set_float32_matmul_precision('high')

def get_optimal_config():
    if not torch.cuda.is_available():
        print("⚠️  ATTENZIONE: CUDA NON RILEVATA! Lo script girerà su CPU (molto lento).")
        return {"batch_size": 1, "fp16": False, "compile": False, "name": "CPU"}
    
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    
    if vram_gb > 40:
        batch_size = 128
    elif vram_gb > 20:
        batch_size = 64
    elif vram_gb > 12:
        batch_size = 32
    else:
        batch_size = 16
        
    return {
        "batch_size": batch_size,
        "fp16": True,
        "compile": True,
        "name": gpu_name,
        "vram": vram_gb
    }

def load_teacher_model(ckpt_path, device="cuda", use_compile=True):
    print(f"Loading Teacher from {ckpt_path}...")
    params = torch.load(os.path.join(ckpt_path, "params.sav"), map_location=device, weights_only=False)
    cfg = torch.load(os.path.join(ckpt_path, "cfg.sav"), map_location=device, weights_only=False)
    model = Decoder(cfg)
    model.load_state_dict(params)
    model.to(device)
    model.eval()
    
    if use_compile and torch.cuda.is_available():
        try:
            print(f"Compiling Teacher model with torch.compile...")
            model = torch.compile(model)
        except Exception as e:
            print(f"⚠️  Torch compile bypassato: {e}")
            
    wrapper = TeacherWrapper(model, student_vocab_size=757)
    return wrapper

def main():
    hw_config = get_optimal_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_ckpt = "/app/Newclid_Transformer/pt_ckpt"
    output_dir = f"./distill_results_{hw_config['name'].replace(' ', '_')}"
    
    wandb.init(project="simplex-distillation", name=f"run-{hw_config['name']}-{hw_config['batch_size']}")
    
    teacher = load_teacher_model(teacher_ckpt, device=device, use_compile=hw_config['compile'])
    tokenizer = load_tokenizer("/app/simplex-distillery/tokenizer/weights/geometry.757.model")

    print(f"Initializing Student (2-Simplicial) | Batch Size: {hw_config['batch_size']} | FP16: {hw_config['fp16']}")
    config = StudentConfig(vocab_size=757, hidden_size=512, num_layers=8, num_heads=8, use_triton=True, tie_word_embeddings=False)
    student = StudentForCausalLM(config).to(device)
    
    dataset_path = os.getenv("DATASET_PATH", "/app/simplex-distillery/train0901-00000-of-00003.parquet")
    if os.path.exists(dataset_path):
        raw_dataset = load_dataset("parquet", data_files=dataset_path, split="train")
        print(f"Tokenizing dataset using {os.cpu_count()} processes...")
        def tokenize_function(examples):
            full_text = [q + " " + s for q, s in zip(examples["question"], examples["solution"])]
            tokenized = tokenizer(full_text, truncation=True, max_length=1024, padding="max_length")
            tokenized["labels"] = [list(ids) for ids in tokenized["input_ids"]]
            return tokenized
        dataset = raw_dataset.map(tokenize_function, batched=True, num_proc=os.cpu_count(), remove_columns=raw_dataset.column_names)
        dataset.set_format("torch")
    else:
        dummy_data = {"input_ids": [[1]*128]*100, "labels": [[1]*128]*100}
        dataset = Dataset.from_dict(dummy_data)

    # RILEVAMENTO AUTOMATICO eval_strategy vs evaluation_strategy
    train_args_keys = inspect.signature(TrainingArguments).parameters.keys()
    eval_key = "eval_strategy" if "eval_strategy" in train_args_keys else "evaluation_strategy"

    args_dict = {
        "output_dir": output_dir, 
        "per_device_train_batch_size": hw_config['batch_size'],
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 10, 
        "learning_rate": 1e-4, 
        "warmup_steps": 500, 
        "logging_steps": 10,
        eval_key: "no",
        "save_steps": 1000, 
        "fp16": hw_config['fp16'], 
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
        "report_to": "wandb"
    }

    training_args = TrainingArguments(**args_dict)
    
    trainer = KDTrainer(model=student, teacher_model=teacher, args=training_args, train_dataset=dataset)
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final_student"))

if __name__ == "__main__":
    main()
