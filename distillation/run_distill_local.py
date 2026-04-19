import os
import sys
import torch
from transformers import TrainingArguments, Trainer, default_data_collator

# Add project roots to path
sys.path.append("/Users/andrea/Documents/simplex-distillery")
sys.path.append("/Users/andrea/Documents/Newclid_Transformer/src")

from models.student_model import StudentForCausalLM
from models.student_config import StudentConfig
from models.teacher_wrapper import TeacherWrapper
from distillation.kd_trainer import KDTrainer
from alphageo.model import Decoder
import pickle

# --- 1. Load Teacher ---
def load_local_teacher(ckpt_path, device="cpu"):
    print(f"Loading Teacher from {ckpt_path}...")
    with open(os.path.join(ckpt_path, "cfg.sav"), "rb") as f:
        cfg = torch.load(f, weights_only=False)
    
    model = Decoder(cfg)
    params = torch.load(os.path.join(ckpt_path, "params.sav"), weights_only=False)
    model.load_state_dict(params)
    model.eval()
    
    # Wrap it for the trainer
    # student_vocab_size=757 matches the AlphaGeometry tokenizer
    wrapper = TeacherWrapper(model, student_vocab_size=757)
    return wrapper

# --- 2. Initialize Student ---
def init_student():
    print("Initializing Student (2-Simplicial Attention)...")
    config = StudentConfig(
        vocab_size=757,
        hidden_size=256,   # Smaller for local test
        num_hidden_layers=4,
        num_attention_heads=8,
        use_simplex_attention=True, # USE SIMPLICIAL ATTENTION!
        w1=8,
        w2=8
    )
    model = StudentForCausalLM(config)
    return model

# --- 3. Mock Dataset (For testing) ---
def get_dummy_dataset(vocab_size=757, seq_len=128, num_samples=10):
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = input_ids.clone()
    
    dataset = []
    for i in range(num_samples):
        dataset.append({
            "input_ids": input_ids[i],
            "labels": labels[i],
            "attention_mask": torch.ones(seq_len, dtype=torch.long)
        })
    return dataset

def main():
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    
    teacher = load_local_teacher("/Users/andrea/Documents/Newclid_Transformer/pt_ckpt", device=device)
    student = init_student().to(device)
    
    train_dataset = get_dummy_dataset()
    
    training_args = TrainingArguments(
        output_dir="./distill_results",
        per_device_train_batch_size=1, # Very small for local test
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        logging_steps=1,
        learning_rate=1e-4,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none"
    )
    
    trainer = KDTrainer(
        model=student,
        teacher_model=teacher,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        temperature=4.0,
        alpha=0.5
    )
    
    print("Starting Distillation...")
    trainer.train()
    print("Distillation Complete!")

if __name__ == "__main__":
    main()
