import os
import subprocess
import sys
import time
import argparse

# Colori ANSI per logging espressivo
class Colors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

def log_header(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE} 🔷 {title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def log_step(step_num, description):
    print(f"{Colors.BOLD}{Colors.YELLOW}Step {step_num}:{Colors.END} {description}")

def log_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def log_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def run_command(command, description):
    log_step("RUN", f"Esecuzione: {' '.join(command)}")
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Streaming dell'output per logging espressivo
        for line in process.stdout:
            print(f"  {Colors.BLUE}│{Colors.END} {line.strip()}")
            
        process.wait()
        
        if process.returncode != 0:
            log_error(f"Errore durante: {description}")
            sys.exit(1)
            
        duration = time.time() - start_time
        log_success(f"{description} completato in {duration:.2f}s")
        
    except Exception as e:
        log_error(f"Fallimento catastrofico: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="🔷 Simplex Distillery: Master Pipeline")
    parser.add_argument("--teacher_path", type=str, required=True, help="Path al modello AlphaGeometry Maestro")
    parser.add_argument("--data_path", type=str, required=True, help="Path al dataset (.parquet o .txt)")
    parser.add_argument("--output_dir", type=str, default="runs/final_distillation")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32) # Ottimizzato per 32GB VRAM
    
    args = parser.parse_args()
    
    log_header("SIMPLEX DISTILLERY: MASTER PLAN PIPELINE (OPTIMIZED FOR 5090 Ti + EPYC)")
    
    # --- FASE 1: LA FORGIA ---
    log_header("FASE 1: LA FORGIA (SVD + PERTURBED IDENTITY)")
    forged_weights = "studente_inizializzato.pt"
    run_command([
        sys.executable, "scripts/forge_student.py",
        "--teacher_path", args.teacher_path,
        "--output_path", forged_weights,
        "--hidden_size", "384"
    ], "Compressione SVD e inizializzazione Simpliciale")

    # --- FASE 2 & 3: DISTILLAZIONE E PRUNING ---
    log_header("FASE 2 & 3: DISTILLAZIONE (BF16 + TF32 + TORCH.COMPILE)")
    run_command([
        sys.executable, "train.py",
        "--teacher", args.teacher_path,
        "--data_path", args.data_path,
        "--forged_path", forged_weights,
        "--num_layers", "12",
        "--use_simplex_attention", "True",
        "--num_train_epochs", str(args.epochs),
        "--per_device_train_batch_size", str(args.batch_size),
        "--output_dir", args.output_dir,
        "--alpha", "0.5",
        "--temperature", "4.0"
    ], "Training High-Performance (KD + Pruning + BF16)")

    # --- FASE 4: ESTRAZIONE FISICA ---
    log_header("FASE 4: ESTRAZIONE FISICA (MODELLO 2-SIMPLEX)")
    trained_model_path = os.path.join(args.output_dir, "pytorch_model.bin")
    # Nota: KDTrainer salva come pytorch_model.bin se non diversamente specificato
    final_model_name = "Davide_2Simplex_40M_Finale.pt"
    
    run_command([
        sys.executable, "scripts/extract_final_model.py",
        "--input_path", trained_model_path,
        "--output_path", final_model_name
    ], "Estrazione chirurgica dei layer superstiti")

    log_header("PIPELINE COMPLETATA CON SUCCESSO")
    print(f"{Colors.BOLD}{Colors.GREEN}Il tuo modello definitivo è pronto: {final_model_name}{Colors.END}")
    print(f"{Colors.YELLOW}Topologia: 6 Layer (3 Standard + 3 Simpliciali){Colors.END}")
    print(f"{Colors.YELLOW}Training: Distillato da Maestro 151M a Studente 40M{Colors.END}\n")

if __name__ == "__main__":
    main()
