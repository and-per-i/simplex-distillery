import os
import torch
import logging
from pathlib import Path
from tokenizer.hf_tokenizer import load_tokenizer
from alphageo.alphageometry import get_lm, run_alphageometry
from newclid import GeometricSolverBuilder
from newclid.jgex.problem_builder import JGEXProblemBuilder
from alphageo.solver import LegacyGeometricSolver

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def evaluate_teacher(ckpt_path="./pt_ckpt", vocab_path="./pt_ckpt/vocab.model", problem_name="orthocenter"):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    logger.info(f"🚀 Evaluazione Teacher Model su {device.upper()}")
    
    # 1. Caricamento Modello e Tokenizer
    try:
        model = get_lm(Path(ckpt_path), device)
        # Usiamo il wrapper HF con vocab_size=1024 per allinearci all'architettura
        tokenizer = load_tokenizer(vocab_path, vocab_size=1024)
        logger.info(f"✅ Modello e Tokenizer (Vocab: {tokenizer.vocab_size}) caricati con successo.")
    except Exception as e:
        logger.error(f"❌ Errore durante il caricamento: {e}")
        return

    # 2. Configurazione Problema (usiamo un esempio standard)
    # Se non abbiamo un file di problemi, creiamo un setup manuale per 'orthocenter'
    # o cerchiamo se esiste un file di default
    problems_file = "newclid/problems/imo.py" # Solo come riferimento
    
    # Per semplicità, usiamo un prompt diretto se JGEXProblemBuilder non trova il file
    # Ma proviamo a usare la logica di alphageo/__main__.py
    
    # Test Problems
    test_problems = {
        "orthocenter": "a b c = triangle a b c; h : orthocenter h a b c; ? perp c h a b",
        "midpoint": "a b c = triangle a b c; d : midpoint d a b; e : midpoint e a c; ? para d e b c"
    }

    if problem_name not in test_problems:
        logger.error(f"Problema {problem_name} non trovato nei test predefiniti.")
        return

    problem_txt = test_problems[problem_name]
    logger.info(f"📝 Problema: {problem_name}")
    logger.info(f"🔍 Prompt: {problem_txt}")

    # 3. Setup Solver
    try:
        from newclid.jgex.problem_builder import JGEXProblemBuilder
        jb = JGEXProblemBuilder(rng=42)
        
        # Carichiamo il problema e alphabetizziamo per il modello
        jb.with_problem_from_txt(problem_txt, problem_name=problem_name)
        from newclid.jgex.formulation import alphabetize
        jb.jgex_problem, _ = alphabetize(jb.jgex_problem)
        
        setup = jb.build()
        
        solver_builder = GeometricSolverBuilder(rng=42)
        if os.path.exists("new_rules.txt"):
            solver_builder.with_rules_from_file(Path("new_rules.txt"))
            
        nc_solver = solver_builder.build(setup)
        
        solver = LegacyGeometricSolver(nc_solver, jb.jgex_problem, jb)
        
        logger.info("🧠 Avvio ricerca della prova (Neuro-Symbolic)...")
        
        out_folder = Path("./results_eval")
        out_folder.mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            success = run_alphageometry(
                solver=solver,
                model=model,
                tokenizer=tokenizer,
                device=device,
                model_beam_width=4,
                model_num_return_sequences=2,
                search_depth=2,
                beam_size=4,
                out_folder=out_folder
            )
            
        if success:
            logger.info("🎉 SUCCESSO! Il modello ha trovato la soluzione.")
            # Stampiamo i passi della prova
            print("\n--- SOLUZIONE TROVATA ---")
            solver.write_solution(None) # Stampa a video se passiamo None in certe versioni, 
                                        # ma LegacyGeometricSolver.write_solution potrebbe volere un Path
        else:
            logger.warning("⚠️ Il modello non ha trovato una soluzione nel tempo limite.")
            
    except Exception as e:
        logger.error(f"❌ Errore durante l'esecuzione del solver: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="orthocenter")
    args = parser.parse_args()
    evaluate_teacher(problem_name=args.problem)
