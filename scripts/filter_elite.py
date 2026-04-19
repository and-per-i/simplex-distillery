import polars as pl
import argparse
import os
import glob

def filter_dataset(input_path, output_path, min_logical_steps=15, min_aux_constructions=2):
    """
    Filters a large parquet dataset using Polars streaming (sink_parquet).
    
    Args:
        input_path: Path to the input parquet file or glob pattern (e.g., 'data/*.parquet').
        output_path: Path where the filtered parquet file will be saved.
        min_logical_steps: Minimum value for 'step_logici'.
        min_aux_constructions: Minimum value for 'costruzioni_ausiliarie'.
    """
    print(f"🔍 Inizio scansione dei dati: {input_path}")
    
    # 1. Creazione del piano di esecuzione (LazyFrame)
    # Usiamo scan_parquet per il caricamento pigro
    query = (
        pl.scan_parquet(input_path)
        .filter(
            (pl.col("step_logici") >= min_logical_steps) & 
            (pl.col("costruzioni_ausiliarie") >= min_aux_constructions)
        )
    )
    
    # 2. Esecuzione in streaming (sink_parquet)
    # Questo permette di processare file più grandi della RAM
    print(f"🚀 Esecuzione del filtraggio in streaming verso: {output_path}")
    query.sink_parquet(output_path)
    
    print("✅ Filtraggio completato con successo!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filtra dataset geometrici per trovare i problemi 'Elite'.")
    parser.add_argument("--input", type=str, default="/Users/andrea/Documents/dataset_geo/*.parquet", 
                        help="Path o glob pattern dei file parquet di input")
    parser.add_argument("--output", type=str, default="dataset_elite_filtrato.parquet", 
                        help="Path del file parquet di output")
    parser.add_argument("--steps", type=int, default=15, help="Minimo step logici (default: 15)")
    parser.add_argument("--aux", type=int, default=2, help="Minime costruzioni ausiliarie (default: 2)")
    
    args = parser.parse_args()
    
    # Assicuriamoci che la directory di output esista
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filter_dataset(args.input, args.output, args.steps, args.aux)
