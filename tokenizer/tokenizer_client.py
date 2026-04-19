import subprocess
import json
import os
import sys

class AlphaGeometryTokenizer:
    def __init__(self, vocab_path, server_script='sp_server.py'):
        """
        Client for the AlphaGeometry SentencePiece tokenizer.
        
        Args:
            vocab_path (str): Path to the .model file.
            server_script (str): Path to the sp_server.py script.
        """
        # Resolve the absolute path to the server script relative to this file if not provided as absolute
        if not os.path.isabs(server_script):
            server_script = os.path.join(os.path.dirname(__file__), server_script)
            
        # Use the current python interpreter
        python_bin = sys.executable
        
        self.proc = subprocess.Popen(
            [python_bin, server_script, vocab_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        
        # Initialize and get vocab size
        self.proc.stdin.write(json.dumps({'type': 'vocab_size'}) + '\n')
        self.proc.stdin.flush()
        self.vocab_size = json.loads(self.proc.stdout.readline())
        
    def encode(self, text):
        """Encodes text into a list of token IDs."""
        self.proc.stdin.write(json.dumps({'type': 'encode', 'text': text}) + '\n')
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())
        
    def decode(self, ids):
        """Decodes a list of token IDs back into text."""
        if isinstance(ids, int):
            ids = [ids]
        ids = [int(i) for i in ids]
        self.proc.stdin.write(json.dumps({'type': 'decode', 'ids': ids}) + '\n')
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def __del__(self):
        if hasattr(self, 'proc'):
            self.proc.terminate()

if __name__ == '__main__':
    # Example usage
    model_path = os.path.join('weights', 'geometry.757.model')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
        
    tokenizer = AlphaGeometryTokenizer(model_path)
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    text = "a b c coll a b c ;"
    encoded = tokenizer.encode(text)
    print(f"Encoded '{text}': {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'")
