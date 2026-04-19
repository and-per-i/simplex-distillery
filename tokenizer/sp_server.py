import sys
import sentencepiece as spm
import json

def main():
    sp = spm.SentencePieceProcessor(model_file=sys.argv[1])
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        if req['type'] == 'encode':
            res = sp.encode(req['text'])
        elif req['type'] == 'decode':
            res = sp.decode(req['ids'])
        elif req['type'] == 'vocab_size':
            res = sp.vocab_size()
        print(json.dumps(res))
        sys.stdout.flush()

if __name__ == '__main__':
    main()
