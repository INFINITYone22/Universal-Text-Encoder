@echo off
setlocal
call .venv\Scripts\activate
python -m ute.inference --config configs\tiny.yaml --checkpoint runs\tiny_mlm\checkpoints\last.pt --out embeddings.safetensors --texts "Hello world!" "नमस्ते"
endlocal

