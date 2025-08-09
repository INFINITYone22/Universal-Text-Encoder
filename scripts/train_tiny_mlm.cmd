@echo off
setlocal
py -3 -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
python -m ute.train_mlm --config configs\tiny.yaml --data data\toy.txt --max_steps 50 --eval_interval 25 --output_dir runs\tiny_mlm
endlocal

