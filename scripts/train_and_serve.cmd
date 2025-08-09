@echo off
setlocal
rem Train tiny MLM briefly, then launch server with checkpoint and open UI
pushd "%~dp0.."

if not exist ..\.venv\Scripts\python.exe (
  echo Creating venv and installing requirements...
  py -3 -m venv ..\.venv || goto :eof
  ..\.venv\Scripts\python.exe -m pip install -U pip || goto :eof
  ..\.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :eof
)

echo Training tiny MLM (50 steps)...
..\.venv\Scripts\python.exe -m ute.train_mlm --config configs\tiny.yaml --data data\toy.txt --max_steps 50 --eval_interval 25 --output_dir runs\tiny_mlm || goto :eof

echo Launching server with checkpoint...
start "UTE Server" ..\.venv\Scripts\python.exe -m ute.serve --config configs\tiny.yaml --checkpoint runs\tiny_mlm\checkpoints\last.pt --host 127.0.0.1 --port 8000
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:8000/tokenize

popd
endlocal

