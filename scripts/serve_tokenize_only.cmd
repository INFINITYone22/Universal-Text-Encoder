@echo off
setlocal
rem Launch UTE server (no checkpoint) and open tokenizer UI
pushd "%~dp0.."

if not exist ..\.venv\Scripts\python.exe (
  echo Creating venv and installing requirements...
  py -3 -m venv ..\.venv || goto :eof
  ..\.venv\Scripts\python.exe -m pip install -U pip || goto :eof
  ..\.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :eof
)

start "UTE Server" ..\.venv\Scripts\python.exe -m ute.serve --config configs\tiny.yaml --host 127.0.0.1 --port 8000
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:8000/tokenize

popd
endlocal

