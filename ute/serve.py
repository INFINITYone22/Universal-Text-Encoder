from __future__ import annotations

import argparse
from typing import List

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import Body
from pydantic import BaseModel

from .inference import build_from_checkpoint
from .utils import Config, amp_dtype_from_precision, get_torch_device


class EmbedRequest(BaseModel):
    texts: List[str]


def create_app(config_path: str, checkpoint_path: str | None = None) -> FastAPI:
    device = get_torch_device()
    cfg = Config.load(config_path)
    precision = str(cfg.get("model.precision", "bf16"))
    amp_dtype = amp_dtype_from_precision(precision)
    tok = None
    model = None
    if checkpoint_path and isinstance(checkpoint_path, str):
        try:
            tok, model = build_from_checkpoint(config_path, checkpoint_path)
            model.to(device)
            model.eval()
        except Exception:
            tok = None
            model = None
    if tok is None:
        # Fallback tokenizer from config (dynamic mapping); no model
        from .tokenizer import CharacterTokenizer

        tok = CharacterTokenizer(vocab_size=int(cfg.get("model.vocab_size", 4096)), space_normalization=True)

    app = FastAPI()

    @app.post("/embed")
    def embed(req: EmbedRequest):  # type: ignore[override]
        if model is None:
            return JSONResponse({"error": "Model not loaded. Provide a valid --checkpoint to enable embeddings."}, status_code=503)
        texts = req.texts
        input_ids_list = [
            tok.encode(t, add_special_tokens=True, max_length=int(cfg.get("model.max_sequence_length"))) for t in texts
        ]
        max_len = max(len(x) for x in input_ids_list)
        input_ids = torch.full((len(texts), max_len), tok.pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(texts), max_len), dtype=torch.long, device=device)
        for i, seq in enumerate(input_ids_list):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            attention_mask[i, : len(seq)] = 1
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            _, pooled = model(input_ids=input_ids, eos_id=tok.eos_id, attention_mask=attention_mask)
        return {"embeddings": pooled.detach().cpu().tolist()}

    @app.get("/tokenize", response_class=HTMLResponse)
    def tokenize_ui():  # type: ignore[override]
        # Simple HTML+JS UI to visualize character-level tokenization
        html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>UTE Tokenizer Visualizer</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
    .container { max-width: 900px; margin: 0 auto; }
    textarea { width: 100%; height: 120px; font-size: 16px; padding: 8px; }
    button { padding: 8px 14px; margin-top: 8px; cursor: pointer; }
    .tokens { margin-top: 16px; line-height: 2.2; display: flex; flex-wrap: wrap; gap: 6px; }
    .tok { padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(0,0,0,0.1); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .meta { color: #555; font-size: 12px; margin-top: 10px; }
  </style>
  <script>
    function colorForId(id) {
      // deterministic hue based on id (golden ratio)
      const hue = (id * 137.508) % 360;
      return `hsl(${hue}, 70%, 85%)`;
    }
    async function doTokenize() {
      const text = document.getElementById('txt').value;
      const res = await fetch('/tokenize_json', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text })
      });
      const data = await res.json();
      const wrap = document.getElementById('tokens');
      wrap.innerHTML = '';
      data.tokens.forEach(t => {
        const span = document.createElement('span');
        span.className = 'tok';
        span.style.backgroundColor = colorForId(t.id);
        span.title = `id=${t.id}`;
        span.textContent = t.text;
        wrap.appendChild(span);
      });
      document.getElementById('meta').textContent = `length=${data.tokens.length}`;
    }
    window.addEventListener('DOMContentLoaded', () => {
      document.getElementById('btn').addEventListener('click', doTokenize);
    });
  </script>
  </head>
  <body>
    <div class="container">
      <h2>UTE Tokenizer Visualizer</h2>
      <p>Enter text below. Tokens are character-level (including special tokens).</p>
      <textarea id="txt" placeholder="Type text here..."></textarea>
      <br/>
      <button id="btn">Tokenize</button>
      <div id="tokens" class="tokens"></div>
      <div id="meta" class="meta"></div>
    </div>
  </body>
</html>
"""
        return HTMLResponse(html)

    @app.post("/tokenize_json")
    def tokenize_json(payload: dict = Body(...)):  # type: ignore[override]
        text = str(payload.get("text", ""))
        ids = tok.encode(text, add_special_tokens=True, max_length=int(cfg.get("model.max_sequence_length")))
        items = []
        for i in ids:
            ch = tok.id_to_char.get(int(i), "")
            items.append({"id": int(i), "text": ch})
        return JSONResponse({"tokens": items})

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn

    uvicorn.run(create_app(args.config, args.checkpoint or None), host=args.host, port=args.port)


