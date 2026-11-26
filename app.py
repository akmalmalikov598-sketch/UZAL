import os
import threading
from flask import Flask, request, jsonify
from uzai import load_model, reply, CKPT_PATH

app = Flask(__name__)
lock = threading.Lock()
model = tok = None
step = 0

if os.path.exists(CKPT_PATH):
    model, tok, step = load_model()


@app.route("/")
def index():
    return HTML


@app.route("/api/status")
def status():
    return jsonify(trained=model is not None, step=step)


@app.route("/api/chat", methods=["POST"])
def chat():
    if model is None:
        return jsonify(error="Model hali o'qitilmagan (uzai.pt topilmadi)."), 503
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()[:500]
    context = (data.get("context") or "")[-800:]
    if not message:
        return jsonify(error="Xabar bo'sh."), 400

    prompt = f"{context}\nFoydalanuvchi: {message}\nUZAI: "
    with lock:
        answer = reply(model, tok, prompt, max_new=300)
    answer = answer.split("\nFoydalanuvchi:")[0].strip()
    return jsonify(reply=answer)


HTML = """<!DOCTYPE html>
<html lang="uz">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UZAI</title>
<style>
  body { margin:0; font-family: system-ui, sans-serif; background:#0f1115; color:#e8e8e8; display:flex; flex-direction:column; height:100vh; }
  header { padding:14px 20px; background:#171a21; border-bottom:1px solid #262b36; display:flex; justify-content:space-between; align-items:center; }
  header h1 { margin:0; font-size:20px; color:#7cc4ff; }
  #status { font-size:13px; color:#8a93a6; }
  #chat { flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:10px; }
  .msg { max-width:75%; padding:10px 14px; border-radius:12px; white-space:pre-wrap; line-height:1.5; }
  .user { align-self:flex-end; background:#2b6cb0; }
  .bot { align-self:flex-start; background:#1e232d; }
  form { display:flex; gap:8px; padding:14px; background:#171a21; border-top:1px solid #262b36; }
  input { flex:1; padding:12px; border-radius:10px; border:1px solid #303746; background:#0f1115; color:#e8e8e8; font-size:15px; }
  button { padding:12px 18px; border:0; border-radius:10px; background:#2b6cb0; color:#fff; font-size:15px; cursor:pointer; }
</style>
</head>
<body>
<header>
  <h1>UZAI</h1>
  <span id="status">holat tekshirilmoqda...</span>
</header>
<div id="chat"></div>
<form id="f">
  <input id="inp" placeholder="Xabar yozing..." autocomplete="off">
  <button type="submit">Yuborish</button>
</form>
<script>
const box = document.getElementById('chat');
const inp = document.getElementById('inp');
let history = '';

function add(who, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + who;
  d.textContent = text;
  box.appendChild(d);
  box.scrollTop = box.scrollHeight;
}

fetch('/api/status').then(r => r.json()).then(j => {
  document.getElementById('status').textContent =
    j.trained ? 'Model tayyor (qadam ' + j.step + ')' : 'Model o\\'qitilmagan';
});

document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const msg = inp.value.trim();
  if (!msg) return;
  inp.value = '';
  add('user', msg);
  const r = await fetch('/api/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: msg, context: history})
  });
  const j = await r.json();
  const ans = j.reply || j.error || '...';
  add('bot', ans);
  history = (history + '\\nFoydalanuvchi: ' + msg + '\\nUZAI: ' + ans).slice(-800);
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
