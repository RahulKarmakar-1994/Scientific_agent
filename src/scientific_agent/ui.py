import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.scientific_agent.agents import ScientificAgent


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Scientific Agent</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1f2933;
      --muted: #607080;
      --line: #d8dee6;
      --panel: #f7f9fb;
      --accent: #116d6e;
      --accent-dark: #0a4f50;
      --warn: #7a4b00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }
    header {
      border-bottom: 1px solid var(--line);
      padding: 14px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1 { font-size: 18px; margin: 0; font-weight: 650; }
    main {
      max-width: 1040px;
      margin: 0 auto;
      padding: 18px;
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      gap: 18px;
      min-height: calc(100vh - 57px);
    }
    aside {
      border-right: 1px solid var(--line);
      padding-right: 18px;
    }
    label { display: block; font-size: 12px; color: var(--muted); margin: 14px 0 6px; }
    input, select, textarea, button {
      font: inherit;
    }
    input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      background: white;
    }
    .chat {
      display: grid;
      grid-template-rows: minmax(320px, 1fr) auto;
      gap: 14px;
      min-height: 0;
    }
    #messages {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      overflow-y: auto;
      min-height: 420px;
      max-height: calc(100vh - 190px);
    }
    .msg {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 12px;
      margin-bottom: 12px;
      white-space: pre-wrap;
      line-height: 1.45;
    }
    .user { border-left: 4px solid #7c8fa3; }
    .assistant { border-left: 4px solid var(--accent); }
    .meta { color: var(--muted); font-size: 12px; margin-top: 8px; }
    .composer {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 96px;
      gap: 10px;
      align-items: end;
    }
    textarea {
      width: 100%;
      resize: vertical;
      min-height: 70px;
      max-height: 180px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }
    button {
      height: 42px;
      border: 0;
      border-radius: 7px;
      color: white;
      background: var(--accent);
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled { opacity: 0.55; cursor: wait; }
    .hint { color: var(--muted); font-size: 13px; line-height: 1.4; }
    .status { color: var(--warn); font-size: 13px; min-height: 18px; }
    @media (max-width: 780px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); padding: 0 0 16px; }
      #messages { max-height: none; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Scientific Agent</h1>
    <div class="hint">RAG-grounded physics tutor with job tracking</div>
  </header>
  <main>
    <aside>
      <label for="session">Session</label>
      <input id="session" value="physics-ui">
      <label for="provider">Provider</label>
      <select id="provider">
        <option value="ollama">ollama</option>
        <option value="gemini">gemini</option>
        <option value="groq">groq</option>
        <option value="openai">openai</option>
      </select>
      <label for="model">Model</label>
      <input id="model" value="llama3.2:1b">
      <label for="engine">Engine</label>
      <select id="engine">
        <option value="local">local</option>
        <option value="docker">docker</option>
        <option value="external-md">external-md</option>
      </select>
      <label for="prediction">Prediction</label>
      <textarea id="prediction" placeholder="Optional: what do you expect the demo to show?"></textarea>
      <p class="hint">Ask a concept question, then follow up with phrases like "show me the equation" or "demo it with Python".</p>
      <p id="status" class="status"></p>
    </aside>
    <section class="chat">
      <div id="messages"></div>
      <div class="composer">
        <textarea id="prompt" placeholder="Teach me entropy and microstates"></textarea>
        <button id="send">Send</button>
      </div>
    </section>
  </main>
  <script>
    const messages = document.getElementById('messages');
    const prompt = document.getElementById('prompt');
    const send = document.getElementById('send');
    const statusEl = document.getElementById('status');

    function addMessage(role, text, meta = '') {
      const node = document.createElement('div');
      node.className = `msg ${role}`;
      node.textContent = text;
      if (meta) {
        const metaNode = document.createElement('div');
        metaNode.className = 'meta';
        metaNode.textContent = meta;
        node.appendChild(metaNode);
      }
      messages.appendChild(node);
      messages.scrollTop = messages.scrollHeight;
    }

    async function sendPrompt() {
      const text = prompt.value.trim();
      if (!text) return;
      addMessage('user', text);
      prompt.value = '';
      send.disabled = true;
      statusEl.textContent = 'Thinking...';
      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            message: text,
            learner_prediction: document.getElementById('prediction').value.trim() || null,
            session_id: document.getElementById('session').value || 'physics-ui',
            provider: document.getElementById('provider').value,
            model: document.getElementById('model').value || null,
            engine: document.getElementById('engine').value
          })
        });
        document.getElementById('prediction').value = '';
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Request failed');
        const result = data.result || {};
        const answer = result.final_answer || result.answer || JSON.stringify(data, null, 2);
        addMessage('assistant', answer, `job ${data.job_id}`);
        statusEl.textContent = '';
      } catch (error) {
        addMessage('assistant', `Error: ${error.message}`);
        statusEl.textContent = 'Request failed.';
      } finally {
        send.disabled = false;
        prompt.focus();
      }
    }

    send.addEventListener('click', sendPrompt);
    prompt.addEventListener('keydown', event => {
      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        sendPrompt();
      }
    });
    addMessage('assistant', 'Ready. Start with a physics question or a runnable demo request.');
  </script>
</body>
</html>
"""


class ScientificAgentUIHandler(BaseHTTPRequestHandler):
    agent_cache = {}

    def do_HEAD(self):
        if self.path != "/":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        self._send(200, HTML, content_type="text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return

        try:
            payload = self._read_json()
            report = self._agent(payload).run(
                payload.get("message", ""),
                session_id=payload.get("session_id") or "physics-ui",
                learner_prediction=payload.get("learner_prediction"),
            )
            self._send_json(200, report)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _agent(self, payload):
        provider = payload.get("provider") or self.server.provider
        model = payload.get("model") or self.server.model
        engine = payload.get("engine") or self.server.engine
        index_dir = self.server.index_dir
        key = (provider, model, engine, index_dir)
        if key not in self.agent_cache:
            self.agent_cache[key] = ScientificAgent(
                provider=provider,
                model=model,
                engine=engine,
                index_dir=index_dir,
            )
        return self.agent_cache[key]

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def _send_json(self, status, payload):
        self._send(status, json.dumps(payload, indent=2), content_type="application/json")

    def _send(self, status, body, content_type):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        print(f"[ui] {self.address_string()} - {format % args}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Scientific Agent web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--provider", default="ollama", choices=["ollama", "gemini", "groq", "openai"])
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument("--engine", default="local", choices=["local", "docker", "external-md"])
    parser.add_argument("--index-dir", default=".vector_store")
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), ScientificAgentUIHandler)
    server.provider = args.provider
    server.model = args.model
    server.engine = args.engine
    server.index_dir = args.index_dir
    print(f"Scientific Agent UI: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
