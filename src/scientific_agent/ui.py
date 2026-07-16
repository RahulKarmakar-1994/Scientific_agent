import argparse
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

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
      --ok: #0f6b3f;
      --bad: #9f1d1d;
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
      line-height: 1.45;
    }
    .msg pre {
      white-space: pre-wrap;
      margin: 0;
      font: inherit;
    }
    .user { border-left: 4px solid #7c8fa3; }
    .assistant { border-left: 4px solid var(--accent); }
    .meta { color: var(--muted); font-size: 12px; margin-top: 8px; }
    .section {
      border-top: 1px solid var(--line);
      padding-top: 10px;
      margin-top: 10px;
    }
    .section:first-child {
      border-top: 0;
      padding-top: 0;
      margin-top: 0;
    }
    .section-title {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      color: var(--muted);
      background: #fbfcfd;
    }
    .chip.ok { color: var(--ok); border-color: #9ed8ba; }
    .chip.warn { color: var(--warn); border-color: #e2c071; }
    .chip.bad { color: var(--bad); border-color: #e5a1a1; }
    .artifacts {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }
    .artifact {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }
    .artifact img {
      display: block;
      width: 100%;
      height: auto;
      background: #fff;
    }
    .artifact a {
      display: block;
      padding: 8px 10px;
      color: var(--accent-dark);
      text-decoration: none;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
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
      <input id="model" value="__MODEL__">
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

    const initialProvider = "__PROVIDER__";
    const initialModel = "__MODEL__";
    const providerDefaults = {
      ollama: "llama3.2:1b",
      gemini: "gemini-2.5-flash",
      groq: "llama-3.3-70b-versatile",
      openai: "gpt-5.2"
    };

    document.getElementById('provider').value = initialProvider;
    document.getElementById('provider').addEventListener('change', event => {
      const model = document.getElementById('model');
      if (!model.value || model.value === initialModel || Object.values(providerDefaults).includes(model.value)) {
        model.value = providerDefaults[event.target.value] || '';
      }
    });

    function addMessage(role, text, meta = '') {
      const node = document.createElement('div');
      node.className = `msg ${role}`;
      const pre = document.createElement('pre');
      pre.textContent = text;
      node.appendChild(pre);
      if (meta) {
        const metaNode = document.createElement('div');
        metaNode.className = 'meta';
        metaNode.textContent = meta;
        node.appendChild(metaNode);
      }
      messages.appendChild(node);
      messages.scrollTop = messages.scrollHeight;
    }

    function section(title, contentNode) {
      const wrap = document.createElement('div');
      wrap.className = 'section';
      const heading = document.createElement('div');
      heading.className = 'section-title';
      heading.textContent = title;
      wrap.appendChild(heading);
      wrap.appendChild(contentNode);
      return wrap;
    }

    function textBlock(text) {
      const pre = document.createElement('pre');
      pre.textContent = text || '';
      return pre;
    }

    function artifactUrl(path) {
      return `/artifact/${encodeURIComponent(path)}`;
    }

    function basename(path) {
      return String(path || '').split('/').pop();
    }

    function renderChips(result) {
      const chips = document.createElement('div');
      chips.className = 'chips';
      const llm = result.llm || {};
      const verification = result.verification || {};
      const grounding = result.grounding || {};
      const items = [
        [`status: ${result.status || 'unknown'}`, result.status === 'completed' || result.status === 'answered' ? 'ok' : result.status === 'unreliable' || result.status === 'rejected' ? 'bad' : 'warn'],
        [`model: ${llm.last_successful_model || llm.model || 'fallback'}`, llm.error && !llm.last_successful_model ? 'bad' : 'ok'],
        [`verification: ${verification.verdict || 'none'}`, verification.verdict === 'pass' ? 'ok' : verification.verdict === 'fail' ? 'bad' : 'warn'],
        [`grounding: ${grounding.status || 'missing'}`, grounding.status === 'grounded' ? 'ok' : 'warn']
      ];
      for (const [label, tone] of items) {
        const chip = document.createElement('span');
        chip.className = `chip ${tone}`;
        chip.textContent = label;
        chips.appendChild(chip);
      }
      return chips;
    }

    function renderArtifacts(paths) {
      const grid = document.createElement('div');
      grid.className = 'artifacts';
      for (const path of paths || []) {
        const item = document.createElement('div');
        item.className = 'artifact';
        const name = basename(path);
        if (/\\.(png|jpg|jpeg|gif)$/i.test(name)) {
          const img = document.createElement('img');
          img.src = artifactUrl(path);
          img.alt = name;
          item.appendChild(img);
        }
        const link = document.createElement('a');
        link.href = artifactUrl(path);
        link.target = '_blank';
        link.rel = 'noreferrer';
        link.textContent = name;
        item.appendChild(link);
        grid.appendChild(item);
      }
      return grid;
    }

    function renderDemoPlan(plan) {
      const lines = [];
      if (plan.learning_goal) lines.push(`Goal: ${plan.learning_goal}`);
      if (plan.simulation_model) lines.push(`Model: ${plan.simulation_model}`);
      if (plan.expected_behavior) lines.push(`Expected: ${plan.expected_behavior}`);
      if ((plan.plots || []).length) {
        lines.push('');
        lines.push('Planned plots:');
        for (const plot of plan.plots) {
          lines.push(`- ${plot.filename || 'plot'}${plot.description ? `: ${plot.description}` : ''}`);
        }
      }
      return textBlock(lines.join('\\n'));
    }

    function addAssistantReport(data) {
      const result = data.result || {};
      const node = document.createElement('div');
      node.className = 'msg assistant';
      node.appendChild(renderChips(result));

      const answer = result.final_answer || result.answer || JSON.stringify(data, null, 2);
      node.appendChild(section('Answer', textBlock(answer)));

      if (result.rich_demo_plan) {
        node.appendChild(section('Demo Plan', renderDemoPlan(result.rich_demo_plan)));
      }

      const artifacts = ((result.job_files || {}).artifacts || []);
      if (artifacts.length) {
        node.appendChild(section('Artifacts', renderArtifacts(artifacts)));
      }

      if ((result.job_files || {}).generated_code) {
        const codeLink = document.createElement('a');
        codeLink.href = artifactUrl(result.job_files.generated_code);
        codeLink.target = '_blank';
        codeLink.rel = 'noreferrer';
        codeLink.textContent = basename(result.job_files.generated_code);
        node.appendChild(section('Generated Code', codeLink));
      }

      const metaNode = document.createElement('div');
      metaNode.className = 'meta';
      metaNode.textContent = `job ${data.job_id}`;
      node.appendChild(metaNode);
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
        addAssistantReport(data);
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
        if self.path.startswith("/artifact/"):
            self._send_artifact(self.path[len("/artifact/") :])
            return
        if self.path != "/":
            self.send_error(404)
            return
        self._send(200, self._html(), content_type="text/html; charset=utf-8")

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

    def _html(self):
        return (
            HTML.replace("__PROVIDER__", self.server.provider)
            .replace("__MODEL__", self.server.model or "")
        )

    def _send_artifact(self, encoded_path):
        raw_path = unquote(encoded_path or "")
        artifact_path = Path(raw_path)
        if not artifact_path.is_absolute():
            artifact_path = Path.cwd() / artifact_path
        artifact_path = artifact_path.resolve()
        outputs_root = (Path.cwd() / "outputs" / "jobs").resolve()
        try:
            artifact_path.relative_to(outputs_root)
        except ValueError:
            self.send_error(403)
            return
        if not artifact_path.exists() or not artifact_path.is_file():
            self.send_error(404)
            return

        suffix = artifact_path.suffix.lower()
        content_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".txt": "text/plain; charset=utf-8",
            ".py": "text/plain; charset=utf-8",
            ".json": "application/json",
        }.get(suffix, "application/octet-stream")
        data = artifact_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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
