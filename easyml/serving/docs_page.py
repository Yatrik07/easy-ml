"""The page at ``GET /``: what the served model expects, with copy-paste examples."""

import html
import json


def request_examples(info, base_url, needs_key=False):
    """curl / PowerShell / Python examples for this model, as plain strings."""
    url = base_url + "/predict"
    kind = (info.get("inputs") or {}).get("kind")
    key_header = ' -H "Authorization: Bearer <your key>"' if needs_key else ""
    ps_key = ' -Headers @{Authorization="Bearer <your key>"}' if needs_key else ""
    py_key = ', headers={"Authorization": "Bearer <your key>"}' if needs_key else ""
    if kind == "image":
        return {
            "curl": f'curl -X POST {url}{key_header} -F "file=@photo.jpg"',
            "PowerShell": (f'Invoke-RestMethod -Method Post -Uri {url}{ps_key} '
                           '-Form @{file = Get-Item "photo.jpg"}   # PowerShell 7+'),
            "Python": ("import requests  # pip install requests\n\n"
                       'with open("photo.jpg", "rb") as image:\n'
                       f'    print(requests.post("{url}", files={{"file": image}}{py_key}).json())'),
        }
    body = info.get("example") or {}
    one_line = json.dumps(body)
    return {
        "curl": f"curl -X POST {url} -H \"Content-Type: application/json\"{key_header} -d '{one_line}'",
        "PowerShell": (f'Invoke-RestMethod -Method Post -Uri {url} -ContentType "application/json"{ps_key} '
                       f"-Body '{one_line}'"),
        "Python": ("import requests  # pip install requests\n\n"
                   f"body = {json.dumps(body, indent=4)}\n"
                   f'print(requests.post("{url}", json=body{py_key}).json())'),
    }


def render_docs(info, base_url, needs_key=False):
    examples = request_examples(info, base_url, needs_key)
    inputs = info.get("inputs") or {}
    if inputs.get("kind") == "features":
        columns = "".join(f"<li><code>{html.escape(str(c))}</code> <span>{html.escape(str(t))}</span></li>"
                          for c, t in (inputs.get("columns") or {}).items())
        what = ('Send one row as <code>{"features": {...}}</code> or several as <code>{"rows": [...]}</code>. '
                "Values can be numbers or text; a missing column counts as a missing value.")
        cleaning = inputs.get("cleaning")
        if cleaning:
            steps = ", ".join(html.escape(str(s)) for s in cleaning.get("steps") or [])
            what = ('Send one row as <code>{"features": {...}}</code> or several as <code>{"rows": [...]}</code>, '
                    "with the original columns below. The model was trained on cleaned data, so the service "
                    f"cleans each request the same way first (recipe <em>{html.escape(str(cleaning.get('recipe')))}</em>: "
                    f"{steps}). Answers include <code>warnings</code> when a request does not look like the "
                    "training data.")
        inputs_html = f"<p>{what}</p><ul class='cols'>{columns}</ul>"
    elif inputs.get("kind") == "text":
        inputs_html = ('<p>Send <code>{"text": "..."}</code> or <code>{"texts": ["...", "..."]}</code>.</p>')
    else:
        inputs_html = ('<p>Send an image as a multipart <code>file</code> (one or several), or JSON '
                       '<code>{"image_base64": "..."}</code>.</p>')
    classes = info.get("classes")
    classes_html = (f"<p>Possible answers: {', '.join('<code>' + html.escape(str(c)) + '</code>' for c in classes)}</p>"
                    if classes else "")
    key_html = ("<p class='note'>This service needs an API key: send <code>Authorization: Bearer &lt;key&gt;</code> "
                "or <code>X-API-Key</code>.</p>" if needs_key else "")
    blocks = "".join(f"<h3>{html.escape(name)}</h3><pre>{html.escape(code)}</pre>" for name, code in examples.items())
    title = html.escape(f"{info.get('model_label')} · run #{info.get('run_id')}")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Easy-ML prediction API</title>
<style>
  :root {{ color-scheme: light dark; --fg: #18181b; --muted: #52525b; --bg: #ffffff; --panel: #f4f4f5; --accent: #2563eb; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --fg: #f4f4f5; --muted: #a1a1aa; --bg: #0b0b0d; --panel: #18181b; --accent: #60a5fa; }} }}
  body {{ margin: 0; font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--fg); background: var(--bg); }}
  main {{ max-width: 820px; margin: 0 auto; padding: 32px 16px 64px; }}
  h1 {{ font-size: 24px; margin: 0 0 4px; }} h2 {{ font-size: 18px; margin-top: 32px; }} h3 {{ font-size: 14px; margin: 20px 0 6px; color: var(--muted); }}
  p, li {{ color: var(--muted); }} code {{ font-family: ui-monospace, Consolas, monospace; font-size: 13px; color: var(--fg); }}
  pre {{ background: var(--panel); padding: 12px 14px; border-radius: 8px; overflow-x: auto; font: 13px/1.5 ui-monospace, Consolas, monospace; white-space: pre-wrap; word-break: break-word; }}
  .cols {{ columns: 2 240px; padding-left: 18px; }} .cols span {{ font-size: 12px; }}
  .note {{ border-left: 3px solid var(--accent); padding-left: 10px; }} a {{ color: var(--accent); }}
</style></head>
<body><main>
<h1>{html.escape(str(info.get('model_label')))} prediction API</h1>
<p>{html.escape(str(info.get('task_label')))} model from Easy-ML Studio run #{info.get('run_id')}
({html.escape(str(info.get('run_name') or ''))}). Served by <code>easy-ml serve</code> {html.escape(str(info.get('easyml_version')))}.</p>
{key_html}
<h2>Endpoints</h2>
<ul><li><code>POST /predict</code>: make predictions</li><li><code>GET /info</code>: this model's details as JSON</li>
<li><code>GET /health</code>: <code>{{"status": "ok"}}</code> while the service is up</li></ul>
<h2>Inputs</h2>
{inputs_html}
{classes_html}
<h2>Examples</h2>
{blocks}
</main></body></html>"""
