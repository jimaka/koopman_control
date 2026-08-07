#!/usr/bin/env python3
"""Build docs/仿真到实船优化指南.pdf with scalable CJK flowchart images."""
from __future__ import annotations

import base64
import html as htmlmod
import json
import re
import subprocess
import textwrap
import time
import urllib.request
from pathlib import Path

from markdown_it import MarkdownIt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MD = DOCS / "仿真到实船优化指南.md"
OUT_DIR = DOCS / "assets/sim2real/mermaid_pdf"
HTML_OUT = DOCS / "仿真到实船优化指南.html"
PDF_OUT = DOCS / "仿真到实船优化指南.pdf"
TMP = Path("/tmp/sim2real_pdf")

NOTO_TTC = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
CHROME = "/usr/bin/google-chrome"
CDP_PORT = 9333
DPI_SCALE = 3


def crop_whitespace(png: Path, pad: int = 20) -> None:
    im = Image.open(png).convert("RGB")
    w, h = im.size
    px = im.load()
    minx, miny, maxx, maxy = w, h, 0, 0
    found = False
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            if not (r > 248 and g > 248 and b > 248):
                found = True
                minx = min(minx, x)
                miny = min(miny, y)
                maxx = max(maxx, x)
                maxy = max(maxy, y)
    if not found:
        return
    box = (
        max(0, minx - pad),
        max(0, miny - pad),
        min(w, maxx + 1 + pad),
        min(h, maxy + 1 + pad),
    )
    im.crop(box).save(png, optimize=True)


def mermaid_page_html(code: str) -> str:
    code = textwrap.dedent(code).strip()
    font_uri = NOTO_TTC.as_uri()
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
@font-face {{
  font-family: "Noto Sans CJK SC";
  src: url("{font_uri}") format("truetype");
  font-weight: normal;
  font-style: normal;
}}
html, body {{
  margin: 0;
  padding: 24px;
  background: #fff;
  font-family: "Noto Sans CJK SC", "Noto Sans CJK JP", sans-serif;
}}
#stage {{ display: inline-block; background: #fff; }}
.mermaid svg {{ max-width: none !important; }}
</style>
</head><body>
<div id="stage"><pre class="mermaid">{htmlmod.escape(code)}</pre></div>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  try {{
    mermaid.initialize({{
      startOnLoad: false,
      theme: 'default',
      securityLevel: 'loose',
      fontFamily: '"Noto Sans CJK SC", "Noto Sans CJK JP", sans-serif',
      flowchart: {{
        htmlLabels: true,
        curve: 'basis',
        useMaxWidth: false,
        padding: 12,
      }},
    }});
    await mermaid.run({{ querySelector: '.mermaid' }});
    await document.fonts.ready;
    await new Promise(r => setTimeout(r, 200));
    const stage = document.getElementById('stage');
    const r = stage.getBoundingClientRect();
    window.__CLIP__ = {{
      x: r.x, y: r.y, width: Math.max(r.width, 1), height: Math.max(r.height, 1),
    }};
    window.__READY__ = true;
  }} catch (e) {{
    window.__READY__ = false;
    window.__ERR__ = String(e);
  }}
</script>
</body></html>"""


def start_cdp():
    subprocess.run(["pkill", "-f", f"--remote-debugging-port={CDP_PORT}"], capture_output=True)
    time.sleep(0.3)
    proc = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--remote-debugging-port={CDP_PORT}",
            "--remote-allow-origins=*",
            "--font-render-hinting=none",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)
    import websocket

    targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/list"))
    page = next(t for t in targets if t.get("type") == "page")
    ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=90)
    msg_id = 0

    def cdp(method, params=None, timeout=90):
        nonlocal msg_id
        msg_id += 1
        mid = msg_id
        ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        end = time.time() + timeout
        while time.time() < end:
            data = json.loads(ws.recv())
            if data.get("id") == mid:
                if "error" in data:
                    raise RuntimeError(f"{method}: {data['error']}")
                return data.get("result", {})
        raise TimeoutError(method)

    cdp("Page.enable")
    cdp("Runtime.enable")
    return proc, ws, cdp


def wait_ready(cdp):
    for _ in range(300):
        if cdp("Runtime.evaluate", {
            "expression": 'typeof window.__READY__ !== "undefined"',
            "returnByValue": True,
        })["result"]["value"]:
            return
        time.sleep(0.08)
    raise TimeoutError("mermaid render timeout")


def render_png(idx: int, code: str, cdp) -> Path:
    page_path = TMP / f"render_{idx:02d}.html"
    out_png = OUT_DIR / f"mermaid_{idx:02d}.png"
    page_path.write_text(mermaid_page_html(code), encoding="utf-8")

    cdp("Emulation.setDeviceMetricsOverride", {
        "width": 1800,
        "height": 1400,
        "deviceScaleFactor": DPI_SCALE,
        "mobile": False,
    })
    cdp("Page.navigate", {"url": page_path.as_uri()})
    wait_ready(cdp)
    ok = cdp("Runtime.evaluate", {
        "expression": "window.__READY__",
        "returnByValue": True,
    })["result"]["value"]
    if not ok:
        err = cdp("Runtime.evaluate", {
            "expression": 'window.__ERR__ || "unknown"',
            "returnByValue": True,
        })["result"]["value"]
        raise RuntimeError(f"mermaid {idx}: {err}")

    clip = cdp("Runtime.evaluate", {
        "expression": "window.__CLIP__",
        "returnByValue": True,
    })["result"]["value"]
    shot = cdp("Page.captureScreenshot", {
        "format": "png",
        "fromSurface": True,
        "clip": {
            "x": clip["x"],
            "y": clip["y"],
            "width": clip["width"],
            "height": clip["height"],
            "scale": DPI_SCALE,
        },
    })
    out_png.write_bytes(base64.b64decode(shot["data"]))
    crop_whitespace(out_png)
    return out_png


def build_html(md_text: str, n_blocks: int) -> str:
    replaced = md_text
    for i in range(n_blocks):
        fence = re.search(r"```mermaid\n(.*?)```", replaced, flags=re.S)
        if not fence:
            break
        rel = f"assets/sim2real/mermaid_pdf/mermaid_{i:02d}.png"
        img = f'\n<figure class="diagram"><img class="diagram-img" src="{rel}" alt="流程图{i+1}"/></figure>\n'
        replaced = replaced[: fence.start()] + img + replaced[fence.end() :]

    replaced = replaced.replace("./assets/sim2real/pipeline.png", "assets/sim2real/pipeline.png")
    replaced = replaced.replace("./assets/sim2real/roadmap.png", "assets/sim2real/roadmap.png")
    replaced = re.sub(
        r"!\[([^\]]*)\]\((assets/sim2real/(?:pipeline|roadmap)\.png)\)",
        r'<figure class="diagram"><img class="diagram-img" src="\2" alt="\1"/></figure>',
        replaced,
    )

    body = (
        MarkdownIt("commonmark", {"html": True})
        .enable("table")
        .enable("strikethrough")
        .render(replaced)
        .replace("[ ]", "☐")
        .replace("[x]", "☑")
        .replace("[X]", "☑")
    )

    css = r"""
@page { size: A4; margin: 16mm 14mm; @bottom-center { content: counter(page); font-size: 9pt; color:#666; } }
html { font-family: "Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans", sans-serif; }
body { font-size: 10.5pt; line-height: 1.55; color:#1a1a1a; }
h1 { font-size: 20pt; border-bottom: 2px solid #2c3e50; padding-bottom: .3em; page-break-after: avoid; }
h2 { font-size: 14pt; color:#1a5276; margin-top: 1.3em; page-break-after: avoid; }
h3 { font-size: 12pt; color:#2874a6; page-break-after: avoid; }
a { color:#1a5276; text-decoration:none; }
code { font-family: "DejaVu Sans Mono", monospace; font-size:.9em; background:#f4f6f7; padding:.05em .25em; border-radius:3px; }
pre { background:#f4f6f7; border:1px solid #d5d8dc; border-radius:4px; padding:.7em .9em; font-size:8.5pt; white-space:pre-wrap; word-break:break-word; page-break-inside:avoid; }
pre code { background:none; padding:0; }
blockquote { border-left:4px solid #5dade2; margin:1em 0; padding:.4em .9em; background:#ebf5fb; }
table { border-collapse:collapse; width:100%; margin:.8em 0 1.1em; font-size:9pt; page-break-inside:avoid; }
th, td { border:1px solid #bdc3c7; padding:.35em .5em; vertical-align:top; }
th { background:#d6eaf8; text-align:left; }
tr:nth-child(even) td { background:#f8f9f9; }
figure.diagram {
  margin: 0.7em 0 1.0em;
  padding: 0.35em;
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
  background: #fff;
  border: 1px solid #e5e8e8;
  border-radius: 4px;
  page-break-inside: avoid;
  text-align: center;
}
img.diagram-img {
  display: block !important;
  width: 100% !important;
  max-width: 100% !important;
  height: auto !important;
  margin: 0 auto !important;
}
hr { border:none; border-top:1px solid #cacfd2; margin:1.4em 0; }
ul, ol { padding-left:1.4em; }
"""
    return (
        f'<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8"/>\n'
        f'<title>仿真到实船优化指南</title>\n<style>\n{css}\n</style>\n</head>\n'
        f"<body>\n{body}\n</body>\n</html>\n"
    )


def main() -> None:
    if not NOTO_TTC.exists():
        raise SystemExit(f"Missing font: {NOTO_TTC}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    md_text = MD.read_text(encoding="utf-8")
    blocks = re.findall(r"```mermaid\n(.*?)```", md_text, flags=re.S)
    print(f"mermaid blocks: {len(blocks)}")

    proc, ws, cdp = start_cdp()
    try:
        for i, code in enumerate(blocks):
            png = render_png(i, code, cdp)
            print(f"  {i:02d}: {Image.open(png).size} ({png.stat().st_size} bytes)")
    finally:
        try:
            ws.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

    HTML_OUT.write_text(build_html(md_text, len(blocks)), encoding="utf-8")
    print("wrote", HTML_OUT)

    r = subprocess.run(
        [
            "prince",
            str(HTML_OUT),
            "-o",
            str(PDF_OUT),
            f"--baseurl={DOCS.resolve().as_uri()}/",
            "--pdf-title=仿真到实船优化指南",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise SystemExit(f"prince failed: {r.stderr}")
    print("wrote", PDF_OUT, PDF_OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
