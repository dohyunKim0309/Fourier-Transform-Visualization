"""
Jupyter 노트북 → 인터넷·파이썬 없이 동작하는 단일 HTML 파일.

- Plotly figure: 첫 figure에 plotly.js inline 임베드, 이후엔 share
- MathJax 3: tex-mml-chtml.js 번들을 <script> 안에 inline
- Markdown: arithmatex 확장으로 $...$, $$...$$ 보존
- 코드 셀: 기본 미표시 (교수님 제출용; 필요하면 --show-code)

Usage:
    python examples/nb_to_offline_html.py examples/05_beyond_nyquist_hilbert.ipynb [--show-code]
"""
from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import sys
from pathlib import Path

import markdown
import plotly.graph_objects as go
import plotly.io as pio


def inline_svg_images(html: str, base_dir: Path) -> str:
    """`<img src="...svg">` 를 SVG inline 으로 치환. base_dir 기준 상대경로 해석."""
    def repl(m):
        src = m.group(1)
        if src.lower().endswith(".svg"):
            svg_path = (base_dir / src).resolve()
            if svg_path.exists():
                svg = svg_path.read_text(encoding="utf-8")
                # <?xml ...?> 헤더 제거 (HTML 안에 두면 안전하게)
                svg = re.sub(r"<\?xml[^?]*\?>", "", svg, count=1).strip()
                return svg
        return m.group(0)
    # img 태그 src 캡처 (self-closing 또는 닫는 형태 모두)
    return re.sub(r'<img[^>]*src="([^"]+)"[^>]*/?>', repl, html)


def render_markdown(src: str, base_dir: Path) -> str:
    """Markdown → HTML. $...$ / $$...$$ 는 MathJax 가 처리하도록 그대로 둠.
    SVG 이미지는 inline 임베드 (offline 동작용)."""
    placeholders = {}
    counter = [0]

    def stash(m, tag):
        key = f"@@MATHPLACEHOLDER_{tag}_{counter[0]}@@"
        placeholders[key] = m.group(0)
        counter[0] += 1
        return key

    # Display math first (greedy)
    src2 = re.sub(r"\$\$([\s\S]*?)\$\$", lambda m: stash(m, "DISPLAY"), src)
    # Then inline math (avoid breaking $)
    src2 = re.sub(r"\$([^\$\n]+?)\$", lambda m: stash(m, "INLINE"), src2)

    html = markdown.markdown(
        src2,
        extensions=["extra", "tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )

    # SVG inline 임베드 (math placeholder 복원 *전*에 해야 SVG 안의 $ 가 안 깨짐)
    html = inline_svg_images(html, base_dir)

    # 복원
    for key, val in placeholders.items():
        html = html.replace(key, val)
    return html


def render_plotly(fig_dict: dict, include_js: bool) -> str:
    fig = go.Figure(fig_dict)
    return pio.to_html(
        fig,
        include_plotlyjs="inline" if include_js else False,
        include_mathjax=False,
        full_html=False,
        default_height="500px",
        config={"responsive": True, "displayModeBar": False},
    )


def render_code(src: str) -> str:
    return f'<pre class="codeblock"><code>{html_lib.escape(src)}</code></pre>'


def render_stream(text: str) -> str:
    # 노이즈 필터
    if "notice" in text and "pip" in text:
        return ""
    if "kernel" in text and "restart" in text:
        return ""
    cleaned = text.rstrip()
    if not cleaned:
        return ""
    return f'<pre class="streamout">{html_lib.escape(cleaned)}</pre>'


def convert(nb_path: Path, out_path: Path, *, show_code: bool, mathjax_js: str) -> None:
    nb = json.loads(nb_path.read_text())

    parts: list[str] = []
    plotly_first = True

    for cell in nb["cells"]:
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)

        if cell["cell_type"] == "markdown":
            parts.append(f'<section class="cell md">{render_markdown(src, nb_path.parent)}</section>')

        elif cell["cell_type"] == "code":
            if show_code and src.strip():
                parts.append(f'<section class="cell code">{render_code(src)}</section>')

            for out in cell.get("outputs", []):
                ot = out.get("output_type")
                if ot == "stream":
                    t = out.get("text", "")
                    if isinstance(t, list):
                        t = "".join(t)
                    snippet = render_stream(t)
                    if snippet:
                        parts.append(f'<section class="cell out">{snippet}</section>')

                elif ot in ("display_data", "execute_result"):
                    data = out.get("data", {})
                    if "application/vnd.plotly.v1+json" in data:
                        fig_dict = data["application/vnd.plotly.v1+json"]
                        html_snippet = render_plotly(fig_dict, include_js=plotly_first)
                        plotly_first = False
                        parts.append(f'<section class="cell figure">{html_snippet}</section>')
                    elif "image/png" in data:
                        png_b64 = data["image/png"]
                        if isinstance(png_b64, list):
                            png_b64 = "".join(png_b64)
                        parts.append(
                            f'<section class="cell figure"><img src="data:image/png;base64,{png_b64}"/></section>'
                        )
                    elif "image/svg+xml" in data:
                        svg = data["image/svg+xml"]
                        if isinstance(svg, list):
                            svg = "".join(svg)
                        parts.append(f'<section class="cell figure">{svg}</section>')
                    elif "text/html" in data:
                        t = data["text/html"]
                        if isinstance(t, list):
                            t = "".join(t)
                        parts.append(f'<section class="cell out">{t}</section>')
                    elif "text/plain" in data:
                        t = data["text/plain"]
                        if isinstance(t, list):
                            t = "".join(t)
                        snippet = render_stream(t)
                        if snippet:
                            parts.append(f'<section class="cell out">{snippet}</section>')

                # ot == "error" 등은 건너뜀

    body = "\n".join(parts)
    title = nb_path.stem.replace("_", " ")

    html_out = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_lib.escape(title)}</title>
<style>
:root {{
  --bg: #0d1117;
  --bg2: #161b22;
  --fg: #e6edf3;
  --fg2: #b9c1d3;
  --grid: #30363d;
  --accent: #58a6ff;
  --orange: #ffa657;
}}
* {{ box-sizing: border-box; }}
html, body {{
  background: var(--bg);
  color: var(--fg);
  font-family: 'Times New Roman', 'Nanum Myeongjo', serif;
  font-size: 16px;
  line-height: 1.65;
  margin: 0;
  padding: 0;
}}
main {{
  max-width: 1100px;
  margin: 0 auto;
  padding: 40px 30px 80px;
}}
h1, h2, h3, h4 {{ font-family: 'Helvetica Neue', 'Pretendard', sans-serif; color: #f0f6fc; }}
h1 {{ font-size: 2.1em; border-bottom: 1px solid var(--grid); padding-bottom: 12px; margin-top: 1.5em; }}
h2 {{ font-size: 1.55em; margin-top: 2em; color: var(--accent); }}
h3 {{ font-size: 1.2em; color: var(--orange); }}
p {{ margin: 0.7em 0; }}
strong {{ color: var(--fg); }}
em {{ color: var(--fg2); }}
code {{
  background: var(--bg2); padding: 2px 6px; border-radius: 3px;
  color: var(--orange); font-family: 'SF Mono', Consolas, monospace; font-size: 0.92em;
}}
.codeblock {{
  background: var(--bg2); padding: 14px 16px; border-radius: 6px;
  overflow-x: auto; color: #d2dae2;
  font-family: 'SF Mono', Consolas, monospace; font-size: 13px;
  border-left: 3px solid var(--grid);
}}
.streamout {{
  background: var(--bg2); padding: 12px 16px; border-radius: 6px;
  overflow-x: auto; color: #98c379; font-family: 'SF Mono', Consolas, monospace;
  font-size: 13px; white-space: pre; max-height: 400px;
}}
table {{ border-collapse: collapse; margin: 14px 0; font-size: 0.95em; }}
th, td {{ border: 1px solid var(--grid); padding: 7px 14px; text-align: left; }}
th {{ background: var(--bg2); color: var(--accent); }}
blockquote {{
  border-left: 4px solid var(--accent);
  background: rgba(88,166,255,0.07);
  margin: 14px 0; padding: 10px 18px;
  color: var(--fg2);
}}
.cell.figure {{ margin: 16px 0; }}
.cell.md ul, .cell.md ol {{ padding-left: 1.8em; }}
hr {{ border: none; border-top: 1px solid var(--grid); margin: 2em 0; }}
img, svg {{ max-width: 100%; height: auto; }}
mjx-container {{ font-size: 1.05em; }}
</style>
<script>
// 인터넷·외부 fetch 차단: assistiveMml/explorer 끄면 speech-rule-engine CDN 호출 안 일어남.
MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true,
    processEnvironments: true
  }},
  options: {{
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
    enableAssistiveMml: false,
    enableMenu: false,
    menuOptions: {{ settings: {{ assistiveMml: false, explorer: false, collapsible: false }} }}
  }},
  loader: {{ load: ['[tex]/ams'] }},
  startup: {{ typeset: true }}
}};
</script>
<script>
{mathjax_js}
</script>
</head>
<body>
<main>
{body}
</main>
</body>
</html>"""

    out_path.write_text(html_out, encoding="utf-8")
    print(f"wrote {out_path}  ({out_path.stat().st_size/1024/1024:.2f} MB)")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--output", "-o", type=Path, default=None)
    ap.add_argument("--show-code", action="store_true",
                    help="코드 셀 소스도 포함 (기본은 출력만)")
    ap.add_argument("--mathjax-bundle", type=Path,
                    default=Path("/tmp/offline_assets/tex-mml-chtml.js"))
    args = ap.parse_args(argv)

    if not args.notebook.exists():
        print(f"notebook 없음: {args.notebook}", file=sys.stderr)
        return 1
    if not args.mathjax_bundle.exists():
        print(f"MathJax 번들 없음: {args.mathjax_bundle}", file=sys.stderr)
        return 1

    out = args.output or args.notebook.with_suffix(".html")
    mathjax_js = args.mathjax_bundle.read_text(encoding="utf-8")

    convert(args.notebook, out, show_code=args.show_code, mathjax_js=mathjax_js)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
