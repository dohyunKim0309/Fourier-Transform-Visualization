"""
Jupyter 노트북 → 인터넷·파이썬 없이 동작하는 단일 HTML 파일.

04 노트북의 HTML 변환 결과(JupyterLab template) 스타일을 그대로 따라간다:
  1. 각 Plotly figure 의 text/html 표현을 노트북 output 에 주입
     (첫 figure 는 plotly.js inline, 이후는 share)
  2. jupyter nbconvert --to html (lab template) 로 변환
  3. 결과 HTML 의 MathJax CDN 참조를 inline MathJax bundle 로 교체
  4. 셀 간 세로 간격을 살짝 키우는 custom CSS 주입

Usage:
    python examples/nb_to_offline_html.py examples/05_beyond_nyquist_hilbert.ipynb
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio


# ─────────────────────────────────────────────────────────────────────────
# 추가 CSS: cell 간격 늘리기 + Plotly figure block 사이 여백
# ─────────────────────────────────────────────────────────────────────────
EXTRA_CSS = """
/* 노트북 셀 간 세로 간격을 살짝 키워 가독성 향상 */
.jp-Cell {
  margin: 18px 0 !important;
}
.jp-Cell .jp-OutputArea-output {
  margin: 12px 0;
}
.jp-RenderedHTMLCommon .plotly-graph-div,
.jp-RenderedHTMLCommon .js-plotly-plot {
  margin: 14px 0;
}
"""


def inject_plotly_html(nb: dict) -> dict:
    """노트북의 각 Plotly output 에 text/html mimetype 추가.

    첫 plotly figure 만 plotly.js 를 inline 임베드, 이후는 share 하여 크기 절약.
    """
    out = copy.deepcopy(nb)
    plotly_first = True
    for cell in out["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for o in cell.get("outputs", []):
            if "data" not in o:
                continue
            if "application/vnd.plotly.v1+json" not in o["data"]:
                continue
            fig_dict = o["data"]["application/vnd.plotly.v1+json"]
            fig = go.Figure(fig_dict)
            html_snippet = pio.to_html(
                fig,
                include_plotlyjs="inline" if plotly_first else False,
                include_mathjax=False,
                full_html=False,
                config={"responsive": True, "displayModeBar": False},
            )
            plotly_first = False
            o["data"]["text/html"] = html_snippet
    return out


def run_nbconvert(nb_path: Path, out_path: Path) -> None:
    """jupyter nbconvert --to html (lab template) — MathJax 는 CDN 으로 두고
    뒤에서 inline 교체."""
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "html",
        "--template", "lab",
        str(nb_path),
        "--output", str(out_path.resolve()),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def inline_svg_images(html: str, base_dir: Path) -> str:
    """`<img src="...svg">` 를 SVG inline 으로 치환."""
    def repl(m):
        src = m.group(1)
        if src.lower().endswith(".svg"):
            svg_path = (base_dir / src).resolve()
            if svg_path.exists():
                svg = svg_path.read_text(encoding="utf-8")
                svg = re.sub(r"<\?xml[^?]*\?>", "", svg, count=1).strip()
                return svg
        return m.group(0)
    return re.sub(r'<img[^>]*src="([^"]+)"[^>]*/?>', repl, html)


def replace_requirejs_with_inline(html: str, requirejs_js: str) -> str:
    """nbconvert lab template 이 자동 삽입하는 require.js CDN 참조를 inline 으로 교체."""
    pattern = re.compile(
        r'<script[^>]*src="https?://[^"]*require[^"]*"[^>]*>\s*</script>',
        re.IGNORECASE,
    )
    inline_script = f"<script>\n{requirejs_js}\n</script>"
    # lambda 로 치환해야 백슬래시가 regex 백참조로 해석되지 않음
    return pattern.sub(lambda m: inline_script, html, count=1)


def replace_mathjax_with_inline(html: str, mathjax_js: str) -> str:
    """nbconvert 가 만든 MathJax CDN 스크립트 / 설정 블록을 offline inline 버전으로 교체.

    nbconvert 의 lab template 은 보통 다음 형태로 MathJax 를 로드:
        <script src=".../mathjax/MathJax.js?config=..." ></script>
    혹은 inline config + 외부 src. 이를 우리가 가진 tex-mml-chtml bundle 로 교체.
    """
    # 외부 MathJax script src 제거
    html = re.sub(
        r'<script[^>]*src="[^"]*mathjax[^"]*"[^>]*></script>',
        "",
        html,
        flags=re.IGNORECASE,
    )
    # MathJax CDN 참조 (URL) 제거
    html = re.sub(
        r'<script[^>]*src="https?://[^"]*mathjax[^"]*"[^>]*>[^<]*</script>',
        "",
        html,
        flags=re.IGNORECASE,
    )
    # MathJax 설정 + inline bundle 을 </head> 직전에 삽입
    config_and_bundle = f"""
<script>
window.MathJax = {{
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
"""
    # str.replace 는 백슬래시 escape 해석 안 함 — 안전
    html = html.replace("</head>", config_and_bundle + "</head>", 1)
    return html


def inject_extra_css(html: str) -> str:
    """셀/그림 사이 여백을 살짝 키우는 custom CSS 를 </head> 직전에 주입."""
    css_block = f"<style>{EXTRA_CSS}</style>"
    return html.replace("</head>", css_block + "</head>", 1)


def convert(nb_path: Path, out_path: Path, *, mathjax_js: str, requirejs_js: str) -> None:
    # 1. notebook 읽고 plotly text/html 주입
    nb_raw = json.loads(nb_path.read_text())
    nb_processed = inject_plotly_html(nb_raw)

    # 2. 임시 파일로 저장 후 nbconvert 실행
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ipynb", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(nb_processed, tmp, ensure_ascii=False)
        tmp_path = Path(tmp.name)

    try:
        # nbconvert 가 output 옆에 _files 폴더 만들 수 있어 임시 디렉터리에서 작업
        with tempfile.TemporaryDirectory() as work_dir:
            work_out = Path(work_dir) / "out.html"
            run_nbconvert(tmp_path, work_out)
            html = work_out.read_text(encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)

    # 3. SVG 이미지 inline (회로 SVG 등)
    html = inline_svg_images(html, nb_path.parent)

    # 4. MathJax CDN → inline 교체
    html = replace_mathjax_with_inline(html, mathjax_js)

    # 5. require.js CDN → inline 교체
    html = replace_requirejs_with_inline(html, requirejs_js)

    # 6. extra CSS 주입 (셀 간격 등)
    html = inject_extra_css(html)

    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--output", "-o", type=Path, default=None)
    ap.add_argument(
        "--mathjax-bundle",
        type=Path,
        # SVG 출력 번들 (path 로 그려 폰트 파일 불필요 → 항상 깔끔).
        # CHTML 을 쓰려면 tex-mml-chtml.js 를 명시.
        default=Path("/tmp/offline_assets/tex-mml-svg.js"),
    )
    ap.add_argument(
        "--requirejs-bundle",
        type=Path,
        default=Path("/tmp/offline_assets/require.min.js"),
    )
    args = ap.parse_args(argv)

    if not args.notebook.exists():
        print(f"notebook 없음: {args.notebook}", file=sys.stderr)
        return 1
    if not args.mathjax_bundle.exists():
        print(f"MathJax 번들 없음: {args.mathjax_bundle}", file=sys.stderr)
        return 1
    if not args.requirejs_bundle.exists():
        print(f"require.js 번들 없음: {args.requirejs_bundle}", file=sys.stderr)
        return 1

    out = args.output or args.notebook.with_suffix(".html")
    mathjax_js   = args.mathjax_bundle.read_text(encoding="utf-8")
    requirejs_js = args.requirejs_bundle.read_text(encoding="utf-8")

    convert(args.notebook, out, mathjax_js=mathjax_js, requirejs_js=requirejs_js)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
