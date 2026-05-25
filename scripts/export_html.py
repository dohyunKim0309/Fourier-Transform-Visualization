"""Export examples/*.ipynb to single-file offline HTML.

- Re-executes the notebook (skips %pip install cells)
- Converts plotly mime-type outputs to inline HTML divs
- Inlines plotly.js once at the top
- Replaces MathJax CDN with self-contained KaTeX bundle (CSS + JS + base64 fonts)

Usage:
    python3 scripts/export_html.py examples/04_2d_fft_undersampling.ipynb

Build-time needs internet to fetch KaTeX assets (cached in .cache/).
The resulting .html opens fully offline.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.request
from pathlib import Path

import nbformat
import plotly.io as pio
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / '.cache'
CACHE.mkdir(exist_ok=True)

KATEX_VER = '0.16.11'
KATEX_BASE = f'https://cdn.jsdelivr.net/npm/katex@{KATEX_VER}/dist'


def fetch(url: str, cache_name: str) -> bytes:
    p = CACHE / cache_name
    if not p.exists():
        print(f'  fetching {url}')
        urllib.request.urlretrieve(url, p)
    return p.read_bytes()


def build_katex_bundle() -> tuple[str, str, str]:
    """Return (css-with-inline-fonts, katex.min.js, auto-render.min.js)."""
    css = fetch(f'{KATEX_BASE}/katex.min.css', 'katex.min.css').decode()
    # Inline fonts referenced by url(fonts/Name.woff2)
    mime_map = {'woff2': 'font/woff2', 'woff': 'font/woff', 'ttf': 'font/ttf'}
    for m in list(re.finditer(r'url\(fonts/(\S+?\.(woff2|woff|ttf))\)', css)):
        fname, ext = m.group(1), m.group(2)
        try:
            data = fetch(f'{KATEX_BASE}/fonts/{fname}', f'katex_{fname}')
            b64 = base64.b64encode(data).decode()
            css = css.replace(
                f'url(fonts/{fname})',
                f'url(data:{mime_map[ext]};base64,{b64})',
            )
        except Exception as e:  # noqa: BLE001
            print(f'  warn: failed to inline {fname}: {e}')
    katex_js = fetch(f'{KATEX_BASE}/katex.min.js', 'katex.min.js').decode()
    auto_js = fetch(
        f'{KATEX_BASE}/contrib/auto-render.min.js',
        'auto-render.min.js',
    ).decode()
    return css, katex_js, auto_js


def transform_plotly_outputs(nb) -> int:
    """Replace plotly mime-type outputs with text/html divs (no plotly.js)."""
    count = 0
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        new_outs = []
        for out in cell.outputs:
            data = out.get('data') or {}
            if (
                out.output_type in ('display_data', 'execute_result')
                and 'application/vnd.plotly.v1+json' in data
            ):
                plotly_data = data['application/vnd.plotly.v1+json']
                fig = pio.from_json(json.dumps(plotly_data))
                html_div = pio.to_html(
                    fig,
                    include_plotlyjs=False,
                    full_html=False,
                    div_id=f'plotly-{count}',
                    config={'displayModeBar': False, 'responsive': True},
                )
                new_outs.append(
                    nbformat.v4.new_output(
                        output_type='display_data',
                        data={'text/html': html_div},
                        metadata={},
                    )
                )
                count += 1
            else:
                new_outs.append(out)
        cell.outputs = new_outs
    return count


def skip_install_cells(nb) -> int:
    """Comment out %pip install lines so re-execution doesn't reinstall."""
    n = 0
    for c in nb.cells:
        if c.cell_type == 'code' and '%pip install' in c.source:
            c.source = '# (install cell skipped during HTML export)\npass\n'
            n += 1
    return n


def export(nb_path: Path, html_path: Path) -> None:
    print(f'reading {nb_path}')
    nb = nbformat.read(nb_path, as_version=4)

    n_skipped = skip_install_cells(nb)
    print(f'skipped {n_skipped} install cell(s)')

    print('executing notebook')
    ep = ExecutePreprocessor(timeout=240, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': str(nb_path.parent)}})

    n_plotly = transform_plotly_outputs(nb)
    print(f'converted {n_plotly} plotly figure(s) to inline HTML')

    print('exporting to HTML')
    exporter = HTMLExporter()
    exporter.embed_images = True
    body, _ = exporter.from_notebook_node(nb)

    print('inlining plotly.js')
    plotly_script = (
        '<script type="text/javascript">'
        f'/*plotly.js v{pio.__name__}*/{get_plotlyjs()}'
        '</script>'
    )

    print('building KaTeX bundle (fonts inlined)')
    katex_css, katex_js, auto_js = build_katex_bundle()
    katex_block = (
        f'<style>{katex_css}</style>\n'
        f'<script>{katex_js}</script>\n'
        f'<script>{auto_js}</script>\n'
        '<script>\n'
        "window.addEventListener('DOMContentLoaded', function() {\n"
        '  renderMathInElement(document.body, {\n'
        '    delimiters: [\n'
        "      {left: '$$', right: '$$', display: true},\n"
        "      {left: '\\\\[', right: '\\\\]', display: true},\n"
        "      {left: '$', right: '$', display: false},\n"
        "      {left: '\\\\(', right: '\\\\)', display: false}\n"
        '    ],\n'
        '    throwOnError: false\n'
        '  });\n'
        '});\n'
        '</script>\n'
    )

    # Strip the entire MathJax block (loader + config) emitted by nbconvert
    n_before = body.lower().count('cdnjs.cloudflare.com/ajax/libs/mathjax')
    body = re.sub(
        r'<!--\s*Load mathjax\s*-->.*?<!--\s*End of mathjax configuration\s*-->',
        '',
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Fallback: also strip any stray cdnjs mathjax script tags
    body = re.sub(
        r'<script[^>]*cdnjs\.cloudflare\.com/ajax/libs/mathjax[^>]*>\s*</script>',
        '',
        body,
        flags=re.IGNORECASE,
    )
    n_after = body.lower().count('cdnjs.cloudflare.com/ajax/libs/mathjax')
    print(f'  stripped MathJax CDN references: {n_before} -> {n_after}')

    # Strip require.js loader (nbconvert template; we inline plotly directly)
    body = re.sub(
        r'<script[^>]*cdnjs\.cloudflare\.com/ajax/libs/require\.js[^>]*>\s*</script>',
        '',
        body,
        flags=re.IGNORECASE,
    )

    # Inject bundles before </head>
    body = body.replace('</head>', katex_block + plotly_script + '</head>', 1)

    html_path.write_text(body, encoding='utf-8')
    size_mb = html_path.stat().st_size / 1e6
    print(f'wrote {html_path} ({size_mb:.2f} MB)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('notebook', help='Path to .ipynb (relative or absolute)')
    ap.add_argument('--out', help='Output .html path (default: same dir, same stem)')
    args = ap.parse_args()

    nb_path = Path(args.notebook).resolve()
    if not nb_path.exists():
        sys.exit(f'not found: {nb_path}')
    html_path = Path(args.out).resolve() if args.out else nb_path.with_suffix('.html')

    export(nb_path, html_path)


if __name__ == '__main__':
    main()
