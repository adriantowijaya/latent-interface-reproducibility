from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def svg_bar_chart(title: str, values: pd.Series) -> str:
    width, height = 900, 420
    left, top, plot_w, plot_h = 90, 50, 760, 280
    vals = values.fillna(0.0).astype(float)
    max_v = max(float(vals.max()), 1.0)
    bar_w = plot_w / max(len(vals), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="Arial" font-size="18" fill="#111">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for i, (name, value) in enumerate(vals.items()):
        h = (float(value) / max_v) * (plot_h - 10)
        x = left + i * bar_w + 8
        y = top + plot_h - h
        bw = max(bar_w - 16, 8)
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="#2f6f73"/>')
        parts.append(f'<text x="{x + bw / 2:.1f}" y="{top + plot_h + 18}" text-anchor="middle" font-family="Arial" font-size="10" fill="#111">{escape(str(name)[:18])}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "reproduced_outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    made = 0
    for path in sorted((root / "results").rglob("*.csv")):
        df = pd.read_csv(path)
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            continue
        means = numeric[list(numeric.columns[:6])].mean(numeric_only=True)
        (out / f"{path.stem}_numeric_means.svg").write_text(
            svg_bar_chart(path.stem, means), encoding="utf-8"
        )
        made += 1
    print(f"STATUS=PASS figures_written={made} output={out}")


if __name__ == "__main__":
    main()
