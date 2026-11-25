"""규격 문서를 **밖으로 들고 나갈 수 있게** 묶는다 — 묶음(zip)과 단일 HTML 둘.

저장소 안의 `docs/action_spec.md` 는 그림을 `../viz/...` 로 가리킨다. 저장소 밖으로
들고 나가면 그 경로가 깨지므로, 내보낼 때 경로를 바꾸고 그림을 같이 담는다.

    zip   action_spec.md + images/*.png   — 마크다운 뷰어로 열고 고칠 수 있다
    html  그림을 base64 로 **문서 안에** 박은 한 파일 — 어디서나 열린다, 첨부가 쉽다

    python src/export_spec_bundle.py                 # 둘 다
    python src/export_spec_bundle.py --doc docs/meeting_1126_spec.md
"""
import argparse
import base64
import re
import shutil
import zipfile
from pathlib import Path

import viz_v4_common as C

IMG = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
CSS = """
:root{--ink:#111;--muted:#555;--line:#ddd;--bg:#fff;--code:#f6f6f4;--accent:#1c5cab}
@media (prefers-color-scheme:dark){:root{--ink:#e8e8e6;--muted:#a8a8a4;--line:#3a3a38;
  --bg:#17171a;--code:#1f1f22;--accent:#7fb2ef}}
*{box-sizing:border-box}
body{max-width:48rem;margin:0 auto;padding:2.5rem 1rem 6rem;background:var(--bg);color:var(--ink);
  font:16px/1.75 -apple-system,"Apple SD Gothic Neo","Noto Sans KR",Pretendard,sans-serif;
  word-break:keep-all;overflow-wrap:break-word}
h1{font-size:1.75rem;line-height:1.35;margin:0 0 1.5rem;padding-bottom:.6rem;border-bottom:2px solid var(--line)}
h2{font-size:1.3rem;margin:2.6rem 0 .9rem;padding-top:.9rem;border-top:1px solid var(--line)}
h3{font-size:1.1rem;margin:1.9rem 0 .6rem}h4,h5{font-size:1rem;margin:1.5rem 0 .5rem;color:var(--muted)}
img{max-width:100%;height:auto;display:block;margin:1.2rem auto;border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;margin:1.1rem 0;font-size:.9rem;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:.45rem .6rem;text-align:left;vertical-align:top}
th{background:var(--code);font-weight:600}
code{background:var(--code);padding:.12em .35em;border-radius:3px;font-size:.88em}
pre{background:var(--code);padding:.9rem 1rem;border-radius:6px;overflow-x:auto;line-height:1.55}
pre code{background:none;padding:0;font-size:.84rem}
blockquote{margin:1.1rem 0;padding:.1rem 0 .1rem 1rem;border-left:3px solid var(--accent);color:var(--muted)}
blockquote strong{color:var(--ink)}
a{color:var(--accent)}hr{border:0;border-top:1px solid var(--line);margin:2.5rem 0}
"""


def collect(doc):
    """문서가 가리키는 그림을 (원본경로, 담을이름) 으로 모은다. 없는 파일은 알린다."""
    out, base = [], doc.parent
    for m in IMG.finditer(doc.read_text()):
        src = (base / m.group(2)).resolve()
        if not src.exists():
            print(f"  ⚠️ 그림 없음: {m.group(2)}")
            continue
        out.append((src, src.name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="docs/action_spec.md")
    ap.add_argument("--out", default=None, help="기본: viz/export/")
    a = ap.parse_args()

    doc = (C.ROOT / a.doc).resolve()
    out = Path(a.out) if a.out else (C.ROOT / "viz" / "export")
    out.mkdir(parents=True, exist_ok=True)
    text, stem = doc.read_text(), doc.stem
    imgs = collect(doc)
    print(f"{doc.name} · {len(text):,}자 · 그림 {len(imgs)}장")

    # ── ① zip — 그림을 images/ 로 옮기고 경로를 다시 쓴다
    md = IMG.sub(lambda m: f"![{m.group(1)}](images/{Path(m.group(2)).name})", text)
    zp = out / f"{stem}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{stem}/{stem}.md", md)
        for src, name in imgs:
            z.write(src, f"{stem}/images/{name}")
    print(f"  묶음 → {zp}  ({zp.stat().st_size/1024:.0f} KB)")

    # ── ② 단일 HTML — 그림을 base64 로 문서 안에 박는다
    import markdown
    def embed(m):
        src = (doc.parent / m.group(2)).resolve()
        if not src.exists():
            return m.group(0)
        b64 = base64.b64encode(src.read_bytes()).decode()
        return f'![{m.group(1)}](data:image/png;base64,{b64})'
    body = markdown.markdown(IMG.sub(embed, text),
                             extensions=["tables", "fenced_code", "toc", "sane_lists"])
    title = text.split("\n", 1)[0].lstrip("# ").strip()
    hp = out / f"{stem}.html"
    hp.write_text(f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
                  f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                  f'<title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>')
    print(f"  단일 HTML → {hp}  ({hp.stat().st_size/1024:.0f} KB · 그림 포함)")


if __name__ == "__main__":
    main()
