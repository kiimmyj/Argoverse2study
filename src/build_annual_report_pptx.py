#!/usr/bin/env python3
"""연차보고서 슬라이드(김영진 몫)를 가이드 PPT 의 3번 슬라이드 틀로 만든다.

  python src/build_annual_report_pptx.py [가이드.pptx] [출력.pptx]

구성(5장): 데이터셋 조사 → 데이터셋 특성 비교 → 데이터셋별 SOTA 활용 사례 → 데이터셋별 주요 task → Argoverse 2 SOTA 비교·시사점.
가이드 파일을 그대로 열어 3번 슬라이드(제목·부제·소제목·내용·그림 틀)를 복제하므로
서체(NanumSquare)·색·위치가 팀 덱과 같다. 그림은 src/viz_annual_report.py 가 만든 PNG 를 쓴다.
python-pptx 가 필요하다. 수치·용어는 docs/annual_report_dataset_sota.md 와 같고 출처도 거기 있다.
"""
import copy
import os
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "docs", "figures", "annual_report")
GUIDE = sys.argv[1] if len(sys.argv) > 1 else "/home/user/annual_report/guide/guide.pptx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/user/annual_report/slides_dataset_sota.pptx"

INK, MUTED, ACCENT, BAND = "2B2B2B", "6E6E6E", "862633", "F7F4F2"
FONT = "NanumSquare"

# ── 슬라이드 내용 (원문 표기) ─────────────────────────────────────────
SLIDES = [
    dict(kind="text_fig", title="자율주행 데이터셋 조사", fig="fig1_datasets.png",
         caption="그림 1. 데이터셋별 공개 규모(총 주행 시간)와 observation→prediction 구간",
         bullets=[("autonomous vehicle fleet 수집", " — camera·lidar·radar 로 추적한 궤적 + HD map (Argoverse 1·2, WOMD, nuScenes, Lyft Level 5)"),
                  ("aerial perspective·고정 camera", " — 위치가 정밀하나 highway 에 한정 (highD, NGSIM)"),
                  ("prediction horizon 확대", " — 3 s → 6 s → 8 s, 2개 → 6개 도시"),
                  ("vector HD map 표준화", " — lane graph·crosswalks·traffic signal states 제공")],
         source="출처: arxiv.org/abs/1911.02620 · arxiv.org/abs/2301.00493 · waymo.com/open/about · arxiv.org/abs/1903.11027 · arxiv.org/abs/2006.14480 · doi.org/10.1109/ITSC.2018.8569552"),

    dict(kind="table", title="데이터셋 특성 비교",
         table=dict(cols=["Dataset", "지역", "규모", "obs → pred", "map", "metric"],
                    widths=[1.75, 1.55, 2.05, 1.5, 3.25, 2.05], row_h=0.56, size=11,
                    rows=[["Argoverse 1 (2019)", "Pittsburgh, Miami", "324,557 × 5 s", "2 s → 3 s, 10 Hz", "lane centerlines, driveable area, ground height", "minADE / minFDE / MR / DAC"],
                          ["Argoverse 2 (2021)", "미국 6개 도시", "250,000 × 11 s (763 h)", "5 s → 6 s, 10 Hz", "lane segments, 3D lane boundaries, crosswalks", "brier-minFDE6 / minADE6 / MR6"],
                          ["WOMD (2021)", "미국 6개 도시", "103,354 × 20 s (574 h)", "1 s → 8 s, 10 Hz", "lane centers, road edges, crosswalks + traffic signal states", "Soft mAP / mAP / minADE / MR"],
                          ["nuScenes (2020)", "Boston, Singapore", "1,000 × 20 s", "2 s → 6 s, 2 Hz", "semantic map (11 semantic layers)", "minADE_k / MissRate_2_k / OffRoadRate"],
                          ["Lyft Level 5 (2020)", "Palo Alto 단일 route", "170,000 × 25 s (1,118 h)", "5 s (baseline), 10 Hz", "HD semantic map + aerial view", "—"],
                          ["highD (2018)", "독일 highway", "110,000 vehicles (16.5 h)", "— (25 fps)", "—", "—"],
                          ["NGSIM (2006)", "미국 highway", "9,206 vehicles (1.5 h)", "—", "—", "—"]]),
         source="출처: 위 데이터셋 원논문 + waymo.com/open/about (WOMD v1.3.1) + github.com/nutonomy/nuscenes-devkit (prediction README)"),

    dict(kind="text_bigfig", title="데이터셋별 SOTA 모델 활용 사례", fig="fig3_model_dataset_slide.png",
         caption="그림 2. 주요 SOTA 모델이 결과를 보고한 데이터셋",
         bullets=[("Argoverse 1", " — map encoding·invariance 의 초기 benchmark (VectorNet, LaneGCN, HiVT)"),
                  ("Argoverse 2", " — 6 s horizon, 2023년 이후 주 benchmark (QCNet, SIMPL, DeMo)"),
                  ("WOMD", " — joint prediction·generative·Sim Agents (MTR, MotionLM, SMART)"),
                  ("nuScenes", " — lane-graph·end-to-end perception·prediction (PGP, LAformer, ViP3D)"),
                  ("highD·NGSIM", " — highway, 공식 benchmark 없음 (MFTraj)"),
                  ("같은 모델도 데이터셋마다 순위가 바뀜", " — Wayformer 는 WOMD 전 지표 1위, Argoverse 1 은 brier-minFDE 만 1위")],
         source="출처: 각 모델 논문 (arXiv·DOI 링크는 보고서 본문 「출처」 표) · arxiv.org/abs/2207.05844"),

    dict(kind="table", title="데이터셋별 주요 활용 task",
         table=dict(cols=["Dataset", "주 task (궤적 예측)", "대표 SOTA 모델", "그 외 task"],
                    widths=[1.85, 3.75, 3.2, 3.3], row_h=0.62, size=12,
                    rows=[["Argoverse 1", "single-agent motion forecasting (2 s → 3 s)", "VectorNet, LaneGCN, HiVT", "3D tracking"],
                          ["Argoverse 2", "single-agent · multi-agent motion forecasting (5 s → 6 s)", "QCNet, QCNeXt, SIMPL, DeMo, FutureNet-LOF", "3D perception, point cloud forecasting (Sensor · Lidar Dataset)"],
                          ["WOMD", "Motion Prediction, Interaction Prediction, Sim Agents", "MTR, EDA, MotionLM, SMART", "—"],
                          ["nuScenes", "prediction challenge (2 s → 6 s), end-to-end perception · prediction · planning", "PGP, LAformer, ViP3D, UniAD 계열", "3D detection · tracking (원 과제)"],
                          ["Lyft Level 5", "motion forecasting, motion planning, simulation", "—", "—"],
                          ["highD · NGSIM", "highway trajectory prediction, lane change", "MFTraj", "scenario-based safety validation (highD)"]]),
         source="출처: arxiv.org/abs/ 1911.02620 · 2301.00493 · 2104.10133 · 1903.11027 · 2006.14480 · 2208.01582 · 2405.01266 · doi.org/10.1109/ITSC.2018.8569552 · waymo.com/open/challenges"),

    dict(kind="table_fig", title="Argoverse 2 — SOTA 모델 비교 (본 과제 주 데이터)", fig="fig2_av2_sota.png",
         caption="그림 3. Argoverse 2 test minADE6 (K=6) — 논문 보고치, 점선: leaderboard 1위 (2026-11-10)",
         table=dict(cols=["Model", "발표", "minADE6", "minFDE6"],
                    widths=[1.95, 1.45, 1.0, 1.0], row_h=0.33, size=11, numeric=[2, 3],
                    rows=[["QCNet", "CVPR 2023", "0.65", "1.29"],
                          ["ProphNet", "CVPR 2023", "0.68", "1.33"],
                          ["Forecast-MAE", "ICCV 2023", "0.709", "1.392"],
                          ["SIMPL", "RA-L 2024", "0.72", "1.43"],
                          ["DeMo", "NeurIPS 2024", "0.61", "1.17"],
                          ["Trajectory Mamba", "CVPR 2025", "0.64", "1.24"],
                          ["FutureNet-LOF *", "ICRA 2025", "0.58", "1.07"],
                          ["IMR (1위)", "2025-09 제출", "0.5714", "1.0684"]]),
         note="test, K=6, 5 s → 6 s · * ensemble · IMR 은 leaderboard 1위(논문 미확인)",
         bullets=[("성능 포화·ensemble 의존", " — 상위권 minADE6 0.57~0.65 m, 상위 기록 다수가 ensemble"),
                  ("거리 지표 외 평가 부재", " — WOSAC 2023 minADE 1위(QCNeXt 1.083)가 off-road likelihood 최하위(0.360)")],
         source="출처: arxiv.org/abs/ 2303.12071 · 2308.09882 · 2402.02519 · 2410.05982 · 2503.10898 · 2406.14422 · 2405.15677 · doi.org/10.1109/CVPR52729.2023.01713 · eval.ai/web/challenges/challenge-page/1719"),
]


# ── 도우미 ────────────────────────────────────────────────────────────
def shape(slide, name):
    return next(s for s in slide.shapes if s.name == name)


def set_run_font(run, size, color=INK, bold=False):
    f = run.font
    f.size, f.bold, f.name = Pt(size), bold, FONT
    f.color.rgb = RGBColor.from_string(color)
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = rpr.makeelement(qn(tag), {})
            rpr.append(el)
        el.set("typeface", FONT)


def fill_bullets(box, bullets, size, space_after):
    """내용 칸의 첫 문단(▪ 글머리 서식)을 본떠 문단을 만든다. (굵은 머리, 본문) 두 run."""
    tf = box.text_frame
    proto = copy.deepcopy(tf.paragraphs[0]._p)
    for p in list(tf._txBody.findall(qn("a:p"))):
        tf._txBody.remove(p)
    for head, rest in bullets:
        p = copy.deepcopy(proto)
        for r in p.findall(qn("a:r")):
            p.remove(r)
        end = p.find(qn("a:endParaRPr"))
        p.find(qn("a:pPr")).find(qn("a:spcAft")).find(qn("a:spcPts")).set("val", str(int(space_after * 100)))
        tf._txBody.append(p)
        para = tf.paragraphs[-1]
        for txt, bold in ((head, True), (rest, False)):
            r = para.add_run()
            r.text = txt
            set_run_font(r, size, bold=bold)
        if end is not None:                       # endParaRPr 는 run 뒤에 와야 한다
            p.remove(end)
            p.append(end)


def place(shape_, x, y, w, h):
    shape_.left, shape_.top, shape_.width, shape_.height = Inches(x), Inches(y), Inches(w), Inches(h)


def add_text(slide, x, y, w, h, text, size, color=MUTED, bold=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    r = tf.paragraphs[0].add_run()
    r.text = text
    set_run_font(r, size, color, bold)
    return tb


def add_table(slide, spec, x, y):
    cols, rows, widths, row_h, size = spec["cols"], spec["rows"], spec["widths"], spec["row_h"], spec["size"]
    gf = slide.shapes.add_table(len(rows) + 1, len(cols), Inches(x), Inches(y),
                                Inches(sum(widths)), Inches(row_h * (len(rows) + 1)))
    gf.name = "Table"
    tbl = gf.table
    tbl.first_row, tbl.horz_banding = True, False
    for j, w in enumerate(widths):
        tbl.columns[j].width = Inches(w)
    for i in range(len(rows) + 1):
        tbl.rows[i].height = Inches(row_h)
        for j in range(len(cols)):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = 3  # middle
            head = i == 0
            c.fill.solid()
            c.fill.fore_color.rgb = RGBColor.from_string(ACCENT if head else ("FFFFFF" if i % 2 else BAND))
            tf = c.text_frame
            tf.word_wrap = True
            tf.paragraphs[0].text = ""
            r = tf.paragraphs[0].add_run()
            r.text = cols[j] if head else rows[i - 1][j]
            set_run_font(r, size, "FFFFFF" if head else INK, bold=head or j == 0)
            if j in spec.get("numeric", []):
                tf.paragraphs[0].alignment = 3  # 오른쪽 정렬 — 수치끼리 비교
    return gf


def put_figure(slide, frame, png, caption):
    """그림 틀(흰 바탕 둥근 사각형) 안에 그림을 비율 유지로 넣고, '그림' 글자는 지운다."""
    frame.text_frame.paragraphs[0].runs[0].text = ""
    pad = Inches(0.12)
    fw, fh = frame.width - 2 * pad, frame.height - 2 * pad
    pic = slide.shapes.add_picture(png, 0, 0)
    ratio = min(fw / pic.width, fh / pic.height)
    pic.width, pic.height = int(pic.width * ratio), int(pic.height * ratio)
    pic.left = frame.left + (frame.width - pic.width) // 2
    pic.top = frame.top + (frame.height - pic.height) // 2
    pic.name = "Figure"
    add_text(slide, frame.left / 914400, (frame.top + frame.height) / 914400 + 0.08,
             frame.width / 914400, 0.3, caption, 10.5)


def remove(shape_):
    shape_._element.getparent().remove(shape_._element)


def build(slide, spec):
    remove(shape(slide, "직사각형 5"))             # 가이드 안내 상자
    next(s for s in slide.shapes if s.is_placeholder).text_frame.paragraphs[0].runs[0].text = spec["title"]
    content, frame = shape(slide, "Text 4"), shape(slide, "Shape 5")
    kind = spec["kind"]
    if kind == "text_fig":                          # 가이드 그대로: 왼쪽 내용, 오른쪽 그림
        fill_bullets(content, spec["bullets"], 16, 10)
        put_figure(slide, frame, os.path.join(FIG, spec["fig"]), spec["caption"])
    elif kind == "text_bigfig":                     # 세로로 긴 그림 — 그림 틀을 아래로 늘린다
        place(content, 0.6, 2.35, 5.6, 4.4)
        fill_bullets(content, spec["bullets"], 15, 12)
        place(frame, 6.55, 2.1, 6.2, 4.55)
        put_figure(slide, frame, os.path.join(FIG, spec["fig"]), spec["caption"])
    elif kind == "table":
        remove(frame)
        remove(content)
        add_table(slide, spec["table"], 0.6, 2.2)
    elif kind == "table_bullets":                   # 표 + 아래 글머리
        remove(frame)
        t = spec["table"]
        add_table(slide, t, 0.6, 2.2)
        top = 2.2 + t["row_h"] * (len(t["rows"]) + 1) + 0.3
        place(content, 0.6, top, 12.1, 6.85 - top)
        fill_bullets(content, spec["bullets"], 14, 4)
    elif kind == "table_fig":                       # 왼쪽 표(+ 아래 글머리) + 오른쪽 그림
        t = spec["table"]
        add_table(slide, t, 0.6, 2.35)
        bottom = 2.35 + t["row_h"] * (len(t["rows"]) + 1)
        add_text(slide, 0.6, bottom + 0.1, 5.4, 0.3, spec["note"], 10.5)
        if "bullets" in spec:
            place(content, 0.6, bottom + 0.5, 5.4, 6.9 - bottom - 0.5)
            fill_bullets(content, spec["bullets"], 13, 4)
        else:
            remove(content)
        put_figure(slide, frame, os.path.join(FIG, spec["fig"]), spec["caption"])
    elif kind == "two_tables":                      # 좌우 표 두 개 + 아래 글머리
        remove(frame)
        L, R = spec["left"], spec["right"]
        xr = 0.6 + sum(L["widths"]) + 0.45
        for t, x in ((L, 0.6), (R, xr)):
            add_text(slide, x, 2.2, sum(t["widths"]), 0.3, t["label"], 12, INK, True)
            add_table(slide, t, x, 2.55)
        top = 2.55 + L["row_h"] * (len(L["rows"]) + 1) + 0.25
        place(content, 0.6, top, 12.1, 6.75 - top)
        fill_bullets(content, spec["bullets"], 14, 4)
        add_text(slide, xr, 2.55 + R["row_h"] * (len(R["rows"]) + 1) + 0.12, sum(R["widths"]), 0.5, spec["note"], 10.5)
    add_text(slide, 0.6, 7.03, 12.1, 0.3, spec["source"], 8.5)


def main():
    prs = Presentation(GUIDE)
    tmpl = prs.slides[2]                           # '영진 : 1~2페이지' 틀
    slides = [tmpl]
    for _ in range(len(SLIDES) - 1):               # 같은 레이아웃으로 새 슬라이드를 만들고 도형을 통째로 복사
        s = prs.slides.add_slide(tmpl.slide_layout)
        for ph in list(s.placeholders):
            remove(ph)
        for el in tmpl.shapes._spTree.iterchildren():
            if el.tag in (qn("p:sp"), qn("p:pic"), qn("p:graphicFrame"), qn("p:grpSp")):
                s.shapes._spTree.append(copy.deepcopy(el))
        slides.append(s)

    keep = {id(s._element) for s in slides}        # 나머지 슬라이드는 덱에서 뺀다
    sld_ids = prs.slides._sldIdLst
    for sld_id, slide in list(zip(list(sld_ids), list(prs.slides))):
        if id(slide._element) not in keep:
            prs.part.drop_rel(sld_id.rId)
            sld_ids.remove(sld_id)

    for slide, spec in zip(slides, SLIDES):
        build(slide, spec)
    prs.save(OUT)
    print("저장:", OUT, "| 슬라이드", len(prs.slides))


if __name__ == "__main__":
    main()
