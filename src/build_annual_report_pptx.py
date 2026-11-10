#!/usr/bin/env python3
"""연차보고서 슬라이드(김영진 몫 4장)를 가이드 PPT 의 3번 슬라이드 틀로 만든다.

  python src/build_annual_report_pptx.py [가이드.pptx] [출력.pptx]

가이드 파일을 그대로 열어 3번 슬라이드(제목·부제·소제목·내용·그림 틀)를 복제하므로
서체(NanumSquare)·색·위치가 팀 덱과 같다. 그림은 src/viz_annual_report.py 가 만든 PNG 를 쓴다.
python-pptx 가 필요하다.
"""
import copy
import os
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "docs", "figures", "annual_report")
GUIDE = sys.argv[1] if len(sys.argv) > 1 else "/home/user/annual_report/guide/guide.pptx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/user/annual_report/slides_dataset_sota.pptx"

INK, MUTED, ACCENT, LINE, BAND = "2B2B2B", "6E6E6E", "862633", "E3DDD9", "F7F4F2"
FONT = "NanumSquare"

# ── 슬라이드 내용 (원문 표기) ─────────────────────────────────────────
S1 = dict(title="자율주행 데이터셋 조사", fig="fig1_datasets.png",
          caption="그림 1. 데이터셋별 공개 규모(총 주행 시간)와 observation→prediction 구간",
          bullets=[("autonomous vehicle fleet 수집", " — camera·lidar·radar 로 추적한 궤적 + HD map (Argoverse 1·2, WOMD, nuScenes, Lyft Level 5)"),
                   ("aerial perspective·고정 camera", " — 위치가 정밀하나 highway 에 한정 (highD, NGSIM)"),
                   ("prediction horizon 확대", " — 3 s → 6 s → 8 s, 2개 → 6개 도시"),
                   ("task 확장", " — single-agent → multi-agent, Interaction Prediction, Sim Agents")],
          source="출처: arxiv.org/abs/1911.02620 · arxiv.org/abs/2301.00493 · waymo.com/open/about · arxiv.org/abs/1903.11027 · arxiv.org/abs/2006.14480 · doi.org/10.1109/ITSC.2018.8569552")

S2 = dict(title="데이터셋 특성 비교",
          cols=["Dataset", "지역", "규모", "obs → pred", "map", "metric"],
          widths=[1.75, 1.55, 2.05, 1.5, 3.25, 2.05],
          rows=[["Argoverse 1 (2019)", "Pittsburgh, Miami", "324,557 × 5 s", "2 s → 3 s, 10 Hz", "lane centerlines, driveable area, ground height", "minADE / minFDE / MR / DAC"],
                ["Argoverse 2 (2021)", "미국 6개 도시", "250,000 × 11 s (763 h)", "5 s → 6 s, 10 Hz", "lane segments, 3D lane boundaries, crosswalks", "brier-minFDE6 / minADE6 / MR6"],
                ["WOMD (2021)", "미국 6개 도시", "103,354 × 20 s (574 h)", "1 s → 8 s, 10 Hz", "lane centers, road edges, crosswalks + traffic signal states", "Soft mAP / mAP / minADE / MR"],
                ["nuScenes (2020)", "Boston, Singapore", "1,000 × 20 s", "2 s → 6 s, 2 Hz", "semantic map (11 semantic layers)", "minADE_k / MissRate_2_k / OffRoadRate"],
                ["Lyft Level 5 (2020)", "Palo Alto 단일 route", "170,000 × 25 s (1,118 h)", "5 s (baseline), 10 Hz", "HD semantic map + aerial view", "—"],
                ["highD (2018)", "독일 highway", "110,000 vehicles (16.5 h)", "— (25 fps)", "—", "—"],
                ["NGSIM (2006)", "미국 highway", "9,206 vehicles (1.5 h)", "—", "—", "—"]],
          source="출처: 위 데이터셋 원논문 + waymo.com/open/about (WOMD v1.3.1) + github.com/nutonomy/nuscenes-devkit (prediction README)")

S3 = dict(title="최신 SOTA 모델 조사", fig="fig2_av2_sota.png",
          caption="그림 2. Argoverse 2 test minADE6 (K=6) — 논문 보고치, 점선: leaderboard 1위 (2026-11-10)",
          bullets=[("2020 vectorized representation", " — VectorNet, LaneGCN"),
                   ("2022 Transformer·invariance", " — HiVT, MTR"),
                   ("2023 query-centric·self-supervised", " — QCNet, Forecast-MAE"),
                   ("2023–24 language modeling·generative", " — MotionLM, SMART"),
                   ("2024–25 경량화·state space model", " — SIMPL, DeMo, Trajectory Mamba, FutureNet-LOF")],
          source="출처: arxiv.org/abs/2005.04259 · 2007.13732 · 2209.13508 · 2308.09882 · 2309.16534 · 2405.15677 · 2402.02519 · 2410.05982 · 2503.10898 · 2406.14422 · doi.org/10.1109/CVPR52729.2023.01713")

S4 = dict(title="SOTA 모델 비교와 시사점",
          cols=["Model", "발표", "핵심 (원문 표기)", "minADE6", "minFDE6", "비고"],
          widths=[1.9, 1.55, 4.65, 1.05, 1.05, 1.95], numeric=[3, 4],
          rows=[["QCNet", "CVPR 2023", "query-centric paradigm, anchor-based refinement", "0.65", "1.29", "ensemble 0.62 / 1.19"],
                ["ProphNet", "CVPR 2023", "agent-centric model with anchor-informed proposals", "0.68", "1.33", "코드 미공개"],
                ["Forecast-MAE", "ICCV 2023", "masked autoencoders, self-supervised pre-training", "0.709", "1.392", "ensemble 0.690 / 1.338"],
                ["SIMPL", "RA-L 2024", "symmetric global feature fusion, Bernstein basis polynomials", "0.72", "1.43", "1.9M, 공개 코드"],
                ["DeMo", "NeurIPS 2024", "mode queries + state queries, Attention and Mamba", "0.61", "1.17", "ensemble 0.60 / 1.11"],
                ["Trajectory Mamba", "CVPR 2025", "selective state-space model (SSM)", "0.64", "1.24", "4.54M"],
                ["FutureNet-LOF", "ICRA 2025", "future context encoding, Lane Occupancy Field", "0.58", "1.07", "ensemble"],
                ["IMR", "2025-09 제출", "leaderboard 1위 (논문 미확인)", "0.5714", "1.0684", "brier-minFDE6 1.6248"]],
          bullets=[("성능 포화·ensemble 의존", " — 상위권 minADE6 0.57~0.65 m, 상위 기록 다수가 ensemble"),
                   ("거리 지표 외 평가 부재", " — WOSAC 2023 minADE 1위(QCNeXt 1.083)가 off-road likelihood 최하위(0.360)"),
                   ("task 확장", " — Waymo 2025 Challenges: E2E Driving, Scenario Generation, Sim Agents, Interaction Prediction")],
          source="출처: Argoverse 2 test (K=6), 각 모델 논문 · eval.ai/web/challenges/challenge-page/1719/leaderboard/4098 · arxiv.org/abs/2405.15677 · waymo.com/open/challenges")


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
        spc = p.find(qn("a:pPr")).find(qn("a:spcAft")).find(qn("a:spcPts"))
        spc.set("val", str(int(space_after * 100)))
        tf._txBody.append(p)
        para = tf.paragraphs[-1]
        for txt, bold in ((head, True), (rest, False)):
            r = para.add_run()
            r.text = txt
            set_run_font(r, size, bold=bold)
        if end is not None:                       # endParaRPr 는 run 뒤에 와야 한다
            p.remove(end)
            p.append(end)


def add_text(slide, x, y, w, h, text, size, color=MUTED):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    r = tf.paragraphs[0].add_run()
    r.text = text
    set_run_font(r, size, color)
    return tb


def add_table(slide, spec, x, y, row_h, size):
    cols, rows, widths = spec["cols"], spec["rows"], spec["widths"]
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


def main():
    prs = Presentation(GUIDE)
    tmpl = prs.slides[2]                           # '영진 : 1~2페이지' 틀
    slides = [tmpl]
    for _ in range(3):                             # 같은 레이아웃으로 새 슬라이드를 만들고 도형을 통째로 복사
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

    for slide, spec in zip(slides, (S1, S2, S3, S4)):
        remove(shape(slide, "직사각형 5"))         # 가이드 안내 상자
        title_ph = next(s for s in slide.shapes if s.is_placeholder)
        title_ph.text_frame.paragraphs[0].runs[0].text = spec["title"]
        content, frame = shape(slide, "Text 4"), shape(slide, "Shape 5")
        if "fig" in spec:
            fill_bullets(content, spec["bullets"], 16, 10)
            put_figure(slide, frame, os.path.join(FIG, spec["fig"]), spec["caption"])
        else:
            remove(frame)
            if "bullets" in spec:                  # 표 + 시사점
                add_table(slide, spec, 0.6, 2.2, 0.36, 11)
                content.left, content.top = Inches(0.6), Inches(5.65)
                content.width, content.height = Inches(12.1), Inches(1.15)
                fill_bullets(content, spec["bullets"], 14, 4)
            else:
                remove(content)
                add_table(slide, spec, 0.6, 2.2, 0.56, 11)
        add_text(slide, 0.6, 7.03, 12.1, 0.3, spec["source"], 8.5)

    prs.save(OUT)
    print("저장:", OUT, "| 슬라이드", len(prs.slides))


if __name__ == "__main__":
    main()
