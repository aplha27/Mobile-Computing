"""
SensorQA Project Presentation Generator
Generates a comprehensive PowerPoint presentation for the SensorQA project.
Run with: py -3 generate_presentation.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── Colour palette ──────────────────────────────────────────────────────────
DARK_NAVY   = RGBColor(0x0D, 0x1B, 0x2A)   # slide backgrounds (dark)
ACCENT_BLUE = RGBColor(0x1A, 0x73, 0xE8)   # primary accent
ACCENT_CYAN = RGBColor(0x00, 0xC9, 0xFF)   # secondary accent / highlights
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY  = RGBColor(0xE8, 0xF0, 0xFE)
MID_GRAY    = RGBColor(0xB0, 0xC4, 0xDE)
DARK_GRAY   = RGBColor(0x2D, 0x3A, 0x4B)
GREEN       = RGBColor(0x0F, 0x9D, 0x58)
ORANGE      = RGBColor(0xF4, 0xB4, 0x00)
RED         = RGBColor(0xDB, 0x44, 0x37)

# ── Helper: set slide background ────────────────────────────────────────────
def set_bg(slide, color):
    from pptx.oxml.ns import qn
    from lxml import etree
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

# ── Helper: add a plain text box ────────────────────────────────────────────
def add_textbox(slide, text, left, top, width, height,
                font_size=18, bold=False, color=WHITE,
                align=PP_ALIGN.LEFT, italic=False, wrap=True):
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height))
    txBox.word_wrap = wrap
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox

# ── Helper: add a coloured rounded rectangle ────────────────────────────────
def add_rect(slide, left, top, width, height, fill_color, line_color=None):
    from pptx.util import Inches
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape

# ── Helper: add bullet list ──────────────────────────────────────────────────
def add_bullets(slide, items, left, top, width, height,
                font_size=16, color=WHITE, bold_first=False):
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height))
    txBox.word_wrap = True
    tf = txBox.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(4)
        run = p.add_run()
        run.text = item
        run.font.size = Pt(font_size)
        run.font.color.rgb = color
        if bold_first and items.index(item) == 0:
            run.font.bold = True

# ── Helper: section divider card ────────────────────────────────────────────
def make_section_slide(prs, title, subtitle, section_num):
    slide_layout = prs.slide_layouts[6]  # blank
    slide = prs.slides.add_slide(slide_layout)
    set_bg(slide, DARK_NAVY)

    # left accent bar
    add_rect(slide, 0, 0, 0.08, 7.5, ACCENT_BLUE)

    # large faint section number
    add_textbox(slide, str(section_num), 7.5, 0.3, 2.2, 2.0,
                font_size=110, bold=True,
                color=RGBColor(0x1A, 0x73, 0xE8),
                align=PP_ALIGN.RIGHT)

    add_textbox(slide, title, 0.5, 2.5, 8.5, 1.2,
                font_size=44, bold=True, color=WHITE)
    add_textbox(slide, subtitle, 0.5, 3.9, 8.0, 1.0,
                font_size=22, color=ACCENT_CYAN)
    return slide

# ════════════════════════════════════════════════════════════════════════════
#  BUILD PRESENTATION
# ════════════════════════════════════════════════════════════════════════════
prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 1 — Title slide
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(BLANK)
set_bg(slide, DARK_NAVY)

# top gradient bar
add_rect(slide, 0, 0, 13.33, 0.12, ACCENT_BLUE)

# decorative large circle (top-right)
circ = slide.shapes.add_shape(9, Inches(9.5), Inches(-1.5), Inches(5), Inches(5))
circ.fill.solid(); circ.fill.fore_color.rgb = RGBColor(0x1A, 0x73, 0xE8)
circ.line.fill.background()

# Title block
add_textbox(slide, "SensorQA", 0.6, 1.3, 9.5, 1.4,
            font_size=72, bold=True, color=WHITE)
add_textbox(slide, "Grounded Activity Question-Answering", 0.6, 2.7, 10.0, 0.8,
            font_size=30, bold=False, color=ACCENT_CYAN)
add_textbox(slide, "over Wearable Sensor Data", 0.6, 3.3, 8.0, 0.7,
            font_size=28, color=MID_GRAY)

# horizontal rule
add_rect(slide, 0.6, 4.15, 5.5, 0.04, ACCENT_BLUE)

add_textbox(slide, "Challenge 1 — Mobile Computing Project", 0.6, 4.35, 9.0, 0.55,
            font_size=18, color=MID_GRAY)
add_textbox(slide, "Datasets: ExtraSensory · PAMAP2 · SHL Preview", 0.6, 4.9, 9.0, 0.5,
            font_size=16, color=MID_GRAY)

# bottom bar
add_rect(slide, 0, 7.3, 13.33, 0.2, ACCENT_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 2 — Table of Contents
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(BLANK)
set_bg(slide, DARK_NAVY)
add_rect(slide, 0, 0, 13.33, 0.08, ACCENT_BLUE)

add_textbox(slide, "Presentation Overview", 0.5, 0.25, 12.0, 0.7,
            font_size=32, bold=True, color=WHITE)
add_rect(slide, 0.5, 1.05, 8.5, 0.04, ACCENT_BLUE)

sections = [
    ("01", "Problem Statement & Motivation"),
    ("02", "Datasets & Target Activities"),
    ("03", "Research Foundation — 4 Papers"),
    ("04", "System Architecture"),
    ("05", "Model Design & Training"),
    ("06", "Three-Stage QA Pipeline"),
    ("07", "Location & Context Extension"),
    ("08", "Edge & Ubiquitous Deployment"),
    ("09", "Benchmark Results"),
    ("10", "Web Application Demo"),
    ("11", "Key Design Decisions"),
    ("12", "Conclusions & Future Work"),
]

col_w = 5.8
for i, (num, title) in enumerate(sections):
    col = i % 2
    row = i // 2
    x = 0.5 + col * col_w
    y = 1.3 + row * 0.72
    add_rect(slide, x, y, col_w - 0.2, 0.58,
             RGBColor(0x1A, 0x28, 0x3D), ACCENT_BLUE)
    add_textbox(slide, num, x + 0.08, y + 0.05, 0.6, 0.5,
                font_size=18, bold=True, color=ACCENT_CYAN)
    add_textbox(slide, title, x + 0.7, y + 0.08, col_w - 1.0, 0.45,
                font_size=15, color=WHITE)

add_rect(slide, 0, 7.3, 13.33, 0.2, ACCENT_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 3 — Section divider: Problem Statement
# ─────────────────────────────────────────────────────────────────────────────
make_section_slide(prs, "Problem Statement & Motivation",
                   "What are we solving and why does it matter?", "01")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 4 — Problem: what is SensorQA?
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(BLANK)
set_bg(slide, DARK_NAVY)
add_rect(slide, 0, 0, 13.33, 0.08, ACCENT_BLUE)
add_textbox(slide, "The Core Problem", 0.5, 0.2, 12.0, 0.7,
            font_size=32, bold=True, color=WHITE)
add_rect(slide, 0.5, 0.95, 8.0, 0.04, ACCENT_BLUE)

# left panel: problem description
add_rect(slide, 0.4, 1.15, 6.2, 4.8, DARK_GRAY)
add_textbox(slide, "Traditional Approach (Insufficient)", 0.6, 1.25, 5.8, 0.5,
            font_size=17, bold=True, color=ORANGE)
add_bullets(slide, [
    "❌  Standard classifiers only output: 'walking', 'running'",
    "❌  Cannot answer: 'How long was I walking?'",
    "❌  Cannot answer: 'Was I sitting in a car?'",
    "❌  No temporal grounding or evidence attribution",
    "❌  Hallucinate answers for unsupported queries",
], 0.6, 1.8, 5.8, 3.5, font_size=15, color=LIGHT_GRAY)

# right panel: our approach
add_rect(slide, 6.8, 1.15, 6.1, 4.8, RGBColor(0x0A, 0x2A, 0x18))
add_textbox(slide, "SensorQA Approach (Our Solution)", 7.0, 1.25, 5.7, 0.5,
            font_size=17, bold=True, color=GREEN)
add_bullets(slide, [
    "✅  Natural language → grounded structured answer",
    "✅  Exact timestamps + sensor channel attribution",
    "✅  Deterministic math (no LLM arithmetic errors)",
    "✅  Truthful rejection of unsupported queries",
    "✅  Context & place-aware reasoning",
], 7.0, 1.8, 5.8, 3.5, font_size=15, color=LIGHT_GRAY)

# bottom formula
add_rect(slide, 1.5, 6.15, 10.3, 0.85, RGBColor(0x10, 0x20, 0x38))
add_textbox(slide,
    "Sensor Recording  +  Natural-Language Question  →  Evidence-Grounded Structured Answer",
    1.7, 6.25, 9.9, 0.6, font_size=16, bold=True, color=ACCENT_CYAN,
    align=PP_ALIGN.CENTER)

add_rect(slide, 0, 7.3, 13.33, 0.2, ACCENT_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 5 — Section divider: Datasets
# ─────────────────────────────────────────────────────────────────────────────
make_section_slide(prs, "Datasets & Target Activities",
                   "Three complementary wearable sensor datasets", "02")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 6 — Datasets overview
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(BLANK)
set_bg(slide, DARK_NAVY)
add_rect(slide, 0, 0, 13.33, 0.08, ACCENT_BLUE)
add_textbox(slide, "Three Complementary Datasets", 0.5, 0.2, 12.0, 0.7,
            font_size=32, bold=True, color=WHITE)

cards = [
    ("ExtraSensory", ACCENT_BLUE,
     "60 users · 377,346 rows\nPrecomputed per-user feature vectors\nActivity, context & place labels\n25 Hz resampled (acc + gyro)\nPrimary benchmark dataset"),
    ("PAMAP2", GREEN,
     "9 subjects · 3,288 windows\nRaw chest-mounted IMU\n100 Hz · 5-sec windows · 2.5-sec hop\nLeave-one-subject-out eval\nAcc: 83.67% · F1: 0.8453"),
    ("SHL Preview", ORANGE,
     "3 participants · 9 recording days\nRaw hand-placement IMU\n100 Hz · 3 shared activities\nParticipant-held-out eval\nAcc: 76.61% · F1: 0.6976"),
]

for i, (name, color, desc) in enumerate(cards):
    x = 0.4 + i * 4.3
    add_rect(slide, x, 1.1, 4.0, 5.5, RGBColor(0x0F, 0x1C, 0x2E))
    add_rect(slide, x, 1.1, 4.0, 0.55, color)
    add_textbox(slide, name, x + 0.15, 1.15, 3.7, 0.45,
                font_size=20, bold=True, color=WHITE)
    y = 1.8
    for line in desc.split('\n'):
        add_textbox(slide, "▸  " + line, x + 0.15, y, 3.7, 0.38,
                    font_size=13.5, color=LIGHT_GRAY)
        y += 0.41

add_rect(slide, 0, 7.3, 13.33, 0.2, ACCENT_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 7 — 7 Target Activities
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(BLANK)
set_bg(slide, DARK_NAVY)
add_rect(slide, 0, 0, 13.33, 0.08, ACCENT_BLUE)
add_textbox(slide, "7 Target Activities / Postures", 0.5, 0.2, 12.0, 0.7,
            font_size=32, bold=True, color=WHITE)
add_rect(slide, 0.5, 0.95, 9.0, 0.04, ACCENT_BLUE)

activities = [
    ("🛌", "Lying Down",        "Stationary / sleep rest",        ACCENT_BLUE),
    ("🪑", "Sitting",           "Sedentary posture",               ACCENT_BLUE),
    ("🧍", "Standing in Place", "Static upright posture",          GREEN),
    ("🚶", "Standing & Moving", "Slow shuffling movement",         GREEN),
    ("🚶‍", "Walking",            "Regular foot locomotion",         ORANGE),
    ("🏃", "Running",           "High-intensity foot locomotion",  RED),
    ("🚴", "Bicycling",         "Wheeled / pedal locomotion",      ACCENT_CYAN),
]

for i, (icon, name, desc, color) in enumerate(activities):
    row = i // 4
    col = i % 4
    if i == 4:  # last row: 3 items → center them
        col = 0
    if row == 1 and i - 4 < 3:
        col = i - 4
        x = 1.5 + col * 3.4
    else:
        x = 0.4 + col * 3.2
    y = 1.25 + row * 2.8
    add_rect(slide, x, y, 3.0, 2.3, RGBColor(0x0F, 0x1C, 0x2E))
    add_rect(slide, x, y, 3.0, 0.08, color)
    add_textbox(slide, icon, x + 0.1, y + 0.15, 0.7, 0.6, font_size=28, color=WHITE)
    add_textbox(slide, name, x + 0.8, y + 0.2, 2.1, 0.5,
                font_size=14, bold=True, color=WHITE)
    add_textbox(slide, desc, x + 0.1, y + 0.85, 2.8, 0.7,
                font_size=12, color=MID_GRAY)

add_rect(slide, 0, 7.3, 13.33, 0.2, ACCENT_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 8 — Section divider: Research Foundation
# ─────────────────────────────────────────────────────────────────────────────
make_section_slide(prs, "Research Foundation",
                   "Four state-of-the-art papers synthesised into SensorQA", "03")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 9 — 4 Papers
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(BLANK)
set_bg(slide, DARK_NAVY)
add_rect(slide, 0, 0, 13.33, 0.08, ACCENT_BLUE)
add_textbox(slide, "Four Foundational Research Papers", 0.5, 0.2, 12.0, 0.7,
            font_size=32, bold=True, color=WHITE)

papers = [
    ("Sensor2Text", ACCENT_BLUE,
     "Multimodal Sensor Representation Learning",
     "Sensor encoder concept · Temporal feature projection\nSemantic window representations · Generalization to unseen users"),
    ("SensorChat", GREEN,
     "Three-Stage QA Architecture (Most Relevant)",
     "Question Decomposition → Evidence Query → Answer Assembly\nCompact context instead of raw long-duration sensor tokens"),
    ("DrHouse", ORANGE,
     "Calibrated Confidence & Truthful Rejection",
     "Uncertainty-aware evidence · Evidence-first reasoning\n100% truthful rejection of unsupported sensor queries"),
    ("JARVIS", ACCENT_CYAN,
     "Structured Query Planning & Deterministic Execution",
     "Rule-based JSON query plan · LLM as semantic compiler only\nDeterministic Python computes ALL numbers"),
]

for i, (name, color, role, detail) in enumerate(papers):
    x = 0.35 + (i % 2) * 6.5
    y = 1.15 + (i // 2) * 2.9
    add_rect(slide, x, y, 6.1, 2.55, RGBColor(0x0F, 0x1C, 0x2E))
    add_rect(slide, x, y, 0.1, 2.55, color)
    add_textbox(slide, name, x + 0.25, y + 0.12, 5.5, 0.48,
                font_size=20, bold=True, color=color)
    add_textbox(slide, role, x + 0.25, y + 0.6, 5.6, 0.48,
                font_size=13, bold=True, color=WHITE)
    for j, line in enumerate(detail.split('\n')):
        add_textbox(slide, "▸  " + line, x + 0.25, y + 1.1 + j * 0.4, 5.6, 0.38,
                    font_size=12, color=LIGHT_GRAY)

add_rect(slide, 0, 7.3, 13.33, 0.2, ACCENT_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 10 — Section divider: System Architecture
# ─────────────────────────────────────────────────────────────────────────────
make_section_slide(prs, "System Architecture",
                   "End-to-end pipeline from raw sensor data to grounded answers", "04")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 11 — Architecture overview
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(BLANK)
set_bg(slide, DARK_NAVY)
add_rect(slide, 0, 0, 13.33, 0.08, ACCENT_BLUE)
add_textbox(slide, "High-Level System Architecture", 0.5, 0.2, 12.0, 0.65,
            font_size=30, bold=True, color=WHITE)

# Pipeline boxes (vertical flow in center)
stages = [
    ("INPUTS", "ExtraSensory Features  |  PAMAP2 Raw IMU  |  SHL Hand IMU",
     ACCENT_BLUE, 1.5, 1.05, 10.3, 0.6),
    ("STEP 1", "Dataset-Specific Data Loading & Feature Extraction",
     DARK_GRAY, 1.5, 1.85, 10.3, 0.6),
    ("STEP 2", "Activity Classification   (SensorEncoderNN + Gradient Boosted Trees)",
     DARK_GRAY, 1.5, 2.65, 10.3, 0.6),
    ("STEP 3", "Evidence Timelines   (Activity · Context · Place · Confidence)",
     DARK_GRAY, 1.5, 3.45, 10.3, 0.6),
    ("STEP 4", "Rule-Based Query Planning  →  Structured QueryPlan JSON",
     DARK_GRAY, 1.5, 4.25, 10.3, 0.6),
    ("STEP 5", "Deterministic Computation Engine   (Duration · Count · Comparison · Transition)",
     GREEN,     1.5, 5.05, 10.3, 0.6),
    ("STEP 6", "Grounded Answer Assembly  →  Mandatory 7-Field Schema",
     ACCENT_BLUE, 1.5, 5.85, 10.3, 0.6),
]

for label, desc, color, x, y, w, h in stages:
    add_rect(slide, x, y, w, h, color)
    add_textbox(slide, label, x + 0.1, y + 0.1, 1.0, 0.45,
                font_size=12, bold=True, color=ACCENT_CYAN)
    add_textbox(slide, desc, x + 1.15, y + 0.1, w - 1.3, 0.45,
                font_size=14, bold=False, color=WHITE)
    if y < 5.85:
        add_textbox(slide, "▼", x + 5.9, y + 0.62, 0.5, 0.3,
                    font_size=14, color=ACCENT_CYAN, align=PP_ALIGN.CENTER)

add_rect(slide, 0, 7.3, 13.33, 0.2, ACCENT_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 12 — Section divider: Models
# ─────────────────────────────────────────────────────────────────────────────
make_section_slide(prs, "Model Design & Training",
                   "Dual-architecture sensor understanding pipeline", "05")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 13 — Model details
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(BLANK)
set_bg(slide, DARK_NAVY)
add_rect(slide, 0, 0, 13.33, 0.08, ACCENT_BLUE)
add_textbox(slide, "Dual-Architecture Sensor Model", 0.5, 0.2, 12.0, 0.65,
            font_size=30, bold=True, color=WHITE)

# Neural Net card
add_rect(slide, 0.4, 1.0, 5.8, 5.8, RGBColor(0x0A, 0x1E, 0x3A))
add_rect(slide, 0.4, 1.0, 5.8, 0.5, ACCENT_BLUE)
add_textbox(slide, "SensorEncoderNN  (PyTorch)", 0.55, 1.08, 5.5, 0.38,
            font_size=16, bold=True, color=WHITE)
nn_items = [
    "Multi-layer projection network",
    "Batch normalization after each layer",
    "GELU activation functions",
    "Learnable temperature scaling T",
    "Prevents overconfident predictions",
    "Outputs calibrated probability vector",
    "Supports INT8 dynamic quantization",
    "Edge-ready: 0.041 MB (41 KB)",
]
for j, item in enumerate(nn_items):
    add_textbox(slide, "▸  " + item, 0.55, 1.65 + j * 0.52, 5.5, 0.45,
                font_size=13, color=LIGHT_GRAY)

# Tree card
add_rect(slide, 6.5, 1.0, 6.4, 5.8, RGBColor(0x08, 0x22, 0x14))
add_rect(slide, 6.5, 1.0, 6.4, 0.5, GREEN)
add_textbox(slide, "HistGradientBoosting Ensemble  (scikit-learn)", 6.65, 1.08, 6.0, 0.38,
            font_size=16, bold=True, color=WHITE)
tree_items = [
    "Fast gradient-boosted decision trees",
    "Robust on tabular feature vectors",
    "Handles class imbalance gracefully",
    "Separately trained per dataset",
    "ExtraSensory: 7-class activity labels",
    "PAMAP2: 6 directly mapped activities",
    "SHL: 3 shared-taxonomy activities",
    "Context-conditioned variant (+8.8% F1)",
]
for j, item in enumerate(tree_items):
    add_textbox(slide, "▸  " + item, 6.65, 1.65 + j * 0.52, 6.1, 0.45,
                font_size=13, color=LIGHT_GRAY)

add_rect(slide, 0, 7.3, 13.33, 0.2, ACCENT_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 14 — Section divider: QA Pipeline
# ─────────────────────────────────────────────────────────────────────────────
make_section_slide(prs, "Three-Stage QA Pipeline",
                   "Query Planning → Deterministic Computation → Grounded Answer", "06")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 15 — QA Pipeline detail
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(BLANK)
set_bg(slide, DARK_NAVY)
add_rect(slide, 0, 0, 13.33, 0.08, ACCENT_BLUE)
add_textbox(slide, "Three-Stage QA Pipeline — End-to-End Example", 0.5, 0.2, 12.0, 0.65,
            font_size=28, bold=True, color=WHITE)

add_textbox(slide, '💬  User asks:  "How long did the user walk?"',
            0.5, 1.0, 12.3, 0.6, font_size=18, bold=True, color=ACCENT_CYAN)

# Stage 1
add_rect(slide, 0.4, 1.8, 3.7, 4.0, RGBColor(0x0A, 0x1E, 0x3A))
add_rect(slide, 0.4, 1.8, 3.7, 0.45, ACCENT_BLUE)
add_textbox(slide, "Stage 1 — Query Planning", 0.55, 1.85, 3.4, 0.38,
            font_size=14, bold=True, color=WHITE)
add_textbox(slide,
    '{\n  "operation": "duration",\n  "target_activity":\n    "walking",\n  "time_bounds": null,\n  "is_supported": true\n}',
    0.55, 2.4, 3.5, 2.5, font_size=12, color=ACCENT_CYAN)
add_textbox(slide, "Rule-based parser (no LLM math)", 0.55, 5.1, 3.5, 0.5,
            font_size=11, italic=True, color=MID_GRAY)

add_textbox(slide, "→", 4.2, 3.5, 0.5, 0.6, font_size=24, bold=True,
            color=ACCENT_BLUE, align=PP_ALIGN.CENTER)

# Stage 2
add_rect(slide, 4.8, 1.8, 3.8, 4.0, RGBColor(0x08, 0x22, 0x14))
add_rect(slide, 4.8, 1.8, 3.8, 0.45, GREEN)
add_textbox(slide, "Stage 2 — Deterministic Compute", 4.95, 1.85, 3.5, 0.38,
            font_size=14, bold=True, color=WHITE)
add_bullets(slide, [
    "Intervals found:",
    "  905 → 1420 s  (515 s)",
    "  2110 → 2295 s  (185 s)",
    "",
    "Σ = 515 + 185 = 700 s",
    "",
    "Zero LLM arithmetic ✅",
], 4.95, 2.4, 3.5, 3.0, font_size=13, color=LIGHT_GRAY)

add_textbox(slide, "→", 8.7, 3.5, 0.5, 0.6, font_size=24, bold=True,
            color=ACCENT_BLUE, align=PP_ALIGN.CENTER)

# Stage 3
add_rect(slide, 9.25, 1.8, 3.6, 4.0, RGBColor(0x1A, 0x18, 0x05))
add_rect(slide, 9.25, 1.8, 3.6, 0.45, ORANGE)
add_textbox(slide, "Stage 3 — Grounded Answer", 9.4, 1.85, 3.3, 0.38,
            font_size=14, bold=True, color=WHITE)
add_bullets(slide, [
    "Answer: 700 seconds",
    "Activity/Event: Walking",
    "Evidence: 2 intervals",
    "Timestamps: 905–1420 s,",
    "  2110–2295 s",
    "Sensor Modality:",
    "  Acc + Gyro",
    "Channels: X Y Z (×2)",
], 9.4, 2.4, 3.4, 3.0, font_size=12, color=LIGHT_GRAY)

add_rect(slide, 0, 7.3, 13.33, 0.2, ACCENT_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 16 — Section divider: Context & Location
# ─────────────────────────────────────────────────────────────────────────────
make_section_slide(prs, "Location & Context Extension",
                   "Activity QA enriched with place and contextual annotations", "07")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 17 — Context & Location detail
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(BLANK)
set_bg(slide, DARK_NAVY)
add_rect(slide, 0, 0, 13.33, 0.08, ACCENT_BLUE)
add_textbox(slide, "Context & Location-Aware Queries", 0.5, 0.2, 12.0, 0.65,
            font_size=30, bold=True, color=WHITE)

# Supported questions
add_rect(slide, 0.4, 1.05, 5.8, 5.9, RGBColor(0x0A, 0x1E, 0x3A))
add_rect(slide, 0.4, 1.05, 5.8, 0.5, ACCENT_BLUE)
add_textbox(slide, "Supported Context/Place Queries", 0.55, 1.12, 5.5, 0.38,
            font_size=16, bold=True, color=WHITE)
examples = [
    "⏱  \"How long was I sitting in a car?\"",
    "🏋️  \"How long was I at the gym?\"",
    "💼  \"Did I walk while at work?\"",
    "🏠→💼  \"How many times did I go from home to work?\"",
    "🏖️  \"What activities did I do at the beach?\"",
]
for j, ex in enumerate(examples):
    add_textbox(slide, ex, 0.55, 1.72 + j * 0.7, 5.5, 0.58,
                font_size=13.5, color=LIGHT_GRAY)

# Data / coverage
add_rect(slide, 6.5, 1.05, 6.4, 5.9, RGBColor(0x0F, 0x1C, 0x2E))
add_rect(slide, 6.5, 1.05, 6.4, 0.5, GREEN)
add_textbox(slide, "Data Coverage & Honest Reporting", 6.65, 1.12, 6.0, 0.38,
            font_size=16, bold=True, color=WHITE)
coverage = [
    "377,346 rows · 60 users audited",
    "LOC_home positive: 57 users",
    "AT_THE_GYM positive: 8 users",
    "LOC_main_workplace: 32 users",
    "OR_indoors: 59 users",
    "Unknown ≠ Negative (honest handling)",
    "Incomplete coverage → lower-bound report",
    "No raw GPS · No city/address queries",
    "Place labels only (coarse granularity)",
]
for j, item in enumerate(coverage):
    add_textbox(slide, "▸  " + item, 6.65, 1.72 + j * 0.55, 6.0, 0.48,
                font_size=13, color=LIGHT_GRAY)

add_rect(slide, 0, 7.3, 13.33, 0.2, ACCENT_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 18 — Section divider: Edge Deployment
# ─────────────────────────────────────────────────────────────────────────────
make_section_slide(prs, "Edge & Ubiquitous Deployment",
                   "Model compression for smartwatches, ESP32 & microcontrollers", "08")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 19 — Edge profiling
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(BLANK)
set_bg(slide, DARK_NAVY)
add_rect(slide, 0, 0, 13.33, 0.08, ACCENT_BLUE)
add_textbox(slide, "Edge Deployment Profiling", 0.5, 0.2, 12.0, 0.65,
            font_size=30, bold=True, color=WHITE)

# Model A
add_rect(slide, 0.4, 1.1, 5.8, 5.6, RGBColor(0x0A, 0x1E, 0x3A))
add_rect(slide, 0.4, 1.1, 5.8, 0.55, ACCENT_BLUE)
add_textbox(slide, "Model A — Server / Cloud  (FP32)", 0.55, 1.18, 5.5, 0.4,
            font_size=16, bold=True, color=WHITE)
ma = [
    ("Parameters",   "192,520"),
    ("Model Size",   "0.735 MB"),
    ("CPU Latency",  "0.698 ms / window"),
    ("Throughput",   "1,433 windows/sec"),
    ("RAM Usage",    "3.04 MB"),
    ("Use Case",     "Batch processing & offline analytics"),
]
for j, (k, v) in enumerate(ma):
    add_textbox(slide, k + ":", 0.55, 1.85 + j * 0.63, 1.9, 0.5,
                font_size=13, bold=True, color=MID_GRAY)
    add_textbox(slide, v, 2.6, 1.85 + j * 0.63, 3.5, 0.5,
                font_size=14, bold=True, color=WHITE)

# Model B
add_rect(slide, 6.7, 1.1, 6.2, 5.6, RGBColor(0x08, 0x22, 0x14))
add_rect(slide, 6.7, 1.1, 6.2, 0.55, GREEN)
add_textbox(slide, "Model B — Edge / Wearable  (INT8)", 6.85, 1.18, 5.9, 0.4,
            font_size=16, bold=True, color=WHITE)
mb = [
    ("Parameters",         "192,520"),
    ("Model Size",         "0.206 MB  (3.6× smaller)"),
    ("CPU Latency",        "1.12 ms / window"),
    ("Throughput",         "893 windows/sec"),
    ("RAM Footprint",      "0.87 MB"),
    ("Compression Ratio",  "3.6×"),
    ("Targets",            "WearOS · Apple Watch · ESP32"),
]
for j, (k, v) in enumerate(mb):
    add_textbox(slide, k + ":", 6.85, 1.85 + j * 0.6, 2.3, 0.5,
                font_size=13, bold=True, color=MID_GRAY)
    add_textbox(slide, v, 9.25, 1.85 + j * 0.6, 3.5, 0.5,
                font_size=14, bold=True, color=WHITE)

add_rect(slide, 0, 7.3, 13.33, 0.2, ACCENT_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 20 — Section divider: Benchmark Results
# ─────────────────────────────────────────────────────────────────────────────
make_section_slide(prs, "Benchmark Results",
                   "Performance across all four challenge tasks", "09")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 21 — Results dashboard
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(BLANK)
set_bg(slide, DARK_NAVY)
add_rect(slide, 0, 0, 13.33, 0.08, ACCENT_BLUE)
add_textbox(slide, "4-Task Benchmark Dashboard", 0.5, 0.2, 12.0, 0.65,
            font_size=30, bold=True, color=WHITE)

tasks = [
    ("Task 1", "Activity Identification", ACCENT_BLUE,
     [("Accuracy",      "64.98%"),
      ("Weighted F1",   "65.58%"),
      ("Macro F1",      "0.4286"),
      ("Test Windows",  "55,004"),
      ("Best Class",    "Lying Down  F1=0.77"),
      ("Hardest Class", "Standing & Moving")]),
    ("Task 2", "Temporal & Quantitative Reasoning", GREEN,
     [("Duration MAE",       "566.75 s"),
      ("Count Exact Acc.",   "31.58%"),
      ("Comparison Acc.",    "33.33%"),
      ("MAPE",               "61.08%"),
      ("Queries Evaluated",  "126"),
      ("Within 30s Acc.",    "7.02%")]),
    ("Task 3", "Evidence Grounding", ORANGE,
     [("Modality Precision", "100%  ✅"),
      ("Channel Precision",  "100%  ✅"),
      ("Mean Temporal IoU",  "0.1812"),
      ("Grounding Samples",  "56"),
      ("Timestamps",         "Always provided"),
      ("Schema Fields",      "7/7 always present")]),
    ("Task 4", "Open-World & Safety", ACCENT_CYAN,
     [("Semantic Acc.",      "100%  ✅"),
      ("Rejection Rate",     "100%  ✅"),
      ("Open-World Tests",   "11 queries"),
      ("Supported Queries",  "6 answered"),
      ("Rejected Queries",   "5 truthfully"),
      ("Hallucination",      "0  ✅")]),
]

for i, (tid, tname, color, metrics) in enumerate(tasks):
    x = 0.35 + (i % 2) * 6.5
    y = 1.1 + (i // 2) * 3.1
    add_rect(slide, x, y, 6.1, 2.7, RGBColor(0x0F, 0x1C, 0x2E))
    add_rect(slide, x, y, 6.1, 0.5, color)
    add_textbox(slide, f"{tid}  —  {tname}", x + 0.15, y + 0.08, 5.8, 0.4,
                font_size=15, bold=True, color=WHITE)
    for j, (k, v) in enumerate(metrics):
        col = j % 2
        row = j // 2
        mx = x + 0.2 + col * 3.0
        my = y + 0.68 + row * 0.63
        add_textbox(slide, k + ":", mx, my, 1.6, 0.45,
                    font_size=11, color=MID_GRAY)
        add_textbox(slide, v, mx + 1.6, my, 1.2, 0.45,
                    font_size=12, bold=True, color=WHITE)

add_rect(slide, 0, 7.3, 13.33, 0.2, ACCENT_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 22 — PAMAP2 & SHL Results
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(BLANK)
set_bg(slide, DARK_NAVY)
add_rect(slide, 0, 0, 13.33, 0.08, ACCENT_BLUE)
add_textbox(slide, "Raw-Signal Dataset Results", 0.5, 0.2, 12.0, 0.65,
            font_size=30, bold=True, color=WHITE)

# PAMAP2
add_rect(slide, 0.4, 1.1, 5.8, 5.6, RGBColor(0x0A, 0x1E, 0x3A))
add_rect(slide, 0.4, 1.1, 5.8, 0.5, GREEN)
add_textbox(slide, "PAMAP2 Chest-IMU Results", 0.55, 1.18, 5.5, 0.38,
            font_size=17, bold=True, color=WHITE)
pamap = [
    ("Pooled Accuracy",     "83.67%"),
    ("Macro F1",            "0.8453"),
    ("Windows Evaluated",   "3,288"),
    ("Subjects",            "8 (LOSO eval)"),
    ("Window Size",         "5-sec · 100 Hz"),
    ("Hop Length",          "2.5 seconds"),
    ("Best Subject Acc.",   "97.95%"),
    ("Worst Subject Acc.",  "43.74%"),
]
for j, (k, v) in enumerate(pamap):
    add_textbox(slide, k + ":", 0.55, 1.78 + j * 0.56, 2.3, 0.48,
                font_size=13, color=MID_GRAY)
    add_textbox(slide, v, 3.0, 1.78 + j * 0.56, 2.9, 0.48,
                font_size=14, bold=True, color=WHITE)

# SHL
add_rect(slide, 6.7, 1.1, 6.2, 5.6, RGBColor(0x1A, 0x18, 0x05))
add_rect(slide, 6.7, 1.1, 6.2, 0.5, ORANGE)
add_textbox(slide, "SHL Hand-IMU Results (Preview)", 6.85, 1.18, 5.9, 0.38,
            font_size=17, bold=True, color=WHITE)
shl = [
    ("Pooled Accuracy",      "76.61%"),
    ("Macro F1",             "0.6976"),
    ("Windows Evaluated",    "21,940"),
    ("Participants",         "3 (participant-held-out)"),
    ("Window Size",          "5-sec · 100 Hz"),
    ("Activities Mapped",    "3 (walk · run · bike)"),
    ("Best User Acc.",       "81.66%"),
    ("Worst User Acc.",      "73.41%"),
]
for j, (k, v) in enumerate(shl):
    add_textbox(slide, k + ":", 6.85, 1.78 + j * 0.56, 2.5, 0.48,
                font_size=13, color=MID_GRAY)
    add_textbox(slide, v, 9.5, 1.78 + j * 0.56, 3.2, 0.48,
                font_size=14, bold=True, color=WHITE)

add_rect(slide, 0, 7.3, 13.33, 0.2, ACCENT_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 23 — Section divider: Web App
# ─────────────────────────────────────────────────────────────────────────────
make_section_slide(prs, "Interactive Web Application",
                   "FastAPI backend · Real-time QA · Benchmark dashboard", "10")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 24 — Web App overview
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(BLANK)
set_bg(slide, DARK_NAVY)
add_rect(slide, 0, 0, 13.33, 0.08, ACCENT_BLUE)
add_textbox(slide, "Web Application — Feature Overview", 0.5, 0.2, 12.0, 0.65,
            font_size=30, bold=True, color=WHITE)

features = [
    ("💬", "SensorChat Assistant",
     "Interactive QA with ExtraSensory, PAMAP2 & SHL timelines. Upload 6-axis CSV for live classification."),
    ("🧭", "Context & Place QA",
     "Query context annotations: 'How long at gym?' · 'Did I walk at work?' via REST API."),
    ("🔬", "Research Theory Hub",
     "Literature synthesis, architecture specs, and paper references fully integrated in UI."),
    ("📊", "Benchmark Dashboard",
     "Full 4-task benchmark results, per-class F1, edge model profiling side-by-side."),
    ("⚡", "Edge Model Profiling",
     "FP32 vs INT8 latency, throughput, model size comparison. Quantized model at 41 KB."),
    ("🤖", "NVIDIA NIM Integration",
     "Optional LLaMA-3.2 narrative verbalization for activity answers. Key set at runtime only."),
]

for i, (icon, title, desc) in enumerate(features):
    row = i // 2
    col = i % 2
    x = 0.4 + col * 6.5
    y = 1.15 + row * 2.0
    add_rect(slide, x, y, 6.1, 1.75, RGBColor(0x0F, 0x1C, 0x2E))
    add_textbox(slide, icon, x + 0.12, y + 0.15, 0.6, 0.6, font_size=22, color=WHITE)
    add_textbox(slide, title, x + 0.8, y + 0.18, 5.0, 0.45,
                font_size=15, bold=True, color=ACCENT_CYAN)
    add_textbox(slide, desc, x + 0.15, y + 0.7, 5.7, 0.85,
                font_size=12.5, color=LIGHT_GRAY)

add_textbox(slide, "Launch: py -3 -m uvicorn src.web_app:app --host 127.0.0.1 --port 8000",
            0.5, 7.05, 12.3, 0.3, font_size=12, color=MID_GRAY, italic=True)
add_rect(slide, 0, 7.3, 13.33, 0.2, ACCENT_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 25 — Section divider: Design Decisions
# ─────────────────────────────────────────────────────────────────────────────
make_section_slide(prs, "Key Design Decisions",
                   "Why we chose this architecture over alternatives", "11")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 26 — Design Justification table
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(BLANK)
set_bg(slide, DARK_NAVY)
add_rect(slide, 0, 0, 13.33, 0.08, ACCENT_BLUE)
add_textbox(slide, "Architecture Decisions — Why Not Alternatives?", 0.5, 0.2, 12.0, 0.65,
            font_size=28, bold=True, color=WHITE)

alternatives = [
    ("❌ Monolithic Multimodal LLM",
     "Raw sensor stream = 540K tokens/hr. Context explosion. LLM arithmetic errors.",
     "✅ Evidence timelines + deterministic Python computation"),
    ("❌ Pure Classification Pipeline",
     "No natural language. Can't answer 'How long?' or handle multi-step comparisons.",
     "✅ Rule-based query planner maps NL → structured operations"),
    ("❌ LLM-Based Internal Math",
     "LLMs drift on subtraction, overlap merging, transition counting.",
     "✅ 100% Python deterministic code for all calculations"),
    ("❌ Unguarded Open-Domain QA",
     "Fabricates heart rate, food, GPS even when sensor is absent.",
     "✅ 100% truthful rejection rate via DrHouse guardrails"),
]

for i, (alt, problem, solution) in enumerate(alternatives):
    y = 1.1 + i * 1.5
    add_rect(slide, 0.3, y, 12.7, 1.3, RGBColor(0x0F, 0x1C, 0x2E))
    add_textbox(slide, alt, 0.5, y + 0.05, 3.8, 0.45,
                font_size=14, bold=True, color=RED)
    add_textbox(slide, problem, 0.5, y + 0.55, 5.8, 0.55,
                font_size=12, color=LIGHT_GRAY)
    add_textbox(slide, solution, 6.7, y + 0.3, 6.2, 0.65,
                font_size=13, bold=True, color=GREEN)

add_rect(slide, 0, 7.3, 13.33, 0.2, ACCENT_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 27 — Section divider: Conclusion
# ─────────────────────────────────────────────────────────────────────────────
make_section_slide(prs, "Conclusions & Future Work",
                   "Key contributions and the road ahead", "12")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 28 — Conclusions
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(BLANK)
set_bg(slide, DARK_NAVY)
add_rect(slide, 0, 0, 13.33, 0.08, ACCENT_BLUE)
add_textbox(slide, "Conclusions & Future Work", 0.5, 0.2, 12.0, 0.65,
            font_size=30, bold=True, color=WHITE)

add_rect(slide, 0.4, 1.05, 5.8, 5.85, RGBColor(0x0A, 0x1E, 0x3A))
add_rect(slide, 0.4, 1.05, 5.8, 0.5, ACCENT_BLUE)
add_textbox(slide, "Key Contributions", 0.55, 1.12, 5.5, 0.38,
            font_size=17, bold=True, color=WHITE)
contributions = [
    "✅  Evidence-grounded NL-QA for wearable data",
    "✅  Synthesis of 4 research papers into one system",
    "✅  Three separate sensor paths (ExtraSensory / PAMAP2 / SHL)",
    "✅  100% truthful rejection rate (Task 4)",
    "✅  100% modality & channel grounding precision (Task 3)",
    "✅  INT8 quantized edge model (3.6× compression, 41 KB)",
    "✅  Context/place QA extension with honest coverage",
    "✅  Optional NVIDIA NIM for narrative explanation",
    "✅  Mandatory 7-field structured answer schema",
]
for j, item in enumerate(contributions):
    add_textbox(slide, item, 0.55, 1.68 + j * 0.55, 5.5, 0.48,
                font_size=12.5, color=LIGHT_GRAY)

add_rect(slide, 6.5, 1.05, 6.4, 5.85, RGBColor(0x08, 0x22, 0x14))
add_rect(slide, 6.5, 1.05, 6.4, 0.5, GREEN)
add_textbox(slide, "Future Work", 6.65, 1.12, 6.0, 0.38,
            font_size=17, bold=True, color=WHITE)
future = [
    "🔮  Context-conditioned classifier retraining",
    "🔮  Multi-fold benchmark for context QA",
    "🔮  Transport-mode inference from trajectories",
    "🔮  Extend to heart-rate & physiological sensors",
    "🔮  On-device WearOS / Apple Watch deployment",
    "🔮  Multi-user personalisation fine-tuning",
    "🔮  Live streaming sensor inference mode",
    "🔮  GUI-based annotation tool for context labels",
    "🔮  Population-level performance certification",
]
for j, item in enumerate(future):
    add_textbox(slide, item, 6.65, 1.68 + j * 0.55, 6.1, 0.48,
                font_size=12.5, color=LIGHT_GRAY)

add_rect(slide, 0, 7.3, 13.33, 0.2, ACCENT_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 29 — Core Principle (closing statement)
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(BLANK)
set_bg(slide, DARK_NAVY)
add_rect(slide, 0, 0, 13.33, 0.08, ACCENT_BLUE)

add_textbox(slide, "Core Design Principle", 0.5, 0.5, 12.3, 0.7,
            font_size=32, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

add_rect(slide, 1.2, 1.5, 10.9, 3.5, RGBColor(0x0A, 0x1E, 0x3A))
add_rect(slide, 1.2, 1.5, 10.9, 0.08, ACCENT_BLUE)
add_rect(slide, 1.2, 4.92, 10.9, 0.08, ACCENT_BLUE)

add_textbox(slide,
    '"Do not let the language model invent the answer or calculate numbers.\n\n'
    'Let the sensor-processing pipeline discover the physical evidence,\n'
    'let deterministic algorithms perform measurable computations,\n'
    'and let the language model reason over, synthesise, and explain that evidence."',
    1.5, 1.8, 10.3, 2.8,
    font_size=19, italic=True, color=ACCENT_CYAN,
    align=PP_ALIGN.CENTER)

pipeline = "Sensor  →  Activity  →  Evidence  →  Timeline  →  Query  →  Deterministic Reasoning  →  LLM Explanation  →  Final Answer"
add_textbox(slide, pipeline, 0.5, 5.3, 12.3, 0.6,
            font_size=13.5, bold=True, color=MID_GRAY, align=PP_ALIGN.CENTER)

add_textbox(slide, "Thank You", 0.5, 6.2, 12.3, 0.65,
            font_size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_textbox(slide, "Questions & Discussion", 0.5, 6.8, 12.3, 0.4,
            font_size=18, color=MID_GRAY, align=PP_ALIGN.CENTER)

add_rect(slide, 0, 7.3, 13.33, 0.2, ACCENT_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
out_path = r"e:\MC\SensorQA_Presentation.pptx"
prs.save(out_path)
print(f"Saved: {out_path}")
print(f"Total slides: {len(prs.slides)}")
