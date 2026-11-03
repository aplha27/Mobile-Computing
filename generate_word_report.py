"""
generate_word_report.py
=======================
Generates a professional, comprehensive Word Document (.docx) report
for Challenge 1: Grounded Activity Question-Answering (SensorQA).
Uses Python standard library zipfile and XML to produce a native, fully styled DOCX file.
"""

import os
import zipfile
import json
from datetime import datetime

DOCX_PATH = "SensorQA_Challenge_1_Comprehensive_Project_Report.docx"


def escape_xml(text):
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


class DocxBuilder:
    def __init__(self):
        self.body_xml = []

    def add_title(self, title_text, subtitle_text=None):
        xml = f"""
        <w:p>
          <w:pPr>
            <w:pStyle w:val="Title"/>
            <w:jc w:val="center"/>
            <w:spacing w:before="360" w:after="120"/>
          </w:pPr>
          <w:r>
            <w:rPr>
              <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
              <w:b/>
              <w:color w:val="1E3A8A"/>
              <w:sz w:val="52"/>
            </w:rPr>
            <w:t>{escape_xml(title_text)}</w:t>
          </w:r>
        </w:p>
        """
        self.body_xml.append(xml)
        if subtitle_text:
            xml_sub = f"""
            <w:p>
              <w:pPr>
                <w:jc w:val="center"/>
                <w:spacing w:before="0" w:after="360"/>
              </w:pPr>
              <w:r>
                <w:rPr>
                  <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
                  <w:i/>
                  <w:color w:val="4B5563"/>
                  <w:sz w:val="28"/>
                </w:rPr>
                <w:t>{escape_xml(subtitle_text)}</w:t>
              </w:r>
            </w:p>
            """
            self.body_xml.append(xml_sub)

    def add_heading1(self, text):
        xml = f"""
        <w:p>
          <w:pPr>
            <w:pStyle w:val="Heading1"/>
            <w:spacing w:before="360" w:after="120"/>
          </w:pPr>
          <w:r>
            <w:rPr>
              <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
              <w:b/>
              <w:color w:val="1E3A8A"/>
              <w:sz w:val="36"/>
            </w:rPr>
            <w:t>{escape_xml(text)}</w:t>
          </w:r>
        </w:p>
        """
        self.body_xml.append(xml)

    def add_heading2(self, text):
        xml = f"""
        <w:p>
          <w:pPr>
            <w:pStyle w:val="Heading2"/>
            <w:spacing w:before="240" w:after="80"/>
          </w:pPr>
          <w:r>
            <w:rPr>
              <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
              <w:b/>
              <w:color w:val="2563EB"/>
              <w:sz w:val="28"/>
            </w:rPr>
            <w:t>{escape_xml(text)}</w:t>
          </w:r>
        </w:p>
        """
        self.body_xml.append(xml)

    def add_heading3(self, text):
        xml = f"""
        <w:p>
          <w:pPr>
            <w:pStyle w:val="Heading3"/>
            <w:spacing w:before="180" w:after="60"/>
          </w:pPr>
          <w:r>
            <w:rPr>
              <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
              <w:b/>
              <w:color w:val="374151"/>
              <w:sz w:val="24"/>
            </w:rPr>
            <w:t>{escape_xml(text)}</w:t>
          </w:r>
        </w:p>
        """
        self.body_xml.append(xml)

    def add_paragraph(self, text, bold=False, italic=False, color="1F2937", space_after=120):
        b_tag = "<w:b/>" if bold else ""
        i_tag = "<w:i/>" if italic else ""
        xml = f"""
        <w:p>
          <w:pPr>
            <w:spacing w:before="0" w:after="{space_after}"/>
            <w:jc w:val="both"/>
          </w:pPr>
          <w:r>
            <w:rPr>
              <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
              {b_tag}
              {i_tag}
              <w:color w:val="{color}"/>
              <w:sz w:val="22"/>
            </w:rPr>
            <w:t xml:space="preserve">{escape_xml(text)}</w:t>
          </w:r>
        </w:p>
        """
        self.body_xml.append(xml)

    def add_bullet(self, text, bold_prefix="", level=0):
        xml = f"""
        <w:p>
          <w:pPr>
            <w:pStyle w:val="ListBullet"/>
            <w:numPr>
              <w:ilvl w:val="{level}"/>
              <w:numId w:val="1"/>
            </w:numPr>
            <w:spacing w:before="40" w:after="60"/>
          </w:pPr>
        """
        if bold_prefix:
            xml += f"""
            <w:r>
              <w:rPr>
                <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
                <w:b/>
                <w:color w:val="111827"/>
                <w:sz w:val="22"/>
              </w:rPr>
              <w:t xml:space="preserve">{escape_xml(bold_prefix)} </w:t>
            </w:r>
            """
        xml += f"""
          <w:r>
            <w:rPr>
              <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
              <w:color w:val="374151"/>
              <w:sz w:val="22"/>
            </w:rPr>
            <w:t xml:space="preserve">{escape_xml(text)}</w:t>
          </w:r>
        </w:p>
        """
        self.body_xml.append(xml)

    def add_callout(self, text, label="CORE DESIGN PRINCIPLE"):
        xml = f"""
        <w:tbl>
          <w:tblPr>
            <w:tblW w:w="9200" w:type="dxa"/>
            <w:tblBorders>
              <w:top w:val="none"/>
              <w:left w:val="single" w:sz="24" w:space="0" w:color="3B82F6"/>
              <w:bottom w:val="none"/>
              <w:right w:val="none"/>
            </w:tblBorders>
            <w:tblCellMar>
              <w:top w:w="120" w:type="dxa"/>
              <w:left w:w="240" w:type="dxa"/>
              <w:bottom w:w="120" w:type="dxa"/>
              <w:right w:w="240" w:type="dxa"/>
            </w:tblCellMar>
          </w:tblPr>
          <w:tr>
            <w:tc>
              <w:tcPr>
                <w:tcW w:w="9200" w:type="dxa"/>
                <w:shd w:val="clear" w:color="auto" w:fill="EFF6FF"/>
              </w:tcPr>
              <w:p>
                <w:pPr><w:spacing w:before="60" w:after="40"/></w:pPr>
                <w:r>
                  <w:rPr>
                    <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
                    <w:b/>
                    <w:color w:val="1E40AF"/>
                    <w:sz w:val="20"/>
                  </w:rPr>
                  <w:t>{escape_xml(label)}</w:t>
                </w:r>
              </w:p>
              <w:p>
                <w:pPr><w:spacing w:before="0" w:after="60"/></w:pPr>
                <w:r>
                  <w:rPr>
                    <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
                    <w:i/>
                    <w:color w:val="1E3A8A"/>
                    <w:sz w:val="22"/>
                  </w:rPr>
                  <w:t>{escape_xml(text)}</w:t>
                </w:r>
              </w:p>
            </w:tc>
          </w:tr>
        </w:tbl>
        <w:p><w:pPr><w:spacing w:before="0" w:after="120"/></w:pPr></w:p>
        """
        self.body_xml.append(xml)

    def add_table(self, headers, rows):
        total_w = 9300
        num_cols = len(headers)
        col_w = int(total_w / num_cols)

        xml = f"""
        <w:tbl>
          <w:tblPr>
            <w:tblW w:w="{total_w}" w:type="dxa"/>
            <w:tblBorders>
              <w:top w:val="single" w:sz="6" w:space="0" w:color="D1D5DB"/>
              <w:left w:val="single" w:sz="6" w:space="0" w:color="D1D5DB"/>
              <w:bottom w:val="single" w:sz="6" w:space="0" w:color="D1D5DB"/>
              <w:right w:val="single" w:sz="6" w:space="0" w:color="D1D5DB"/>
              <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E5E7EB"/>
              <w:insideV w:val="single" w:sz="4" w:space="0" w:color="E5E7EB"/>
            </w:tblBorders>
            <w:tblCellMar>
              <w:top w:w="100" w:type="dxa"/>
              <w:left w:w="140" w:type="dxa"/>
              <w:bottom w:w="100" w:type="dxa"/>
              <w:right w:w="140" w:type="dxa"/>
            </w:tblCellMar>
          </w:tblPr>
        """
        # Header row
        xml += '<w:tr><w:trPr><w:tblHeader/></w:trPr>'
        for h in headers:
            xml += f"""
            <w:tc>
              <w:tcPr>
                <w:tcW w:w="{col_w}" w:type="dxa"/>
                <w:shd w:val="clear" w:color="auto" w:fill="1E3A8A"/>
              </w:tcPr>
              <w:p>
                <w:pPr><w:spacing w:before="60" w:after="60"/></w:pPr>
                <w:r>
                  <w:rPr>
                    <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
                    <w:b/>
                    <w:color w:val="FFFFFF"/>
                    <w:sz w:val="20"/>
                  </w:rPr>
                  <w:t>{escape_xml(h)}</w:t>
                </w:r>
              </w:p>
            </w:tc>
            """
        xml += '</w:tr>'

        # Body rows
        for row_idx, r in enumerate(rows):
            fill_color = "F9FAFB" if row_idx % 2 == 1 else "FFFFFF"
            xml += '<w:tr>'
            for cell in r:
                xml += f"""
                <w:tc>
                  <w:tcPr>
                    <w:tcW w:w="{col_w}" w:type="dxa"/>
                    <w:shd w:val="clear" w:color="auto" w:fill="{fill_color}"/>
                  </w:tcPr>
                  <w:p>
                    <w:pPr><w:spacing w:before="40" w:after="40"/></w:pPr>
                    <w:r>
                      <w:rPr>
                        <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
                        <w:color w:val="1F2937"/>
                        <w:sz w:val="20"/>
                      </w:rPr>
                      <w:t>{escape_xml(str(cell))}</w:t>
                    </w:r>
                  </w:p>
                </w:tc>
                """
            xml += '</w:tr>'
        xml += '</w:tbl><w:p><w:pPr><w:spacing w:before="0" w:after="160"/></w:pPr></w:p>'
        self.body_xml.append(xml)

    def add_code_block(self, code_text):
        xml = f"""
        <w:tbl>
          <w:tblPr>
            <w:tblW w:w="9200" w:type="dxa"/>
            <w:tblBorders>
              <w:top w:val="single" w:sz="6" w:space="0" w:color="4B5563"/>
              <w:left w:val="single" w:sz="6" w:space="0" w:color="4B5563"/>
              <w:bottom w:val="single" w:sz="6" w:space="0" w:color="4B5563"/>
              <w:right w:val="single" w:sz="6" w:space="0" w:color="4B5563"/>
            </w:tblBorders>
            <w:tblCellMar>
              <w:top w:w="120" w:type="dxa"/>
              <w:left w:w="180" w:type="dxa"/>
              <w:bottom w:w="120" w:type="dxa"/>
              <w:right w:w="180" w:type="dxa"/>
            </w:tblCellMar>
          </w:tblPr>
          <w:tr>
            <w:tc>
              <w:tcPr>
                <w:tcW w:w="9200" w:type="dxa"/>
                <w:shd w:val="clear" w:color="auto" w:fill="111827"/>
              </w:tcPr>
              <w:p>
                <w:pPr><w:spacing w:before="40" w:after="40"/></w:pPr>
                <w:r>
                  <w:rPr>
                    <w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>
                    <w:color w:val="A7F3D0"/>
                    <w:sz w:val="19"/>
                  </w:rPr>
                  <w:t xml:space="preserve">{escape_xml(code_text)}</w:t>
                </w:r>
              </w:p>
            </w:tc>
          </w:tr>
        </w:tbl>
        <w:p><w:pPr><w:spacing w:before="0" w:after="140"/></w:pPr></w:p>
        """
        self.body_xml.append(xml)

    def add_page_break(self):
        self.body_xml.append(
            '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'
        )

    def save(self, filepath):
        body_content = "\n".join(self.body_xml)
        document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
                    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
          <w:body>
            {body_content}
            <w:sectPr>
              <w:pgSz w:w="12240" w:h="15840"/>
              <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
              <w:cols w:space="720"/>
              <w:docGrid w:linePitch="360"/>
            </w:sectPr>
          </w:body>
        </w:document>
        """

        content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
          <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
          <Default Extension="xml" ContentType="application/xml"/>
          <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
          <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
        </Types>
        """

        rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
          <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
        </Relationships>
        """

        doc_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
          <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
        </Relationships>
        """

        styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:docDefaults>
            <w:rPrDefault>
              <w:rPr>
                <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>
                <w:sz w:val="22"/>
                <w:lang w:val="en-US"/>
              </w:rPr>
            </w:rPrDefault>
          </w:docDefaults>
        </w:styles>
        """

        with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", content_types_xml.strip())
            z.writestr("_rels/.rels", rels_xml.strip())
            z.writestr("word/_rels/document.xml.rels", doc_rels_xml.strip())
            z.writestr("word/document.xml", document_xml.strip())
            z.writestr("word/styles.xml", styles_xml.strip())

        print(f"Native Word Document successfully generated: {filepath}")


def _generate_legacy_report():
    doc = DocxBuilder()

    # Title
    doc.add_title(
        "Grounded Activity Question-Answering (SensorQA)",
        "Challenge 1 — Comprehensive Technical Architecture, Mathematical Foundations & Empirical Benchmarks"
    )

    # Metadata Card
    doc.add_table(
        ["Specification", "Project Value", "Research Context"],
        [
            ["Dataset", "ExtraSensory Dataset (60 Subjects)", "In-the-wild daily living continuous records"],
            ["Sensors", "Acc X/Y/Z + Gyro X/Y/Z @ 25 Hz", "Tri-axial Accelerometer & Gyroscope (6 Channels)"],
            ["Activities", "7 Target Mutually Exclusive Classes", "Lying down, Sitting, Standing, Walking, Running, Bicycling"],
            ["Language Model", "NVIDIA NIM (meta/llama-3.2-11b-vision-instruct)", "Constrained semantic compiler & narrative verbalizer"],
            ["Cross-Validation", "Official cv5Folds (Fold 0: 48 Train / 12 Test)", "55,004 completely unseen test evaluation windows"],
            ["Core Philosophy", "Zero LLM Arithmetic Hallucination", "Sensors find evidence -> Code calculates math -> LLM explains"]
        ]
    )

    # 1. Executive Summary
    doc.add_heading1("1. Executive Summary & Problem Understanding")
    doc.add_paragraph(
        "The objective of Challenge 1 is to build an explainable, evidence-grounded natural-language question-answering "
        "system capable of reasoning over continuous wearable motion sensor streams. The challenge goes fundamentally "
        "beyond simple activity classification: the system must accurately answer temporal, quantitative, comparative, "
        "and open-world questions while providing verifiable mathematical proofs and sensor attribution."
    )
    doc.add_callout(
        "Do not let the language model invent the answer or calculate numbers. Let the physical sensor-processing pipeline "
        "discover evidence, let deterministic algorithms perform measurable computations, and let the language model "
        "reason over, synthesize, and explain that verified evidence.",
        "FUNDAMENTAL DESIGN RULE (CHALLENGE 1)"
    )

    # 2. Dataset & Target Activities
    doc.add_heading1("2. Dataset Architecture & Target Activities")
    doc.add_paragraph(
        "The system is developed and benchmarked on the ExtraSensory dataset, comprising continuous multi-sensor recordings "
        "from 60 human subjects during unconstrained daily life. Signals are resampled to 25 Hz and segmented into 20-second "
        "sliding windows (500 samples per channel)."
    )
    doc.add_heading2("Target Activity Classes & Physical Motion Signatures")
    doc.add_table(
        ["Activity / Posture", "Class ID", "Biomechanical Kinematic Profile", "Sample Count (60 Users)"],
        [
            ["Lying down", "Class 0", "Gravity aligned with Z-axis, near-zero rotational variance", "104,210 (27.6%)"],
            ["Sitting", "Class 1", "Gravity shared between Y & Z axes, sedentary physiological tremor", "136,356 (36.1%)"],
            ["Standing in place", "Class 2", "Upright gravity vector on Y-axis, stationary low variance", "37,743 (10.0%)"],
            ["Standing & moving", "Class 3", "Upright posture with low-frequency shifting / strolling", "806 (0.2%)"],
            ["Walking", "Class 4", "Rhythmic vertical/forward acceleration at 1.8 Hz step cadence", "22,136 (5.9%)"],
            ["Running", "Class 5", "High-intensity impact peaks at 3.0 Hz cadence, high jerk", "1,090 (0.3%)"],
            ["Bicycling", "Class 6", "Smooth rotational gyro velocity, pedaling at 1.3 Hz (78 RPM)", "5,020 (1.3%)"]
        ]
    )

    # 3. Literature Synthesis
    doc.add_heading1("3. Literature Synthesis & Theoretical Foundations")
    doc.add_paragraph(
        "Rather than reproducing a single isolated paper, our system combines the best innovations of four foundational works:"
    )
    doc.add_bullet("Learns temporal-spectral feature projections directly from 25 Hz 6-axis signals.", "Sensor2Text:")
    doc.add_bullet("Establishes a 3-Stage QA Pipeline (Question Decomposition -> Sensor Query -> Answer Assembly) over 20-second ExtraSensory windows.", "SensorChat:")
    doc.add_bullet("Applies temperature scaling for uncertainty calibration and enforces 100% truthful rejection for out-of-scope queries (food, heart rate, GPS).", "DrHouse:")
    doc.add_bullet("Compiles questions into structured JSON plans and delegates all arithmetic (sums, counts, differences) to deterministic code.", "JARVIS:")

    # 4. Two-Level Architecture
    doc.add_heading1("4. Two-Level, Three-Stage System Architecture")
    doc.add_paragraph(
        "To prevent context window explosion and arithmetic hallucinations, the system decouples sensor perception from language reasoning:"
    )
    doc.add_heading2("Model A: Sensor Understanding Pipeline (Perception Layer)")
    doc.add_paragraph(
        "Model A ingests 98 kinematic features per window and executes a dual-model ensemble: a PyTorch Residual Sensor "
        "Encoder (ResSensorNet) with temperature scaling (T=1.25) combined with a regularized Gradient Boosted Tree Ensemble "
        "(HistGradientBoosting). Predictions are smoothed into a continuous Activity Timeline Store."
    )
    doc.add_heading2("Model B: Language Reasoning & Grounding Engine (Cognitive Layer)")
    doc.add_bullet("NVIDIA NIM (meta/llama-3.2-11b-vision-instruct) compiles the user's natural language question into a strictly-typed JSON QueryPlan.", "Stage 1 (Query Decomposition):")
    doc.add_bullet("Deterministic Python algorithms execute mathematical aggregations (durations, counts, comparisons, transitions) over the timeline, producing an immutable EvidencePackage.", "Stage 2 (Deterministic Execution):")
    doc.add_bullet("NVIDIA NIM verbalizes the mathematical evidence trace into a scientific narrative adhering strictly to the mandatory 7-field schema.", "Stage 3 (Structured Answer Assembly):")

    # 5. Mandatory Schema
    doc.add_heading1("5. Mandatory 7-Field Structured Answer Output Specification")
    doc.add_paragraph("Every query response adheres strictly to the Challenge 1 schema specification:")
    doc.add_code_block(
        "Answer: 700 seconds\n\n"
        "Activity/Event: Walking\n\n"
        "Evidence: Walking was detected across 2 distinct intervals.\n"
        "Timestamp(s): 905-1420 s, 2110-2295 s\n\n"
        "Sensor Modality:\n"
        "Accelerometer + Gyroscope\n\n"
        "Sensor Channel(s):\n"
        "Acc X + Acc Y + Acc Z + Gyro X + Gyro Y + Gyro Z\n\n"
        "Explanation:\n"
        "The walking duration of 700 seconds was calculated by combining the accelerometer and gyroscope data "
        "from the 905-1420 s and 2110-2295 s time intervals. This result is derived from the deterministic mathematical trace, "
        "which sums the durations of 515 seconds and 185 seconds, respectively."
    )

    # 6. Mathematical Foundations
    doc.add_heading1("6. Comprehensive Mathematical Formulations")
    doc.add_heading2("A. Signal Processing & Feature Extraction")
    doc.add_paragraph("Linear interpolation maps irregular sensor packets onto a uniform 25 Hz grid:")
    doc.add_paragraph("x(t) = x_k + (t - t_k) * (x_{k+1} - x_k) / (t_{k+1} - t_k),   t_k <= t <= t_{k+1}", italic=True)
    doc.add_paragraph("Kinematic jerk measures motion smoothness (first derivative of acceleration):")
    doc.add_paragraph("j(t) = da(t)/dt ~= (a_{i+1} - a_i) / Delta_t", italic=True)
    doc.add_paragraph("Discrete Fourier Transform (DFT) extracts spectral energy bands across 0-12.5 Hz:")
    doc.add_paragraph("X[k] = Sum_{n=0}^{N-1} x[n] * exp(-j * 2*pi*k*n / N)", italic=True)

    doc.add_heading2("B. Machine Learning & Calibrated Uncertainty")
    doc.add_paragraph("Temperature-scaled softmax calibration:")
    doc.add_paragraph("P(y = c | x) = exp(z_c / T) / Sum_{j=1}^K exp(z_j / T),   T = 1.25", italic=True)
    doc.add_paragraph("Class-weighted cross-entropy loss addressing severe class imbalance:")
    doc.add_paragraph("L_CE = - Sum_{c=1}^K w_c * y_c * log(p_c),   w_c = N_total / (K * N_c)", italic=True)

    doc.add_heading2("C. Deterministic Query Mathematics")
    doc.add_paragraph("Total Activity Duration:   Duration(A) = Sum_{I_k in E_A} (end_k - start_k)")
    doc.add_paragraph("Finite State Transition Counting:   Transitions(A -> B) = Sum_{i=1}^{M-1} I(s_i = A and s_{i+1} = B)")
    doc.add_paragraph("Temporal Intersection-over-Union:   IoU = Measure(I_pred intersect I_gt) / Measure(I_pred union I_gt)")

    # 7. Empirical Benchmarks
    doc.add_heading1("7. Empirical Evaluation & Multi-Task Benchmark Results")
    doc.add_paragraph(
        "The model was evaluated across 55,004 completely unseen test windows collected from 12 unseen test subjects "
        "under the official 5-fold cross-validation protocol (Fold 0)."
    )

    doc.add_heading2("Task 1: Activity Classification Performance (55,004 Unseen Windows)")
    doc.add_table(
        ["Activity Class", "Precision", "Recall", "F1-Score", "Test Support Windows"],
        [
            ["Lying down", "72.50%", "81.77%", "0.7686", "17,191"],
            ["Sitting", "75.83%", "62.48%", "0.6851", "25,473"],
            ["Standing in place", "31.75%", "34.34%", "0.3299", "6,179"],
            ["Standing & moving", "0.00%", "0.00%", "0.0000", "0"],
            ["Walking", "52.27%", "56.44%", "0.5428", "5,691"],
            ["Running", "27.31%", "47.97%", "0.3481", "123"],
            ["Bicycling", "22.45%", "74.06%", "0.3445", "347"],
            ["OVERALL / WEIGHTED", "72.14%", "64.76%", "0.6537", "55,004"]
        ]
    )

    doc.add_heading2("Summary Across All 4 Challenge Tasks")
    doc.add_table(
        ["Challenge Task", "Primary Metric", "Empirical Score", "Verification Notes"],
        [
            ["Task 1: Activity Identification", "Overall Accuracy / Weighted F1", "64.76% / 0.6537", "Cross-subject generalization on 12 unseen users"],
            ["Task 2: Quantitative Reasoning", "Duration MAE / Count Match", "598.4 s / 38.6%", "Zero calculation drift via deterministic engine"],
            ["Task 3: Evidence Grounding", "Modality & Channel Precision", "100.0% / 100.0%", "Exact 6-axis physical attribution"],
            ["Task 4: Open-World Reasoning", "Semantic Question Accuracy", "100.0%", "Accurate mapping of wheeled/sedentary queries"],
            ["Task 4: Truthful Safety Rejection", "Out-of-Scope Rejection Rate", "100.0%", "Zero hallucination on food, heart rate, GPS"]
        ]
    )

    # 8. Edge Deployment
    doc.add_heading1("8. Edge Deployment & Ubiquitous Computing Profiling")
    doc.add_paragraph(
        "The system profiles two operational configurations for cloud servers versus battery-powered wearable edge devices:"
    )
    doc.add_table(
        ["Profiling Metric", "Server Model A (FP32)", "Ubiquitous Edge Model B (INT8)", "Edge Efficiency Gain"],
        [
            ["Model Parameters", "38,536 weights", "38,536 quantized weights", "Identical representation"],
            ["Model Storage Size", "0.147 MB", "0.041 MB (41 KB)", "3.6x Compression Ratio"],
            ["CPU Inference Latency", "0.194 ms / window", "0.539 ms / window", "Sub-millisecond on edge CPU"],
            ["Inference Throughput", "5,150 windows/sec", "1,856 windows/sec", "High real-time concurrency"],
            ["RAM Memory Footprint", "1.57 MB", "0.57 MB (< 600 KB)", "Fits on ARM Cortex / ESP32"]
        ]
    )

    # 9. Conclusion
    doc.add_heading1("9. Conclusion & Operational Status")
    doc.add_paragraph(
        "SensorQA demonstrates that decoupling multimodal sensor perception from natural language reasoning produces "
        "a reliable, hallucination-free question-answering system over wearable sensors. By grounding every response in "
        "calibrated physical evidence and executing all mathematical calculations deterministically, the system achieves "
        "state-of-the-art explainability, 100% truthful safety, and real-time ubiquitous deployment feasibility."
    )

    doc.save(DOCX_PATH)


def generate_full_report():
    """Generate an updated report grounded in the current project artifacts."""
    doc = DocxBuilder()

    doc.add_title(
        "SensorQA",
        "Comprehensive Project Report | Grounded Wearable Activity Question-Answering",
    )
    doc.add_paragraph(
        "Project status: functional research prototype. This report describes the "
        "current implementation and the evaluation artifacts available in the "
        "workspace as of 2 November 2026.",
        bold=True,
        color="1E3A8A",
    )
    doc.add_table(
        ["At a glance", "Current implementation"],
        [
            ["Primary capability", "Evidence-grounded activity and temporal QA"],
            ["Data paths", "ExtraSensory, PAMAP2 chest IMU, SHL Hand IMU, StudentLife"],
            ["Web stack", "FastAPI, Jinja2, HTML, CSS, JavaScript"],
            ["ML stack", "Python, NumPy, pandas, scikit-learn, PyTorch"],
            ["Team structure", "4-member engineering team (equal 25% distribution)"],
            ["Optional language service", "NVIDIA NIM for activity explanation only"],
            ["Local entry point", "http://127.0.0.1:8000"],
        ],
    )
    doc.add_callout(
        "The project was executed by a collaborative engineering team of four members with "
        "equal contribution (25% each). Work was divided into four distinct, interdependent "
        "functional domains: (1) Sensor Perception & Multimodal Modeling, (2) Temporal Evidence "
        "Store & Location/Context Computing, (3) Deterministic Math Engine & Question Planning, and "
        "(4) Full-Stack System Integration, API Services & Evaluation.",
        "PROJECT TEAM & EQUAL WORK ATTRIBUTION",
    )
    doc.add_heading2("Team Roles and Work Distribution (Equal 25% Share)")
    doc.add_table(
        ["Team Member", "Assigned Functional Domain", "Key Responsibilities & Deliverables", "Effort"],
        [
            [
                "Member 1\n(Perception & Modeling Lead)",
                "Sensor Understanding & Model Architecture",
                "• Designed and trained Model A (PyTorch SensorEncoderNN with temperature scaling T).\n"
                "• Implemented HistGradientBoosting tabular classifier for ExtraSensory 7 target classes.\n"
                "• Engineered PAMAP2 chest-IMU 100 Hz windowing and leave-one-subject-out evaluation.\n"
                "• Implemented PyTorch dynamic INT8 quantization achieving 3.6x edge compression (41 KB).",
                "25%"
            ],
            [
                "Member 2\n(Data & Context Lead)",
                "Multimodal Data Ingestion, Context & Location",
                "• Built ExtraSensoryLoader with 60-user train/val/test official fold splitting.\n"
                "• Integrated StudentLife dataset (49 users) with GPS, Wi-Fi location and EMA surveys.\n"
                "• Engineered SHL preview loader (Hand motion stream, contiguous window validation).\n"
                "• Designed ContextTimeline & LocationTimeline with explicit observation fraction metrics.",
                "25%"
            ],
            [
                "Member 3\n(Reasoning & Math Lead)",
                "Query Planning & Deterministic Computation",
                "• Developed rule-based QueryPlanner for intent classification and entity extraction.\n"
                "• Implemented DeterministicEngine (zero LLM arithmetic, 100% precision duration/counts).\n"
                "• Built temporal intersection algorithms for activity ∩ context/location co-occurrence.\n"
                "• Designed DrHouse truthful rejection safety guardrails (100% rejection rate for food/HR/GPS).",
                "25%"
            ],
            [
                "Member 4\n(System & Evaluation Lead)",
                "Full-Stack Integration, Web Services & Evaluation",
                "• Built FastAPI asynchronous backend, REST endpoints and multi-session routing.\n"
                "• Developed interactive front-end (real-time chat, Chart.js waveforms, timeline view).\n"
                "• Automated complete 4-Task benchmark evaluation suite and edge profiling scripts.\n"
                "• Created structured answer assembly enforcing mandatory 7-field schema & NIM explanation client.",
                "25%"
            ]
        ]
    )
    doc.add_heading2("Report contents")
    for item in [
        "1. Executive summary and project goals",
        "2. Team structure and work distribution (4 equal members)",
        "3. Datasets, provenance, and supported evidence (ExtraSensory, PAMAP2, SHL, StudentLife)",
        "4. Architecture and important technologies",
        "5. Model and signal-processing workflows",
        "6. Question answering, evidence, and safeguards",
        "7. Evaluation results and their interpretation",
        "8. Web application, operation, and key handling",
        "9. Limitations, recommendations, and conclusion",
    ]:
        doc.add_bullet(item)
    doc.add_paragraph(
        "This is a technical project report, not a clinical validation, scientific "
        "certification, or claim that the models generalize to every wearable, "
        "placement, or population."
    )

    doc.add_page_break()
    doc.add_heading1("1. Executive Summary and Project Goals")
    doc.add_paragraph(
        "SensorQA is a research prototype for answering natural-language questions "
        "about wearable activity evidence. Its aim extends beyond assigning an "
        "activity label: it builds session timelines and supports queries such as "
        "what activity occurred, how long it lasted, how many intervals were seen, "
        "when it began, whether one activity exceeded another, and whether two "
        "activities transitioned in sequence."
    )
    doc.add_paragraph(
        "The system separates perception from answer calculation. Dataset-specific "
        "models produce activity predictions; timeline components organize "
        "predictions and available annotations; a rule-based planner maps supported "
        "questions to structured operations; and deterministic Python routines "
        "calculate durations, interval intersections, counts, comparisons, and "
        "transitions. The optional NVIDIA NIM integration can rewrite some activity "
        "explanations, but it is not the source of the measurements or arithmetic."
    )
    doc.add_heading2("Current project scope")
    doc.add_bullet(
        "ExtraSensory: precomputed per-user feature and label rows, activity "
        "classification, and sparse observed context/place annotations.",
        "Primary QA path:",
    )
    doc.add_bullet(
        "PAMAP2: a separate model trained and evaluated on raw chest-mounted "
        "accelerometer and gyroscope recordings.",
        "Raw-signal path:",
    )
    doc.add_bullet(
        "SHL preview: a separate model using the Hand-placement IMU stream for "
        "walking, running, and bicycling only.",
        "Additional raw-signal path:",
    )
    doc.add_paragraph(
        "The prototype includes a local web interface, session timelines, measured "
        "waveform views for PAMAP2 and SHL, compatible raw CSV preview and supported "
        "PAMAP2 inference, benchmark artifacts, and automated tests. ExtraSensory "
        "does not contain raw continuous streams in the bundled feature archive, so "
        "its chart waveform is illustrative rather than measured sensor evidence."
    )
    doc.add_heading2("What the project does not claim")
    doc.add_paragraph(
        "The recorded metrics are dataset- and split-specific. The SHL preview has "
        "only three participants, the PAMAP2 evaluation has eight usable protocol "
        "subjects and substantial subject variability, and one ExtraSensory fold "
        "does not establish performance across all settings. The system is not a "
        "medical device and cannot answer questions about food, heart rate, exact "
        "GPS, addresses, audio, phone screens, or other unobserved evidence."
    )

    doc.add_page_break()
    doc.add_heading1("2. Datasets, Provenance, and Supported Evidence")
    doc.add_heading2("ExtraSensory: feature and annotation QA")
    doc.add_paragraph(
        "The ExtraSensory archive is a collection of per-user feature/label rows. "
        "SensorQA uses the dataset's existing feature representation and activity "
        "labels; it does not reconstruct raw accelerometer or gyroscope waveforms "
        "from these rows. Activity targets include lying down, sitting, standing in "
        "place, standing and moving, walking, running, and bicycling. The "
        "context-conditioned classifier can use annotation values together with "
        "explicit observation masks, so a missing annotation is not silently treated "
        "as a negative label."
    )
    doc.add_paragraph(
        "The local archive audit covered 377,346 rows across 60 users. Positive "
        "home labels were present for 57 users and positive gym labels for 8. This "
        "demonstrates sparse annotation coverage; these counts are not GPS fixes or "
        "independent visits. Place outputs are limited to coarse labels. There is "
        "no raw coordinate sequence in this data path."
    )
    doc.add_heading2("PAMAP2: raw chest-IMU sessions")
    doc.add_paragraph(
        "PAMAP2 is integrated as an independent raw-signal pipeline. The protocol "
        "loader uses chest accelerometer and gyroscope channels, processes "
        "approximately 100 Hz recordings in five-second windows with 2.5-second "
        "hops, and maps only direct shared classes: lying, sitting, standing, "
        "walking, running, and cycling. Subjects 101-108 contribute to the "
        "subject-held-out evaluation. Subject 109 is excluded because its protocol "
        "recording contains only rope-jumping labels, outside the mapped classes."
    )
    doc.add_heading2("SHL preview: Hand-placement IMU sessions")
    doc.add_paragraph(
        "Three local SHL preview archives provide nine recording days from three "
        "participants. SensorQA streams the Hand_Motion and Label files directly "
        "from ZIP archives and does not extract the multi-gigabyte data. The "
        "approximately 100 Hz stream is processed in five-second windows with "
        "2.5-second hops. Only walking, running, and bicycling map directly to the "
        "SensorQA labels. Still, car, bus, train, subway, and unlabeled classes are "
        "excluded rather than reinterpreted as unrelated activities."
    )
    doc.add_heading2("StudentLife: Continuous sensing and place semantics")
    doc.add_paragraph(
        "The newly integrated StudentLife dataset (Dartmouth, 49 students, 10-week continuous sensing) "
        "bridges inertial activity with real-world location and context. Loaded directly from "
        "studentlife.zip, it provides native Android ActivityRecognition (sitting, walking, running), "
        "GPS traces with travelstate (stationary vs. moving), Wi-Fi-based indoor localization "
        "(in[building] / near[building]), and Ecological Momentary Assessment (EMA) self-reports "
        "(dining, exercise, study spaces, social interactions). Building names are mapped to "
        "standard place labels (e.g. Baker-Berry Library -> main_workplace, Alumni Gym -> gym, "
        "53 Commons -> restaurant, dormitories -> home), enabling deterministic location duration, "
        "visit counting, and dwell-time query execution."
    )
    doc.add_callout(
        "A model's label support is determined by the dataset and sensor placement. "
        "A PAMAP2 chest model, SHL Hand model, ExtraSensory feature model, and StudentLife "
        "timeline store are strictly separated by provenance to prevent domain leakage.",
        "DATA PROVENANCE & STRICT DATASET ISOLATION",
    )

    doc.add_page_break()
    doc.add_heading1("3. Architecture and Important Technologies")
    doc.add_paragraph(
        "SensorQA is organized as a Python application with dataset-specific "
        "loaders and classifiers, a timeline/evidence layer, a query and computation "
        "layer, and a browser-based interface. Each classifier is trained and "
        "evaluated separately; downstream QA uses the session's corresponding "
        "timeline and observed annotations."
    )
    doc.add_table(
        ["Technology", "Role in SensorQA"],
        [
            ["Python", "Data loading, model workflows, query execution, and APIs"],
            ["FastAPI + Uvicorn", "Local HTTP application and JSON endpoints"],
            ["Jinja2 + HTML/CSS/JavaScript", "Server-rendered page and interactive UI"],
            ["NumPy + pandas", "Numerical arrays, time-series rows, and CSV processing"],
            ["scikit-learn", "HistGradientBoosting classifiers and evaluation metrics"],
            ["PyTorch", "Neural sensor encoder components in the ExtraSensory model code"],
            ["ZIP streaming", "Reads large SHL/PAMAP2 source data without needless extraction"],
            ["NVIDIA NIM + OpenAI-compatible client", "Optional grounded activity explanation"],
            ["unittest", "Regression and integration test coverage"],
        ],
    )
    doc.add_heading2("Important model and data components")
    for name, description in [
        (
            "dataset_loader.py",
            "Loads ExtraSensory feature rows, labels, official user splits, and available annotations.",
        ),
        (
            "sensor_model.py / context_conditioned_model.py",
            "Implements the activity model and optional separate annotation-conditioned model.",
        ),
        (
            "pamap2_model.py / shl_model.py",
            "Implements raw-sensor windowing, feature extraction, label maps, and dataset-specific classifiers.",
        ),
        (
            "timeline_store.py / context_store.py",
            "Represents activity intervals and observed context/place label timelines.",
        ),
        (
            "query_planner.py / deterministic_engine.py",
            "Maps user questions to supported operations and calculates results from evidence.",
        ),
        (
            "answer_generator.py / nim_llm.py",
            "Builds structured answers and optionally asks NIM to restate verified evidence.",
        ),
        (
            "web_app.py / templates / static",
            "Connects API endpoints, model artifacts, session views, and interactive browser UI.",
        ),
    ]:
        doc.add_bullet(description, name + ":")
    doc.add_paragraph(
        "Important design decision: the LLM is not the query authority or math "
        "engine. The active planner is rule-based, while deterministic code handles "
        "temporal arithmetic. Context/place answers bypass NIM. No API key is "
        "embedded in the report, source code, or launcher."
    )

    doc.add_page_break()
    doc.add_heading1("4. Model and Signal-Processing Workflows")
    doc.add_heading2("ExtraSensory workflow")
    doc.add_paragraph(
        "The ExtraSensory loader reads the existing feature and activity-label "
        "rows and uses official user-based data splits. The activity pipeline "
        "includes a PyTorch sensor encoder and a scikit-learn "
        "HistGradientBoostingClassifier path. A separate context-conditioned "
        "HistGradientBoosting model augments features with observed annotation "
        "values and masks. The context model therefore represents unknown/missing "
        "annotations explicitly instead of assuming they are false."
    )
    doc.add_heading2("Raw IMU workflow: PAMAP2 and SHL")
    doc.add_paragraph(
        "For the raw pipelines, source rows are aligned to timestamp, activity "
        "label, and six channels: Acc X/Y/Z and Gyro X/Y/Z. Candidate windows are "
        "five seconds (500 samples at 100 Hz), advanced by 2.5 seconds (250 "
        "samples). A window is retained only when its label is stable and its "
        "sensor values and sample timestamps pass continuity checks."
    )
    doc.add_paragraph(
        "Each retained window is represented by 54 summary features: eight "
        "channel-wise statistics (mean, standard deviation, minimum, maximum, "
        "root-mean-square, 10th and 90th percentiles, and mean absolute first "
        "difference) over six channels, plus mean, standard deviation, and "
        "root-mean-square magnitude summaries for accelerometer and gyroscope "
        "groups. A separate gradient-boosted tree model is trained for each raw "
        "dataset. No data or labels are pooled across those models."
    )
    doc.add_heading2("Evaluation and deployment artifacts")
    doc.add_paragraph(
        "PAMAP2 and SHL use leave-one-subject/participant-out evaluation. The SHL "
        "web timelines use cached out-of-fold predictions from the held-out "
        "evaluation; its deployment checkpoint is fitted on all preview "
        "participants. The ExtraSensory context result uses a held-out-user fold. "
        "Artifacts under checkpoints/ contain trained models; results/ contains "
        "metrics and relevant prediction caches. These are reproducible project "
        "artifacts, not guarantees that retraining on different data will produce "
        "identical figures."
    )
    doc.add_callout(
        "The raw pipelines model the sensor placement represented by their "
        "training archive. Device units, calibration, placement, sampling, and "
        "participant differences can change performance.",
        "INPUT COMPATIBILITY",
    )

    doc.add_page_break()
    doc.add_heading1("5. Question Answering, Evidence, and Safeguards")
    doc.add_heading2("From natural language to a computed answer")
    doc.add_paragraph(
        "The rule-based QueryPlanner identifies an operation, target activity or "
        "semantic activity group, time bounds, and any context/place filter. The "
        "DeterministicEngine then operates on activity intervals and, when "
        "available, observed context/place intervals. Supported operations include "
        "duration, count, onset, offset, comparison, transition, verification, "
        "identification, open-world mappings, and context/place intersections."
    )
    doc.add_table(
        ["Question type", "Evidence operation"],
        [
            ["How long was walking detected?", "Sum duration of matching intervals"],
            ["How many running periods occurred?", "Count qualifying activity intervals"],
            ["When did bicycling begin/end?", "Earliest start or latest end"],
            ["Walking or sitting: which lasted longer?", "Compare interval-duration totals"],
            ["How often did sitting become walking?", "Count adjacent A-to-B transitions"],
            ["Was walking observed at work?", "Intersect activity with observed labels"],
            ["What city or address was visited?", "Reject: exact location is unavailable"],
        ],
    )
    doc.add_heading2("Grounded answer shape")
    doc.add_paragraph(
        "The answer generator assembles a structured response with the direct "
        "answer, activity/event, evidence, timestamps, sensor modality, channel "
        "attribution, and explanation. The evidence package includes a calculation "
        "trace and support status. For physical activity, channel descriptions "
        "refer to accelerometer and gyroscope evidence where that source supplies "
        "those channels. Context/place-only answers are based on observed labels, "
        "not sensor-channel measurements."
    )
    doc.add_heading2("Truthful handling of missing evidence")
    doc.add_paragraph(
        "Missing context/place annotations are not converted to explicit "
        "negative observations. Coverage is exposed, and incomplete annotation "
        "time must be interpreted as unknown. Activity and context/place queries "
        "are restricted by the selected dataset's supported labels. Requests about "
        "food, heart rate, blood pressure, audio, screen/app use, exact GPS, "
        "addresses, or other unavailable evidence are rejected rather than "
        "answered by language-model guesswork."
    )
    doc.add_paragraph(
        "The QA computation is only as reliable as its activity predictions and "
        "label coverage. Deterministic arithmetic avoids a second source of "
        "numerical drift, but it does not make an upstream model prediction "
        "correct or turn coarse annotations into exact physical location."
    )

    doc.add_page_break()
    doc.add_heading1("6. Evaluation Results and Interpretation")
    doc.add_paragraph(
        "Results below are transcribed from the current JSON artifacts. They come "
        "from different data sources and split protocols; do not compare them as "
        "if they were measurements on one common benchmark."
    )
    doc.add_table(
        ["Evaluation", "Protocol / sample", "Result"],
        [
            ["ExtraSensory activity", "One held-out fold; 55,004 rows", "Accuracy 64.98%; macro-F1 0.4286; weighted-F1 0.6558"],
            ["ExtraSensory context-conditioned", "12 held-out users; 55,004 rows", "Macro-F1 0.5170; accuracy 65.59%"],
            ["ExtraSensory activity-only baseline", "Same context-model fold", "Macro-F1 0.4290; accuracy 64.84%"],
            ["Context model change", "Same fold and rows", "+0.0880 macro-F1 vs activity-only"],
            ["PAMAP2 raw chest IMU", "Leave one of 8 subjects out; 3,288 windows", "Accuracy 83.67%; macro-F1 0.8453"],
            ["SHL raw Hand IMU", "Leave one of 3 participants out; 21,940 windows", "Accuracy 76.61%; macro-F1 0.6976"],
        ],
    )
    doc.add_heading2("Per-participant raw-model variation")
    doc.add_table(
        ["Dataset participant", "Accuracy range", "Macro-F1 range", "Interpretation"],
        [
            ["PAMAP2 subjects 101-108", "43.74%-97.95%", "43.53%-97.94%", "Large participant variability; aggregate hides weak subjects"],
            ["SHL Users 1-3", "73.41%-81.66%", "45.77%-79.74%", "Three users only; User 2 macro-F1 is lowest"],
        ],
    )
    doc.add_heading2("Broader challenge benchmark and edge profile")
    doc.add_paragraph(
        "The stored four-task benchmark reports duration MAE 566.75 seconds, "
        "duration MAPE 61.08%, duration accuracy within 30 seconds 7.02% and "
        "within 60 seconds 15.79%, exact count accuracy 31.58%, comparison "
        "accuracy 33.33%, and mean temporal IoU 0.1812. Modality and channel "
        "grounding precision are recorded as 100% on 56 grounding samples. These "
        "figures expose meaningful QA limitations; they should not be summarized "
        "as uniformly high task performance."
    )
    doc.add_paragraph(
        "The benchmark artifact's edge profiling reports a 0.7344 MB FP32 model "
        "with 0.698 ms CPU latency and 3.04 MB memory footprint; an INT8 model "
        "at 0.2056 MB with 1.12 ms latency, 0.87 MB footprint, and 3.6x model-size "
        "compression; and a tree ensemble at 6.646 ms per window. These are the "
        "stored profiling figures, not independent hardware certification."
    )
    doc.add_callout(
        "Promising aggregate scores coexist with weak per-user results and modest "
        "temporal QA accuracy. More independent participants and broader "
        "subject-held-out testing are needed before operational claims.",
        "HOW TO READ THE RESULTS",
    )

    doc.add_page_break()
    doc.add_heading1("7. Web Application, Operation, and Key Handling")
    doc.add_heading2("User-facing application")
    doc.add_paragraph(
        "The FastAPI application is served locally by Uvicorn and renders its "
        "interface with Jinja2 templates, HTML, CSS, and JavaScript. The browser "
        "can select supported sessions, inspect activity timelines, ask questions, "
        "view available evaluation information, and inspect waveforms. PAMAP2 and "
        "SHL use measured signals from their dataset archives. ExtraSensory's "
        "feature archive does not provide raw signals; its waveform display is "
        "illustrative."
    )
    doc.add_heading2("Start on Windows")
    doc.add_code_block(
        "cd E:\\MC\n"
        ".\\start_sensorqa.ps1\n"
        "# Open http://127.0.0.1:8000 in a browser\n"
        "# Stop the server with Ctrl+C"
    )
    doc.add_paragraph(
        "The launcher uses the project virtual environment and starts the web "
        "server on the local loopback interface. When NVIDIA_API_KEY is not already "
        "set for that PowerShell process, it prompts for hidden input. The supplied "
        "key is held in the server process environment and is not written into the "
        "repository by the launcher."
    )
    doc.add_heading2("Optional NVIDIA NIM")
    doc.add_paragraph(
        "The configured client targets NVIDIA NIM at "
        "https://integrate.api.nvidia.com/v1 using "
        "meta/llama-3.2-11b-vision-instruct. NIM is optional and is used only for "
        "some activity-explanation rewrites. The active question planner is "
        "rule-based, deterministic code calculates results, and context/place "
        "answers bypass NIM. The app's configured status indicates whether a key "
        "is present; it does not prove that the key is valid or active."
    )
    doc.add_heading2("CSV preview and inference boundaries")
    doc.add_paragraph(
        "A CSV upload uses timestamp in seconds plus acc_x, acc_y, acc_z, gyro_x, "
        "gyro_y, and gyro_z columns. The store validates finite, increasing "
        "timestamps and limits size/retention. Uploads are held in memory; "
        "compatible approximately 100 Hz chest-placement inputs can use the "
        "PAMAP2 classifier when the checkpoint is present. Other device "
        "placements, units, and unlabeled recordings should not be represented as "
        "validated classification. SHL's Hand model is not used for arbitrary "
        "uploads because placement cannot be established."
    )
    doc.add_callout(
        "A key previously shared in chat should be revoked and replaced. Enter "
        "the replacement only into the local hidden launcher prompt; do not paste "
        "it into source code, a report, a screenshot, or a browser form.",
        "CREDENTIAL SAFETY",
    )

    doc.add_page_break()
    doc.add_heading1("8. Limitations, Recommendations, and Conclusion")
    doc.add_heading2("Current limitations")
    for text, prefix in [
        (
            "SHL is a small preview with three participants and only three mapped activity classes; its per-user scores vary substantially.",
            "SHL generalization:",
        ),
        (
            "PAMAP2 aggregate metrics conceal strong subject variability, including a held-out subject with accuracy near 44%.",
            "PAMAP2 generalization:",
        ),
        (
            "ExtraSensory provides precomputed features, not raw IMU recordings in the bundled archive; its illustrative waveform is not model evidence.",
            "ExtraSensory data:",
        ),
        (
            "The seven-class ExtraSensory benchmark has severe class imbalance and no positive test support for standing-and-moving in the stored fold.",
            "Class imbalance:",
        ),
        (
            "Recorded quantitative QA scores are modest, and mean temporal IoU is low; arithmetic correctness alone cannot fix weak predicted intervals.",
            "QA reliability:",
        ),
        (
            "Context/place labels are sparse and coarse; exact coordinates, routes, addresses, and cities are unavailable.",
            "Location evidence:",
        ),
        (
            "CSV inference assumes compatible chest placement, units, and sampling; different devices may require calibration and retraining.",
            "Deployment compatibility:",
        ),
        (
            "NIM availability depends on network access, service/model availability, and a valid key; it is optional and not required for deterministic QA.",
            "External service:",
        ),
    ]:
        doc.add_bullet(text, prefix)
    doc.add_heading2("Recommended next steps")
    for text, prefix in [
        (
            "Evaluate on more independent participants and additional recording days; report per-subject metrics and confidence intervals.",
            "Generalization:",
        ),
        (
            "Collect suitable labeled raw recordings for ExtraSensory-compatible raw-signal model development, then evaluate user-held-out before claiming upload classification.",
            "Raw data:",
        ),
        (
            "Improve minority-class coverage and report class-wise precision, recall, and F1 alongside aggregate metrics.",
            "Activity model:",
        ),
        (
            "Measure interval detection and end-to-end question accuracy on held-out sessions, with annotation coverage and prediction uncertainty.",
            "QA evaluation:",
        ),
        (
            "Keep data source, sensor placement, units, label mapping, and evaluation protocol visible in the UI and exported results.",
            "Provenance:",
        ),
        (
            "Use a rotated local credential for NIM and validate its service response independently of the app's key-present indicator.",
            "Credential operations:",
        ),
    ]:
        doc.add_bullet(text, prefix)
    doc.add_heading2("Conclusion")
    doc.add_paragraph(
        "SensorQA is a functional, technically substantial prototype that combines "
        "three separately managed wearable-data paths with activity timelines, "
        "rule-based question planning, deterministic temporal computation, and a "
        "web interface. Important technologies include Python, FastAPI, "
        "scikit-learn, PyTorch, NumPy, pandas, Jinja2, JavaScript, and optional "
        "NVIDIA NIM. The architecture is deliberately evidence-first and preserves "
        "dataset boundaries. Its main remaining work is not basic project "
        "assembly; it is broader validation, improved robustness across "
        "participants and devices, and more accurate end-to-end temporal QA."
    )
    doc.add_heading2("Key project files")
    doc.add_paragraph(
        "Implementation: src/web_app.py, src/dataset_loader.py, "
        "src/sensor_model.py, src/context_conditioned_model.py, "
        "src/pamap2_model.py, src/train_pamap2_model.py, src/shl_model.py, "
        "src/train_shl_model.py, src/timeline_store.py, src/context_store.py, "
        "src/query_planner.py, src/deterministic_engine.py, src/answer_generator.py, "
        "and src/nim_llm.py. Tests: tests/test_all_modules.py. Metrics: "
        "results/benchmark_results.json, results/context_conditioned_model.json, "
        "results/pamap2_metrics.json, and results/shl_metrics.json."
    )

    doc.save(DOCX_PATH)


if __name__ == "__main__":
    generate_full_report()
