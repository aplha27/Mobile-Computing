# SensorQA — Grounded Wearable Activity Question-Answering

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An evidence-grounded natural-language question-answering system over wearable sensor data (Accelerometer & Gyroscope), coarse context/place annotations, and multimodal student mobility logs. Developed for **Challenge 1** in Mobile Computing.

---

## 📌 Executive Overview

SensorQA enforces a foundational, non-negotiable architectural design rule:

> **"Do not let the language model invent the answer or calculate numbers. Let the sensor-processing pipeline discover the physical evidence, let deterministic algorithms perform measurable computations, and let the language model reason over, synthesize, and explain that verified evidence."**

Traditional multimodal systems either pass high-frequency sensor tokens directly to an LLM (causing token explosion and arithmetic hallucination) or use static classifiers that lack conversational expressiveness. **SensorQA bridges this gap through a 3-Stage Pipeline (JARVIS + SensorChat paradigm)**:
1. **Rule-Based Question Decomposition:** Natural language queries are compiled into structured, machine-executable JSON query plans.
2. **Deterministic Execution Engine:** All durations $\sum(\Delta t)$, visit frequencies, temporal overlaps, and posture transitions are computed with 100% mathematical precision in Python.
3. **Grounded Answer Assembly:** Results are assembled into a mandatory 7-field structured schema citing exact timestamps, sensor channels, and verified physical evidence.

---

## 👥 Equal Team Workload Distribution (4 Members — 25% Each)

The project was executed collaboratively by a four-member engineering team with equal work distribution:

| Team Member / Role | Functional Domain | Primary Responsibilities & Deliverables | Code Attribution | Effort |
|:---|:---|:---|:---|:---:|
| **Member 1**<br>*(Perception Lead)* | **Sensor Modeling & Edge Compression** | • Trained PyTorch `SensorEncoderNN` with temperature scaling $T$<br>• Tabular `HistGradientBoosting` ensemble for 7 ExtraSensory classes<br>• PAMAP2 chest-IMU 100 Hz windowing and leave-one-subject-out cross-validation<br>• PyTorch dynamic INT8 quantization achieving **3.6× compression (41 KB)** | `src/sensor_model.py`<br>`src/pamap2_model.py`<br>`src/train_pamap2_model.py` | **25%** |
| **Member 2**<br>*(Context Lead)* | **Data Ingestion, Context & Places** | • Built `ExtraSensoryLoader` across 60 subjects with official 5-fold CV<br>• Integrated `StudentLifeLoader` (49 users) with GPS, Wi-Fi location & EMA surveys<br>• Streaming parser for SHL raw Hand-placement motion files<br>• Built `ContextTimeline` & `LocationTimeline` tracking observation coverage | `src/dataset_loader.py`<br>`src/studentlife_loader.py`<br>`src/shl_model.py`<br>`src/context_store.py` | **25%** |
| **Member 3**<br>*(Reasoning Lead)* | **Query Planning, Math & Safety** | • Rule-based `QueryPlanner` mapping questions to structured JSON<br>• Pure-Python `DeterministicEngine` for duration sums, counts & comparisons<br>• Temporal interval intersection for Activity $\cap$ Context/Location co-occurrence<br>• DrHouse truthful rejection (**100% rejection rate** for food, heart rate, GPS) | `src/query_planner.py`<br>`src/deterministic_engine.py` | **25%** |
| **Member 4**<br>*(Full-Stack Lead)* | **Web Services, UI & Evaluation** | • FastAPI backend, REST endpoints, and multi-session routing<br>• Interactive front-end (real-time chat, Chart.js waveforms, timeline viewer)<br>• Automated 4-Task benchmark evaluation suite and edge hardware profiling<br>• Answer generator with 7-field schema & optional NVIDIA NIM client | `src/web_app.py`<br>`src/answer_generator.py`<br>`src/evaluator.py`<br>`src/templates/`<br>`src/static/` | **25%** |

---

## 🏗️ System Architecture

```text
       DATASET-SPECIFIC INPUT AND CLASSIFIERS
 [ExtraSensory features]  → [ExtraSensory classifier]
 [PAMAP2 chest IMU]       → [PAMAP2 raw-signal model]
 [SHL Hand IMU]           → [SHL shared-label model]
 [StudentLife Sensing]    → [ActivityRec + Wi-Fi Location + GPS]
                    │
                    ↓
          [ STEP 1: User / Session Data Loading ]
                    ↓
          [ STEP 2: Dataset-Specific Inference ]
                    ↓
          [ STEP 3: Multi-Layer Evidence Timelines ]
          • ActivityTimeline (Predicted intervals & confidence)
          • ContextTimeline  (Observed travel states & EMA)
          • LocationTimeline (Observed coarse place labels)
                    │
                    ↓
User Question ──► [ STEP 4: Rule-Based Query Planning ]
                    ↓
              Structured QueryPlan JSON
                    ↓
     ====================================================
     STAGE 2: DETERMINISTIC COMPUTATION ENGINE
     • Duration:     Σ(end - start)
     • Count:        N distinct intervals
     • Comparison:   Duration(A) - Duration(B)
     • Transitions:  Detected A → B switches
     • Co-occurrence: Activity ∩ Context/Place intervals
     • Place Dwell:  Summed dwell time across visits
     ====================================================
                    ↓
              Evidence Package & Math Trace
                    ↓
     [ STEP 6: Stage 3 — Grounded Answer Assembly ]
                    ↓
     [ FINAL OUTPUT: Mandatory 7-Field Structured Schema ]
```

---

## 🎯 Target Datasets & Provenance

SensorQA enforces **strict data provenance isolation** to avoid false domain generalization:

| Dataset | Modality / Sensor Inputs | Sampling | Target Classes Supported | Provenance & Use Case |
|:---|:---|:---:|:---|:---|
| **ExtraSensory** | Accelerometer + Gyroscope (precomputed features) | 25 Hz | 7 classes (Lying down, Sitting, Standing in place, Standing & moving, Walking, Running, Bicycling) | Primary QA benchmark; sparse context/place annotations |
| **PAMAP2** | Raw chest-mounted IMU (Acc + Gyro) | 100 Hz | 6 classes (Lying, Sitting, Standing, Walking, Running, Cycling) | Subject-held-out raw signal verification |
| **SHL Preview** | Raw Hand-placement IMU (Acc + Gyro) | 100 Hz | 3 classes (Walking, Running, Bicycling) | Hand placement locomotion analysis |
| **StudentLife** | Android ActivityRecognition + Wi-Fi + GPS + EMA | Event/Periodic | 3 classes (Sitting, Walking, Running) + Places (gym, workplace, home, restaurant) | Real-world continuous location- and context-aware QA |

---

## 🏆 Four Progressive Challenge Tasks & Results

SensorQA is benchmarked across the four Challenge 1 evaluation tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   4-TASK EMPIRICAL BENCHMARK SCORES                    │
├──────────────────────────┬─────────────────────────────────────────────┤
│ Task 1: Identification   │ 64.98% Accuracy · 0.6558 Weighted F1        │
│ Task 2: Temporal Math    │ Zero Math Hallucination · 566.75s Duration  │
│ Task 3: Grounding        │ 100% Modality Precision · 100% Channel Prec │
│ Task 4: Open-World & Safe│ 100% Semantic Acc · 100% Truthful Rejection │
└──────────────────────────┴─────────────────────────────────────────────┘
```

- **Task 1 — Activity Identification:** Tested across 55,004 completely unseen test windows (Fold 0). Best performing class: *Lying down* (F1 = 0.77); *Sitting* (F1 = 0.69).
- **Task 2 — Temporal & Quantitative Reasoning:** Exact count accuracy of 31.58% and comparative accuracy of 33.33% across 126 queries with **zero calculation drift**.
- **Task 3 — Evidence Grounding & Attribution:** **100% modality precision** (Acc + Gyro) and **100% 6-channel attribution** with temporal interval IoU.
- **Task 4 — Open-World Reasoning & Safety:** **100% semantic reasoning accuracy** (*wheeled movement*, *sedentary postures*) and **100% truthful rejection** blocking out-of-scope queries (food, heart rate, exact GPS).

---

## ⚡ Edge & Ubiquitous Computing Profiling

SensorQA benchmarks two distinct operational targets for resource-constrained wearable edge devices:

| Metric | Server Full Precision (FP32) | Quantized Edge (Dynamic INT8) | Advantage |
|:---|:---:|:---:|:---:|
| **Parameters** | 192,520 | 192,520 | Identical topology |
| **Model Size** | 0.734 MB | **0.206 MB (41 KB Core)** | **3.6× smaller** |
| **Latency** | 0.698 ms / window | 1.120 ms / window | Real-time capable (< 2 ms) |
| **Throughput** | 1,433.4 windows/sec | 892.8 windows/sec | High-frequency on-device streaming |
| **RAM Footprint** | 3.04 MB | **0.87 MB** | Fits comfortably under 1 MB RAM |

---

## 📋 Mandatory 7-Field Structured Output Schema

Every answer generated strictly adheres to the Challenge 1 output format:

```text
Answer: At least 37510 seconds

Activity/Event: Gym

Evidence: SUM(Gym labeled intervals) = 37510.0 s; label observation fraction 0.7%
Timestamp(s): 28107–28149 s, 35761–35862 s, 40714–40754 s, 49406–49514 s...

Sensor Modality:
Wi-Fi location / place labels (coarse)

Sensor Channel(s):
N/A (no raw sensor channels)

Explanation:
SUM(Gym labeled intervals) = 37510.0 s. Unlabeled time is unknown and is not counted as a negative observation. These are coarse place annotations; raw GPS coordinates and exact addresses are not available.
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites & Environment Setup
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/aplha27/Mobile-Computing.git
cd Mobile-Computing

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install fastapi uvicorn jinja2 numpy pandas scikit-learn torch python-pptx
```

### 2. Run Test Suite
Run the comprehensive unit test suite:
```bash
python -m unittest discover -s tests -p test_all_modules.py
```

### 3. Launch Interactive Web Application
```bash
python -m uvicorn src.web_app:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser to access:
- 💬 **SensorChat Assistant:** Interactive conversational QA across ExtraSensory, PAMAP2, SHL, and StudentLife sessions.
- 📈 **Sensor Waveform Visualizer:** 6-channel real-time inertial signal viewer.
- 📊 **Benchmark Hub:** Complete 4-task evaluation metrics and edge profiling.
- 🧭 **Location & Context Explorer:** Live queries for place dwell times and visit frequencies.

---

## 📁 Repository Structure

```text
Mobile-Computing/
├── src/
│   ├── dataset_loader.py          # ExtraSensory feature/label loader & fold splitting
│   ├── studentlife_loader.py      # StudentLife continuous sensing & place mapper
│   ├── pamap2_model.py            # PAMAP2 raw chest-IMU model & windowing
│   ├── shl_model.py               # SHL Hand-placement IMU loader & shared-label model
│   ├── sensor_model.py            # PyTorch SensorEncoderNN & HistGradientBoosting
│   ├── timeline_store.py          # Central ActivityTimeline store & interval indexer
│   ├── context_store.py           # ContextTimeline & LocationTimeline with coverage
│   ├── query_planner.py           # Rule-based question decomposition & intent parser
│   ├── deterministic_engine.py    # Deterministic math engine (zero LLM arithmetic)
│   ├── answer_generator.py        # 7-field structured answer assembly & NIM client
│   ├── nim_llm.py                 # Optional NVIDIA NIM narrative explanation client
│   ├── evaluator.py               # Complete 4-task multi-metric evaluation suite
│   ├── web_app.py                 # FastAPI backend & session router
│   ├── templates/index.html       # Web UI interface
│   └── static/app.js              # Front-end charting, chat, and dashboard scripts
├── tests/
│   └── test_all_modules.py        # Complete unit & regression test suite
├── checkpoints/                   # Trained model weights & quantized checkpoints
├── results/                       # Empirical benchmark metrics (JSON)
├── generate_presentation.py       # Automated PowerPoint generation script
├── generate_word_report.py        # Automated Word report generation script
├── SensorQA_Presentation.pptx     # 29-Slide PowerPoint deck
├── SensorQA_Challenge_1_Comprehensive_Project_Report.docx # Comprehensive project report
├── start_sensorqa.ps1             # Windows startup script with secure key handling
├── .gitignore                     # Git tracking rules
└── README.md                      # Project documentation
```

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
