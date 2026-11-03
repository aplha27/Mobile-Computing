# Project Overview — Grounded Activity Question-Answering (SensorQA)

## 📌 Project Summary

**SensorQA** is an evidence-grounded prototype for **natural-language question-answering over wearable activity, context, and coarse place annotations**.

SensorQA has three separate activity paths: ExtraSensory precomputed per-user feature/label rows, PAMAP2 raw chest-IMU recordings, and the SHL preview's raw Hand-placement IMU. The PAMAP2 and SHL models use five-second windows with 2.5-second hops at approximately 100 Hz. Their models and metrics remain separate because the sensor placements and activity labels differ. PAMAP2 and SHL sessions use measured archive signals; ExtraSensory waveforms remain illustrative.

SensorQA keeps activity classification and temporal arithmetic in Python. Its rule-based planner maps questions to structured queries; NVIDIA NIM is optional for activity-answer explanation only.

```text
┌────────────────────────────────────────────────────────┐
│ ExtraSensory features/labels + User Question           │
└──────────────────────────┬─────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────┐
│ Evidence-Grounded Structured Answer                   │
└────────────────────────────────────────────────────────┘
```

---

## 🎯 Target Dataset & Activities

The system is trained and evaluated on the benchmark **ExtraSensory dataset**:

### Activity and context data
- Activity classification uses the selected ExtraSensory feature columns; the model input width is determined by the loaded data/checkpoint.
- ExtraSensory rows also include multi-label context and place annotations, plus coarse location-derived summary features.
- ExtraSensory still has no raw GPS coordinate sequence. Its waveform endpoint is illustrative; PAMAP2 and SHL sessions display measured archive IMU signals.

### 7 Target Activities & Postures
1. 🛌 **Lying down**
2. 🪑 **Sitting**
3. 🧍 **Standing in place**
4. 🚶‍♂️ **Standing and moving**
5. 🚶 **Walking**
6. 🏃 **Running**
7. 🚴 **Bicycling**

---

## 🏆 The Four Progressive Challenge Tasks

The system is designed and benchmarked across four progressively harder tasks:

### 1. Task 1 — Activity Identification
- *Goal*: Accurately identify what activity the user was performing during specific recording intervals.
- *Examples*: *"What activity was the user performing between 300s and 600s?"*, *"Is the user running?"*

### 2. Task 2 — Temporal & Quantitative Reasoning
- *Goal*: Compute exact durations, frequencies, comparisons, onset times, and posture transitions without numerical hallucinations.
- *Examples*:
  - **Duration**: *"How long did the user walk?"* $\to$ *700 seconds*
  - **Count / Frequency**: *"How many times did the user run?"* $\to$ *4 times*
  - **Comparison**: *"Did the user spend more time walking or sitting?"* $\to$ *Sitting (by 300 seconds)*
  - **Transition**: *"How many times did the user transition from sitting to walking?"* $\to$ *2 times*
  - **Onset**: *"When did running begin?"* $\to$ *At 1200 seconds*

### 3. Task 3 — Evidence Grounding & Provenance
- *Goal*: Every answer must be strictly attributed to physical sensor evidence with temporal intersection-over-union (IoU), exact timestamps, and 6-axis sensor channel attribution.

### 4. Task 4 — Open-World Semantic Reasoning & Truthful Safety Rejection
- *Goal*: Handle abstract, non-class semantic questions while safely rejecting queries requiring unavailable sensors.
- *Examples*:
  - **Open-World**: *"Was the user using a wheeled or pedal-based mode of movement?"* $\to$ *Yes (Bicycling detected)*
  - **Truthful Rejection**: *"What food did the user eat for dinner?"* $\to$ *Truthfully rejected: food/nutrition is not present in the available evidence.*

### Location- and context-aware queries
The current extension supports annotated context/place questions such as *"How long was I sitting in a car?"*, *"How long was I at the gym?"*, *"Did I walk while at work?"*, and *"How many times did I go from home to work?"*. It intersects activity intervals with observed labels and reports annotation coverage. Missing annotations are treated as unknown, not as negative evidence.

The archive audit found 377,346 rows across 60 users. Positive `LOC_home` labels occur in 57 users and positive `AT_THE_GYM` labels in 8 users; this illustrates the label sparsity. These are row counts, not GPS fixes or independent ground-truth visits. Place answers use only coarse place labels. Exact GPS coordinates, addresses, and cities remain unsupported, and location values are not sent to NVIDIA NIM.

The ExtraSensory context-conditioned model and PAMAP2 raw-signal model are independent checkpoints and are not evaluated against each other because their datasets and inputs differ.

### PAMAP2 raw-signal activity path

The bundled PAMAP2 archive contains nine protocol participants. Its protocol
subject 109 file contains only rope-jumping labels, so it cannot evaluate the
six common SensorQA activities. PAMAP2 training maps only direct label matches:
lying, sitting, standing, walking, running, and cycling; it does not invent a
match for "standing and moving" or the other PAMAP2 activities.

Train and evaluate the separate raw-IMU model:

```powershell
.\.venv\Scripts\python.exe -m src.train_pamap2_model
```

The trainer uses five-second chest-IMU windows at 100 Hz, a 2.5-second hop,
and leave-one-subject-out evaluation across subjects 101-108. It then fits a
deployment checkpoint. In the bundled run, pooled out-of-fold accuracy was
83.67% and six-class macro-F1 was 0.8453 across 3,288 windows. Subject-level
results vary substantially (accuracy 43.74%-97.95%), so these are not a
universal performance guarantee. The evaluation metrics are in
`results/pamap2_metrics.json`.

The web app loads the PAMAP2 checkpoint when present, adds PAMAP2 subject
timelines, and uses real archive signals in the waveform chart. Six-axis CSV
uploads can also be classified when they are approximately 100 Hz and match
the chest sensor placement and units (accelerometer m/s², gyroscope rad/s).
Other placements, device calibrations, or sensor units may yield unreliable
predictions.

### SHL Hand-placement raw-signal activity path

The three local SHL preview archives contain nine recording days from three
participants. Each day includes per-sample labels and separate sensor files for
multiple phone placements. SensorQA reads only `Hand_Motion.txt` and `Label.txt`
from each ZIP; it does not extract the multi-gigabyte archives. The Hand stream
is approximately 100 Hz. Windows are retained only when all six acceleration
and gyroscope channels are finite, timestamps are contiguous, and the mapped
activity label is stable for the full five seconds.

The SHL activity taxonomy is not interchangeable with PAMAP2. The SHL model
maps only direct shared classes—walking, running, and bicycling—to SensorQA
activities. Still is not guessed to mean sitting, lying, or standing, and
motorized-transport labels are not treated as walking or bicycling. Therefore
only the three mapped activities can be queried for SHL sessions. SHL does not
provide ExtraSensory context/place labels, and its Hand model is not used to
classify uploaded CSVs.

Train the independent model and participant-held-out evaluation:

```powershell
.\.venv\Scripts\python.exe -m src.train_shl_model
```

The evaluator holds out all days for one participant at a time (three folds for
this preview), then saves a separate deployment checkpoint, held-out metrics,
and out-of-fold predictions for the web session timelines. The preview has only
three participants; its results are exploratory and do not certify performance
on new users. In this preview, pooled out-of-fold accuracy was 76.61% and
three-class macro-F1 was 0.6976 over 21,940 windows. Per-user accuracy ranged
from 73.41% to 81.66%, while macro-F1 ranged from 0.4577 to 0.7974. This
variation and the small participant count are important limitations. See
`results/shl_metrics.json`. The dataset license requests credit to the
originators and a current recommended citation; follow its `LICENSE` before
publishing results.

---

## 🔬 Four Foundational Research Papers Synthesized

| Paper | Core Innovation Borrowed | Role in SensorQA |
| :--- | :--- | :--- |
| **Sensor2Text** | Multimodal sensor representation learning | Provides motivation for learned sensor representations; the current classifier consumes ExtraSensory feature rows rather than newly windowed raw 25 Hz signals. |
| **SensorChat** | Three-stage QA architecture | Decomposes questions into structured queries, retrieves compact temporal episodes, and passes verified evidence rather than raw time-series tokens. |
| **DrHouse** | Calibrated confidence & truthful rejection | Uses confidence-aware activity evidence and rejects unavailable information such as exact GPS coordinates, heart rate, and food. |
| **JARVIS** | Structured query planning & deterministic execution | Uses a rule-based parser to produce structured query plans and delegates arithmetic and interval operations to deterministic Python code. |

---

## 📋 Mandatory 7-Field Structured Output Schema

Every answer generated by the system strictly satisfies the Challenge 1 schema specification:

```text
Answer: <Direct factual numerical or categorical answer>

Activity/Event: <Target activity name>

Evidence: <Summary of sensor detection across recording intervals>
Timestamp(s): <e.g. 905–1420 s, 2110–2295 s>

Sensor Modality:
Accelerometer + Gyroscope

Sensor Channel(s):
Acc X + Acc Y + Acc Z + Gyro X + Gyro Y + Gyro Z

Explanation:
<Explanation grounded in the calculation trace, activity evidence, and/or annotated context/place labels>
```

---

## 🚀 NVIDIA NIM LLM Integration

NVIDIA NIM `meta/llama-3.2-11b-vision-instruct` is optional and used only to rewrite some activity explanations. Query planning is rule-based. Set `NVIDIA_API_KEY` in the environment to enable NIM; there is no API key in source code. Context/place answers bypass NIM, and no raw coordinates are available or transmitted.

On Windows, `.\start_sensorqa.ps1` prompts for the key with hidden input when the
environment variable is not already set. The launcher keeps the key in the
server process environment only; it does not write it to the project. The
sidebar reports whether a key is configured, not whether NVIDIA has validated
or activated that key.

The PAMAP2 classifier provides a raw-signal training and inference path for
six mapped activities. It is separate from the seven-class ExtraSensory model,
and does not support PAMAP2-only activities or context/place questions.

The waveform card accepts a UTF-8 CSV with `timestamp` (seconds) and `acc_x`,
`acc_y`, `acc_z`, `gyro_x`, `gyro_y`, and `gyro_z` columns. Imported samples
are held in memory (up to three recordings), not written to disk or sent to
NIM. When the PAMAP2 checkpoint is loaded, compatible 100 Hz recordings are
classified in 5-second windows; other recordings remain preview-only.

An optional context/place-conditioned classifier can be trained on the
ExtraSensory feature rows and annotations with:

```powershell
.\.venv\Scripts\python.exe -m src.train_context_model
```

It evaluates on the official held-out users and saves a separate checkpoint;
the web application uses it when that checkpoint exists. Missing annotation
values are represented separately from explicit negatives. This model still
uses precomputed ExtraSensory features, not raw wearable recordings.

On the bundled official fold, the context-conditioned model reached a
test macro-F1 of 0.5170 versus 0.4290 for the scaler-aware activity-only
baseline (+0.0880) across 12 unseen users and 55,004 rows. On those held-out
rows, about 52.6% of context labels and 51.8% of place labels were observed.
This is a result on one fold, not a multi-fold certification or evidence that
context annotations will always improve predictions. Details are saved in
`results/context_conditioned_model.json`.

---

## 📁 Repository Structure

```text
e:/MC/
├── src/
│   ├── dataset_loader.py       # ExtraSensory activity, context/place labels and location summaries
│   ├── sensor_model.py         # Model A (PyTorch SensorEncoderNN + Boosted Tree ensemble + INT8 Quantizer)
│   ├── pamap2_model.py        # PAMAP2 raw six-axis windows and six-activity model
│   ├── train_pamap2_model.py  # Subject-held-out PAMAP2 training and evaluation
│   ├── shl_model.py           # SHL Hand IMU parser, shared-label windows, and model
│   ├── train_shl_model.py     # Participant-held-out SHL training and evaluation
│   ├── timeline_store.py       # Central Activity Timeline Store (interval aggregation & confidence tracking)
│   ├── context_store.py        # Context and coarse place timelines with annotation coverage
│   ├── query_planner.py        # Rule-based question decomposition and semantic mapping
│   ├── deterministic_engine.py # Stage 2 deterministic math engine (sums, counts, comparisons, transitions)
│   ├── answer_generator.py     # Stage 3 structured answer assembly & NIM narrative verbalization
│   ├── nim_llm.py              # Optional NVIDIA NIM explanation client
│   ├── evaluator.py            # Complete 4-task multi-metric evaluation suite
│   ├── web_app.py              # FastAPI web server backend with REST endpoints
│   ├── templates/index.html    # Modern interactive web UI with inline design system
│   └── static/app.js           # Front-end interactive charts, chat handlers, and benchmark dashboards
├── tests/
│   └── test_all_modules.py     # Unit & integration tests for activity/context/location QA
├── checkpoints/
│   └── sensor_model.pkl        # Trained Model A checkpoint
│   └── pamap2_activity_model.pkl # PAMAP2 chest-IMU model checkpoint
│   └── shl_activity_model.pkl   # SHL Hand-IMU model checkpoint
├── results/
│   └── benchmark_results.json  # Comprehensive 4-task benchmark metrics & edge profiling data
│   └── pamap2_metrics.json     # PAMAP2 subject-held-out raw-signal metrics
│   └── shl_metrics.json        # SHL participant-held-out preview metrics
├── Challenge_1_Core_Research_Conclusion.md # Original core research document
├── System_Architecture_and_Design_Justification.md # Detailed architecture breakdown & justification
└── Project_Overview_and_Explanation.md # This project documentation
```

---

## ⚡ Edge & Ubiquitous Computing Profiling

For deployment on wearable devices, smartwatches, and microcontrollers (e.g. ARM Cortex-M, ESP32, Apple Watch, WearOS), the system profiles two operational configurations:

- **Model A (Full Precision FP32)**: Server-side batch model (`0.147 MB`, `0.19 ms` latency, `5,150 windows/sec`).
- **Model B (Quantized INT8 / Edge)**: Ultra-compact edge model (`0.041 MB / 41 KB`, `0.53 ms` latency, `1,856 windows/sec`, `< 0.6 MB` RAM footprint, **3.6x compression ratio**).

---

## 💻 How to Run the Project

### 1. Run Unit Tests
```bash
python -m unittest discover -s tests -p test_all_modules.py
```

### 2. Launch the Interactive Web Application
```bash
python -m uvicorn src.web_app:app --host 127.0.0.1 --port 8000
```
The web app loads the PAMAP2 model from `checkpoints/pamap2_activity_model.pkl`
when available. Rebuild it with `python -m src.train_pamap2_model`.
Open **`http://127.0.0.1:8000`** to access:
- 💬 **SensorChat Assistant**: Interactive conversational QA with ExtraSensory timelines, PAMAP2 raw-signal timelines, and waveform visualization.
- 🧭 **Context/place QA**: Queries supported ExtraSensory labels through `/api/session/{id}/context` and `/api/session/{id}/location`.
- 🔬 **Research Findings & Theory Hub**: Literature synthesis and architectural specifications.
- 📊 **Benchmark Dashboard**: Existing activity benchmark results; context/place QA has not yet been independently benchmarked.
- ⚡ **Edge Model Profiling**: FP32 vs INT8 latency and memory comparison.

### 3. Run Standalone Evaluation Pipeline
```bash
python -m src.run_pipeline
```
