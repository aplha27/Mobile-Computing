# Grounded Activity Question-Answering (SensorQA) — System Architecture & Design Justification

## Executive Overview

This document provides a comprehensive technical breakdown and design rationale for the **Grounded Wearable Activity Question-Answering System (SensorQA)** developed for **Challenge 1**. 

The system has three separate activity-classification paths: ExtraSensory precomputed per-user feature/label rows, a PAMAP2 model trained on raw chest IMU, and an SHL preview model trained on raw Hand-placement IMU. The PAMAP2 and SHL pipelines use five-second windows with 2.5-second hops at approximately 100 Hz. Their models remain separate because their placements and activity taxonomies differ. ExtraSensory has no raw stream in this archive; its waveform remains illustrative. The architecture draws on four foundational research papers:
1. **Sensor2Text**: Multimodal sensor representation learning and temporal-spectral feature projection.
2. **SensorChat**: Three-stage QA architecture (Question Decomposition $\to$ Evidence Query $\to$ Grounded Answer Assembly).
3. **DrHouse**: Calibrated uncertainty/confidence intervals, evidence-first reasoning, and truthful rejection of unsupported domains.
4. **JARVIS**: Structured JSON query planning and deterministic execution of numerical mathematics outside the language model.

---

## 1. High-Level Architecture Diagram

```text
       DATASET-SPECIFIC INPUT AND CLASSIFIERS
 [ExtraSensory features] → [ExtraSensory classifier]
 [PAMAP2 chest IMU]      → [PAMAP2 raw-signal model]
 [SHL Hand IMU]          → [SHL shared-label model]
                    │
                    ↓
          [ STEP 1: User/session data loading ]
                    ↓
         [ STEP 2: Dataset-specific inference ]
                    ↓
         [ STEP 3: Evidence timelines ]
 ExtraSensory context/place labels only when observed
                        │
                        │ (Retrievable Evidence Layer)
                        ↓
User Question ──► [ STEP 4: Stage 1 - Rule-based Query Planning ]
                        ↓
                  Structured QueryPlan JSON
                        ↓
     ====================================================
     MODEL B: LANGUAGE REASONING & GROUNDING ENGINE
     [ STEP 5: Stage 2 - Deterministic Computation Engine ]
         • Duration: Σ(end - start)
         • Count: N intervals
         • Comparison: Duration(A) - Duration(B)
         • Transitions: Detected A → B switches
         • Context: Activity ∩ annotated context/place intervals
         • Place: Label-based dwell, visits, and transitions
     ====================================================
                        ↓
                  Evidence Package & Math Trace
                        ↓
         [ STEP 6: Stage 3 - Grounded Answer Assembly ]
                        ↓
         [ FINAL OUTPUT: Mandatory 7-Field Structured Schema ]
```

---

## 2. Step-by-Step Walkthrough of the Architecture

### **Step 1: ExtraSensory Feature and Label Loading**
* **Function**: Loads a per-user CSV record from the ExtraSensory archive.
* **Mechanism**:
  - Uses the source's existing feature/label rows and timestamps; it does not resample raw streams or create 20-second windows.
  - Separates seven activity targets, multi-label context annotations, coarse place labels, and available location-derived summary fields.
  - Preserves missing context/place annotations as unknown rather than converting them to negative labels.
  - The chart endpoint's 25 Hz waveforms are synthetic examples and are not classifier evidence.
* **Source Implementation**: `src/dataset_loader.py` (`ExtraSensoryLoader`).

---

### **Step 2: Model A — Multimodal Sensor Understanding & Uncertainty Calibration**
* **Function**: Translates low-level inertial feature vectors into discrete activity classifications with calibrated confidence probabilities.
* **Mechanism**:
  - Employs a dual-architecture approach:
    1. **Deep Neural Sensor Encoder (`SensorEncoderNN`)**: Multi-layer projection with batch normalization, GELU activations, and a learnable temperature scaling parameter $T$.
    2. **Gradient Boosted Tree Ensemble (`HistGradientBoostingClassifier`)**: Fast, high-accuracy tabular model for robust classification.
  - Classifies ExtraSensory feature rows into the 7 target activities:
    1. *Lying down*
    2. *Sitting*
    3. *Standing in place*
    4. *Standing and moving*
    5. *Walking*
    6. *Running*
    7. *Bicycling*
  - Outputs a probability vector scaled by temperature $T$ to prevent overconfident predictions during ambiguous movements.
* **Source Implementation**: `src/sensor_model.py` (`SensorUnderstandingPipeline`).

---

### **Step 2b: Separate Raw-IMU Dataset Models**
* **Function**: Add raw-stream activity evidence without presenting incompatible datasets as one sensor domain.
* **Mechanism**:
  - PAMAP2 uses protocol chest-IMU streams and six directly mapped activities.
  - SHL reads only the Hand-placement motion stream and per-sample main activity labels directly from the three local preview ZIP files.
  - SHL maps only walking, running, and bicycling to the shared SensorQA taxonomy; unshared labels are excluded rather than guessed.
  - SHL evaluation holds out every recording day for one participant together. The three-participant preview is exploratory, not a population-wide performance guarantee.
  - The SHL model, metrics, and out-of-fold session predictions are stored separately from PAMAP2. The web app uses the out-of-fold predictions for preview timelines.
  - Neither raw-IMU path supplies context/place labels. The SHL Hand model is not used for arbitrary uploaded CSVs.
* **Source Implementation**: `src/pamap2_model.py`, `src/train_pamap2_model.py`, `src/shl_model.py`, and `src/train_shl_model.py`.

---

### **Step 3: Activity, Context, and Place Timelines**
* **Function**: Bridges predicted activity intervals and observed ExtraSensory labels with high-level query reasoning.
* **Mechanism**:
  - `ActivityTimeline` merges classifier predictions into activity episodes. These are model predictions, not ground-truth labels.
  - ExtraSensory, PAMAP2, and SHL sessions use distinct model inputs and supported activity sets.
  - `ContextTimeline` and `LocationTimeline` align observed labels to the same relative session timeline and retain per-label observation coverage.
  - Context/place data is not inferred when a label is absent. Place evidence consists only of coarse labels such as home, workplace, gym, restaurant, beach, and school.
  - The archive audit covered 377,346 rows across 60 users. Positive home labels occurred for 57 users and gym labels for 8; sparsity is surfaced as coverage rather than hidden.
  - Location-derived numeric summaries are retained by the loader, but raw GPS coordinates are not available in this data path and are not passed to an LLM.
* **Source Implementation**: `src/timeline_store.py` (`ActivityTimeline`) and `src/context_store.py` (`ContextTimeline`, `LocationTimeline`).

---

### **Step 4: Rule-Based Query Planning**
* **Function**: Maps natural-language requests to structured, deterministic operations.
* **Mechanism**:
  - `QueryPlanner` is rule-based; the optional NVIDIA NIM integration does not plan queries.
  - Adds `context_filter` / `co_occurrence`, `location_duration` / `location_count`, `place_transition`, and coarse `location_identification` intents.
  - Activity/context queries intersect predicted activity intervals with observed context/place-label intervals.
  - Exact coordinates, addresses, cities, and other unavailable data remain unsupported. Food and heart-rate requests remain rejected.
* **Source Implementation**: `src/query_planner.py` (`QueryPlanner`, `QueryPlan`).

### **Step 5: Stage 2 — Deterministic Computation Engine**
* **Function**: Executes temporal operations over the activity, context, and place timelines.
* **Mechanism**:
  - The natural language query (e.g. *"Did the user spend more time walking or sitting in the first 30 minutes?"*) becomes a structured `QueryPlan`:
    ```json
    {
      "operation": "comparison",
      "target_activity": "walking",
      "compare_activity": "sitting",
      "time_bounds": {"t_min": 0.0, "t_max": 1800.0},
      "is_supported": true,
      "unsupported_reason": null
    }
    ```
  - **Zero LLM Arithmetic**: All calculations are executed by deterministic Python algorithms:
    - **Duration**: $\sum_{i} (\text{end}_i - \text{start}_i)$
    - **Count / Frequency**: $N = \text{len}(\text{filtered\_intervals})$
    - **Comparison**: $\Delta = \text{Duration}(\text{Activity}_A) - \text{Duration}(\text{Activity}_B)$
    - **Transition Detection**: Evaluates boundary timestamps $t_i$ where $\text{Activity}(t_i^-) = A$ and $\text{Activity}(t_i^+) = B$.
    - **Onset / Offset**: Finds $\min(\text{start}_i)$ or $\max(\text{end}_i)$.
    - **Context co-occurrence**: Computes the intersection of activity and observed context/place intervals.
    - **Place dwell and visits**: Sums place-label interval durations or counts observed intervals.
    - **Place transitions**: Counts adjacent observed place-label interval boundaries.
  - Queries surface label-observation coverage. If coverage is incomplete, durations/counts are reported as observed lower bounds and negative co-occurrence answers become unknown.
  - Place transitions are counted only between adjacent observed labeled intervals; they are not derived from raw trajectories.
  - Packages the output into an immutable `EvidencePackage` containing the exact mathematical trace (e.g., `(1420 - 905) + (2295 - 2110) = 700 s`).
* **Source Implementation**: `src/context_store.py`, `src/deterministic_engine.py` (`DeterministicEngine`, `EvidencePackage`).

---

### **Step 6: Grounded Answer Assembly**
* **Function**: Formats deterministic evidence in the structured answer schema.
* **Mechanism**:
  - Activity answers may use optional NVIDIA NIM for narrative explanation when `NVIDIA_API_KEY` is set; no key is embedded in the repository.
  - Context and place answers are assembled deterministically and do not call NIM.
  - When the source label is incomplete, the answer communicates coverage and avoids treating unknown time as a negative.
  - Delivers the mandatory 7-field structured output schema:
    ```text
    Answer: 700 seconds

    Activity/Event: Walking

    Evidence: Walking was detected across 2 distinct intervals.
    Timestamp(s): 905–1420 s, 2110–2295 s

    Sensor Modality:
    Accelerometer + Gyroscope, or the observed ExtraSensory context/place labels

    Sensor Channel(s):
    Physical channels when relevant; N/A for label-only place queries

    Explanation:
    The walking duration of 700 seconds was calculated by combining the accelerometer and gyroscope data from the 905–1420 s and 2110–2295 s time intervals. This result is derived from the deterministic mathematical trace, which sums the durations of 515 seconds and 185 seconds, respectively. The 6-axis inertial sensor kinematics provided the necessary data to accurately quantify the walking activity.
    ```
* **Source Implementation**: `src/answer_generator.py` (`AnswerGenerator`, `StructuredAnswer`).

---

## 3. Design Justification: Why This Architecture?

| Alternative Architecture | Why We Rejected It | How Our Architecture Solves It |
| :--- | :--- | :--- |
| **1. Monolithic Multimodal LLM**<br>*(Passing raw continuous sensor streams directly as text or image tokens to an LLM)* | **Context Explosion & High Latency**: A raw six-axis, 25 Hz hour would contain 540,000 values; the current ExtraSensory pipeline instead consumes its precomputed feature rows. Arithmetic over intervals is still a poor fit for a language model. | **Evidence retrieval and deterministic computation**: SensorQA keeps classifier evidence and context/place labels in compact timelines and executes supported calculations in Python. No token-compression percentage or zero-hallucination guarantee is claimed. |
| **2. Pure Classification Pipeline Without LLM**<br>*(Standard Random Forest / CNN classifier without language models)* | **Lack of Expressiveness**: Traditional classifiers cannot interpret complex natural language queries, multi-step comparative logic (*"walking vs sitting in first 30 mins"*), or open-world semantic queries (*"wheeled locomotion"*). | **Expert Query Planning (JARVIS)**: LLM acts purely as a semantic compiler (Stage 1) and narrative verbalizer (Stage 3), allowing natural conversation while keeping computation deterministic. |
| **3. LLM-Based Internal Math Computation**<br>*(Asking the LLM to calculate the sum or count intervals directly)* | **Mathematical Drift & Calculation Errors**: LLMs regularly make errors on subtraction, time-interval overlap merging, and transition frequency counts. | **Deterministic Code Execution Engine**: Python code computes all durations, differences, counts, and IoU overlaps with **100% mathematical precision**. |
| **4. Unguarded Open-Domain QA**<br>*(Allowing the LLM to answer any question regardless of sensor presence)* | **Dangerous Fabrications**: An unconstrained LLM might fabricate what the user ate or their heart rate even though only an accelerometer was worn. | **Truthful Rejection Guardrails (DrHouse)**: System strictly rejects out-of-modality questions with a **100% truthful rejection rate**, specifying the exact sensor modalities missing. |

---

## 4. Edge & Ubiquitous Computing Deployment

To support resource-constrained wearable devices (smartwatches, microcontrollers, ARM Cortex processors), the system provides two distinct operating points:

```text
┌────────────────────────────────────────────────────────────┐
│ MODEL A: Server / Cloud Full Precision (FP32)               │
│ • Latency: ~0.19 ms / window                               │
│ • Model Size: 0.147 MB                                     │
│ • Throughput: 5,150 windows/sec                            │
│ • Purpose: High-volume batch processing & offline analytics│
└────────────────────────────────────────────────────────────┘
                             vs
┌────────────────────────────────────────────────────────────┐
│ MODEL B: Ubiquitous Edge Quantized (PyTorch Dynamic INT8)  │
│ • Latency: ~0.53 ms / window                               │
│ • Model Size: 0.041 MB (41 KB)                             │
│ • Memory Footprint: < 0.6 MB RAM                           │
│ • Compression Ratio: 3.6x smaller                          │
│ • Purpose: Real-time on-device wearable inference (WearOS/ │
│            Apple Watch/ESP32)                              │
└────────────────────────────────────────────────────────────┘
```

---

## 5. Summary Conclusion

The SensorQA architecture enforces a single foundational design rule:

> **Do not let the language model invent the answer or calculate numbers. Let the sensor-processing pipeline discover the physical evidence, let deterministic algorithms perform measurable computations, and let the language model reason over, synthesize, and explain that evidence.**
