# Challenge 1 --- Core Research Conclusion

## 1. Problem Understanding

The challenge is **not simply an activity-classification problem**.

The required system is a **grounded, explainable activity
question-answering system over wearable sensor signals**.

The overall task is:

``` text
Sensor recording + Natural-language question
                    ↓
          Evidence-grounded answer
```

The answer must be supported by evidence from the sensor recording,
including the relevant temporal interval and sensor information.

The challenge requires a structured answer containing:

-   Answer
-   Activity/Event
-   Evidence
-   Timestamp(s)
-   Sensor Modality
-   Sensor Channel(s)
-   Explanation

------------------------------------------------------------------------

## 2. Required Dataset

The challenge specifies **ExtraSensory** as the dataset for development
and validation.

### Required sensor modalities

-   Accelerometer: X, Y, Z
-   Gyroscope: X, Y, Z

The sensor streams are to be resampled to **25 Hz**.

### Seven target activities/postures

1.  Lying down
2.  Sitting
3.  Standing in place
4.  Standing and moving
5.  Walking
6.  Running
7.  Bicycling

Therefore, the primary dataset for our implementation is:

> **ExtraSensory**

The other datasets appearing in the reference papers are not
replacements for ExtraSensory.

------------------------------------------------------------------------

## 3. Reference Papers and What We Learn From Them

### 3.1 Sensor2Text

**Main idea:** convert wearable sensor information into a learned
semantic representation that can interact with language.

Architecture:

``` text
Multiple Sensor Modalities
          ↓
     Sensor Encoder
          ↓
 Learned Sensor Representation
          ↓
     Language Decoder
          ↑
     Text Prompt
          ↓
     Text Response
```

The paper uses ActionSense and explores sensor-language interaction
through a sensor encoder and language decoder.

### What we should borrow

-   Sensor encoder concept
-   Multimodal sensor fusion
-   Temporal processing
-   Semantic representation of sensor windows
-   Evaluation on unseen users/generalization

### What we should not directly copy

-   ActionSense as our main dataset
-   Video-teacher setup
-   Kitchen-activity setup

Our challenge is specifically constrained to ExtraSensory,
accelerometer + gyroscope, and seven activities.

------------------------------------------------------------------------

## 4. SensorChat

**This is the most directly relevant reference paper.**

SensorChat proposes a three-stage pipeline:

``` text
User Question
      ↓
1. Question Decomposition
      ↓
2. Sensor Data Query
      ↓
3. Answer Assembly
      ↓
Final Answer
```

### Stage 1 --- Question decomposition

The natural-language question is converted into structured information
such as:

``` json
{
  "activity": "walking",
  "operation": "duration",
  "time_range": "specified range"
}
```

The decomposition identifies the context of interest, temporal
constraints, and the required query operation.

### Stage 2 --- Sensor data query

The complete raw sensor history is not directly passed to the LLM.

Instead:

``` text
Full Sensor History
        ↓
Sensor Encoder
        ↓
Sensor Embeddings
        ↓
Relevant Segment Retrieval
        ↓
Compact Sensor Context
```

SensorChat uses sensor/text representations and retrieves relevant
sensor segments.

### Stage 3 --- Answer assembly

The retrieved information is summarized and supplied to the final
language model.

For example:

``` text
Walking:
905–1420 s
2110–2295 s
```

The final answer can then be generated from this evidence.

### What we should borrow

-   Three-stage QA architecture
-   Question decomposition
-   Relevant sensor/evidence retrieval
-   Compact context instead of raw long-duration sensor input
-   Deterministic query functions
-   Separation between sensor understanding and language generation

SensorChat is especially relevant because it also uses **ExtraSensory
for sensor-text encoder pretraining**, with sensor samples aligned into
20-second windows.

------------------------------------------------------------------------

## 5. DrHouse

DrHouse focuses on reasoning using sensor evidence together with other
knowledge sources.

Conceptually:

``` text
User Input
    ↓
Evidence Retrieval
    ↓
Sensor Data Retrieval
    ↓
Knowledge Integration
    ↓
LLM Reasoning
    ↓
Decision
```

### Important idea for our project: uncertainty

Sensor data should not automatically be treated as perfect.

We should maintain information such as:

``` text
Activity: Walking
Confidence: 0.91
Start: 10:31:20
End: 10:32:40
```

This allows the QA system to take model confidence/evidence quality into
account.

### What we should borrow

-   Evidence-based reasoning
-   Confidence/uncertainty awareness
-   Retrieval before reasoning
-   Explicit integration of sensor evidence
-   Avoiding unsupported conclusions

### What we should not borrow

The medical knowledge/diagnosis components are specific to DrHouse and
are irrelevant to our activity-QA problem.

------------------------------------------------------------------------

## 6. JARVIS

JARVIS is particularly useful for **structured query planning**.

Its key principle is:

``` text
User Question
      ↓
Expert LLM
      ↓
Structured Instructions
      ↓
Data Retrieval
      ↓
Data Processing
      ↓
Response Generation
```

The Expert-LLM interprets the question and produces structured
instructions rather than directly producing the final answer.

### Important principle

The LLM should not perform numerical/statistical computation when a
deterministic program can do it.

For example:

> "How many times did I transition from sitting to walking?"

Instead of asking the LLM to count:

``` text
Activity Timeline
        ↓
Transition Detector
        ↓
Sitting → Walking
Sitting → Walking
Sitting → Walking
        ↓
Count = 3
```

The LLM then verbalizes the computed result.

### What we should borrow

-   Structured query planning
-   JSON-based intermediate representation
-   Separation of semantic reasoning and computation
-   Deterministic data processing
-   Truthful failure handling for unavailable information

------------------------------------------------------------------------

# 7. Combined Research Direction

The strongest approach is to combine the useful parts of the four papers
rather than reproduce one paper.

``` text
Sensor2Text
    ↓
Sensor representation
    +
SensorChat
    ↓
Question decomposition + evidence retrieval
    +
DrHouse
    ↓
Confidence/uncertainty-aware evidence reasoning
    +
JARVIS
    ↓
Structured query planning + deterministic processing
```

------------------------------------------------------------------------

# 8. Proposed Architecture

The proposed system is:

``` text
                  EXTRA SENSORY
                       │
          ┌────────────┴────────────┐
          │                         │
   Accelerometer               Gyroscope
      X/Y/Z                      X/Y/Z
          │                         │
          └────────────┬────────────┘
                       ↓
                Preprocessing
                 25 Hz / windows
                       ↓
              Sensor Encoder
                       ↓
            Activity Classifier
                       ↓
       ┌───────────────┴───────────────┐
       ↓                               ↓
 Activity Labels                  Confidence
       │                               │
       └───────────────┬───────────────┘
                       ↓
              Temporal Evidence
                     Store
                       ↓
               ┌────────────┐
User Question →│Query Planner│
               └─────┬──────┘
                     ↓
             Structured Query
                     ↓
             Evidence Retrieval
                     ↓
          Deterministic Operations
       ┌─────────┬──────────┬─────────┐
       ↓         ↓          ↓         ↓
     Count    Duration   Transition  Time
       └─────────┴──────────┴─────────┘
                     ↓
              Evidence Package
                     ↓
              Answer Generator
                     ↓
             Grounded Answer
```

------------------------------------------------------------------------

# 9. Why We Should Use a Two-Level System

We should separate the system into:

## Model A --- Sensor understanding

``` text
Accelerometer + Gyroscope
          ↓
     Sensor Model
          ↓
Activity + confidence + timestamp
```

## Model B --- Language reasoning

``` text
Question
   ↓
Query Planner
   ↓
Retrieved Activity Evidence
   ↓
Deterministic Computation
   ↓
Reasoning
   ↓
Grounded Answer
```

The LLM should therefore **not directly process the entire raw sensor
history**.

Instead, the sensor model creates a structured representation that the
query system can retrieve and process.

------------------------------------------------------------------------

# 10. Activity Timeline --- Central Data Structure

A key component should be an activity/evidence timeline.

Example:

``` json
{
  "activity": "walking",
  "start": 905,
  "end": 1420,
  "confidence": 0.93,
  "modality": [
    "accelerometer",
    "gyroscope"
  ],
  "channels": [
    "Acc X", "Acc Y", "Acc Z",
    "Gyro X", "Gyro Y", "Gyro Z"
  ]
}
```

Another interval:

``` json
{
  "activity": "running",
  "start": 1512,
  "end": 1980,
  "confidence": 0.96
}
```

This representation enables deterministic operations.

### Duration

``` text
SUM(end - start)
```

### Count

``` text
COUNT(activity intervals)
```

### Comparison

``` text
walking_duration vs running_duration
```

### Onset

``` text
FIRST(running_interval.start)
```

### Evidence

``` text
RETURN corresponding interval
```

This structure is central to achieving grounded answers.

------------------------------------------------------------------------

# 11. Example End-to-End Query

Question:

> "How long did I walk?"

Sensor model produces:

``` text
Walking:
905–1420 s
2110–2295 s
```

Query engine calculates:

``` text
1420 - 905 = 515 s
2295 - 2110 = 185 s

Total = 700 s
```

Final response:

``` text
Answer: 700 seconds

Activity/Event: Walking

Evidence:
Timestamp(s): 905–1420 s, 2110–2295 s

Sensor Modality:
Accelerometer + Gyroscope

Sensor Channel(s):
Acc X/Y/Z + Gyro X/Y/Z

Explanation:
Walking was detected during two intervals lasting
515 seconds and 185 seconds, giving a total of
700 seconds.
```

The important point is that the **700-second calculation is performed
deterministically**, while the language model is responsible for
interpreting the question and presenting the result.

------------------------------------------------------------------------

# 12. Tasks the System Must Support

The challenge can be viewed as four progressively harder levels.

### Task 1 --- Activity identification

Examples:

-   What activity is the user performing?
-   Is the user running?

### Task 2 --- Temporal and quantitative reasoning

Examples:

-   How long did the user walk?
-   How many times did the user run?
-   When did running begin?
-   Did the user spend more time walking or running?

### Task 3 --- Evidence grounding

The answer should identify:

-   Correct activity/event
-   Correct timestamp
-   Correct sensor modality
-   Correct sensor channels
-   Explanation connected to the evidence

### Task 4 --- Open-world activity reasoning

The system should handle semantic questions that may not simply ask for
one of the seven class labels.

Example:

> "Was the user using a wheeled or pedal-based mode of movement?"

The system should reason from the sensor evidence and produce a
justified conclusion.

------------------------------------------------------------------------

# 13. Evaluation Philosophy

The system should not be evaluated only by classification accuracy.

Important evaluation dimensions include:

-   Accuracy by question type
-   Activity confusion matrix
-   Precision / Recall / F1
-   Numeric accuracy under different tolerances
-   MAE/MAPE for numerical answers
-   Temporal grounding using IoU
-   Evidence/modality/channel correctness
-   Open-world reasoning quality
-   Robustness to sensor noise and missing samples
-   Computational cost

The challenge also requires reporting model size, memory, latency, and
other efficiency measurements.

------------------------------------------------------------------------

# 14. Edge Deployment Direction

A useful additional research direction is lightweight deployment.

We can compare at least two operating points:

``` text
Model A:
Full precision
Higher accuracy
Higher resource usage

Model B:
Quantized / distilled
Slightly lower accuracy
Lower memory and latency
```

Possible techniques include:

-   Quantization
-   Knowledge distillation
-   Pruning

This aligns well with the challenge's edge-efficiency component and with
SensorChat's edge variant.

------------------------------------------------------------------------

# 15. Final Research Position

We should **not claim** that we invented sensor-based QA.

Existing work already establishes:

-   Sensor-to-language modeling --- Sensor2Text
-   Long-term sensor QA --- SensorChat
-   Sensor-evidence reasoning --- DrHouse
-   Structured query planning and deterministic processing --- JARVIS

Our research should instead focus on integrating these ideas for the
specific challenge setting:

> **A grounded natural-language question-answering framework for
> activity reasoning over accelerometer and gyroscope data, combining
> sensor-based activity recognition, temporal evidence retrieval,
> structured query planning, confidence-aware reasoning, deterministic
> computation, and LLM-based answer generation.**

------------------------------------------------------------------------

# 16. Core Principle

The single most important design principle is:

> **Do not let the language model invent the answer. Let the
> sensor-processing pipeline discover the evidence, let deterministic
> components perform measurable computations, and let the language model
> reason over and explain that evidence.**

Therefore:

``` text
Sensor
  ↓
Activity
  ↓
Evidence
  ↓
Timeline
  ↓
Question
  ↓
Structured Query
  ↓
Deterministic Reasoning
  ↓
Grounded Evidence
  ↓
LLM Explanation
  ↓
Final Answer
```

This is the core conclusion from the challenge requirements and the four
reference papers.

------------------------------------------------------------------------

# 17. Location- and Context-Aware Extension

The project now uses ExtraSensory's existing context and place annotations
alongside the activity timeline. This is a QA/data-layer extension: it does
not yet condition or retrain the activity classifier on context or location.

## Data available and limits

The challenge specification's 25 Hz/20-second preprocessing description
is a target design, not the current classifier input path: this repository
uses the ExtraSensory archive's existing feature rows, typically spaced
about 60 seconds apart. The web waveform preview is synthesized and is not
used to classify the archived sessions.

An audit of the bundled ExtraSensory archive found 377,346 rows across 60
users. Examples of positive annotations include `LOC_home` for 57 users,
`LOC_main_workplace` for 32, `OR_indoors` for 59, and `AT_THE_GYM` for 8.
These counts are positive feature rows, not independent GPS fixes or
confirmed visits; label coverage is sparse and varies by user.

The loader retains multi-label contexts (for example indoors/outdoors,
in-a-car/on-a-bus, driving/passenger, with-friends/with-coworkers) and
coarse place labels (home, work, restaurant, gym, beach, school). The
archive contains location-derived summary fields, but this path does not
expose a raw coordinate trajectory. Therefore the system does not geocode
places, calculate exact routes/distances, or answer exact-address,
city, latitude/longitude, or GPS-coordinate questions.

## Supported deterministic questions

- Activity/context overlap, such as *"How long was I sitting in a car?"*
  and *"Did I walk while at work?"*
- Place-label dwell and observed visit counts, such as
  *"How long was I at the gym?"*
- Place-label transitions, such as *"How many times did I go from home to
  work?"*
- Listing observed coarse place labels for a session.

`ContextTimeline` and `LocationTimeline` keep explicit annotation coverage.
Unlabeled time is unknown, not a negative label. Incomplete-coverage
durations/counts are reported as observed lower bounds, and a negative
co-occurrence result becomes unknown when the label coverage is incomplete.
Place answers do not include raw coordinates and bypass external language
model calls.

## Implementation status

`src/context_store.py` builds the context/place timelines;
`src.query_planner.py` recognizes the added intents;
`src.deterministic_engine.py` performs interval intersections, dwell,
counts, and adjacent-label transitions. The FastAPI app exposes
`/api/session/{session_id}/context` and
`/api/session/{session_id}/location`.

The activity model has not been retrained with location-derived or context
features. Context-conditioned classification, transport-mode inference
from trajectories, multi-fold ablations, and independent QA benchmark
results remain future work; no performance improvement is claimed.
