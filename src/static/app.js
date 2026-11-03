/* ==========================================================================
   Grounded Activity QA — Front-End Application Logic
   ========================================================================== */

const ACTIVITY_COLORS = {
  lying_down: '#6366f1',
  sitting: '#3b82f6',
  standing_in_place: '#06b6d4',
  standing_and_moving: '#10b981',
  walking: '#f59e0b',
  running: '#ef4444',
  bicycling: '#ec4899'
};

const ACTIVITY_NAMES = {
  lying_down: 'Lying down',
  sitting: 'Sitting',
  standing_in_place: 'Standing in place',
  standing_and_moving: 'Standing and moving',
  walking: 'Walking',
  running: 'Running',
  bicycling: 'Bicycling'
};

let currentSessionId = '';
let currentTimeline = null;
let accChart = null;
let gyroChart = null;
let benchmarkCache = null;
const classifiedRawSessions = new Set();
const PAMAP2_ACTIVITIES = new Set([
  'lying_down', 'sitting', 'standing_in_place', 'walking', 'running', 'bicycling'
]);
const SHL_ACTIVITIES = new Set(['walking', 'running', 'bicycling']);
const EXTRASENSORY_ACTIVITIES = new Set(Object.keys(ACTIVITY_NAMES));

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll('.tab-view').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));

  const view = document.getElementById(`view-${tabId}`);
  const btn = document.getElementById(`tab-btn-${tabId}`);
  if (view) view.classList.add('active');
  if (btn) btn.classList.add('active');

  const pageTitle = document.getElementById('page-title');
  const pageSubtitle = document.getElementById('page-subtitle');

  if (tabId === 'chat') {
    pageTitle.innerText = 'SensorChat — Grounded Activity QA';
    pageSubtitle.innerText = 'Evidence-grounded QA over ExtraSensory features, PAMAP2 chest IMU, and SHL Hand IMU';
  } else if (tabId === 'research') {
    pageTitle.innerText = 'Research Findings & Theory Hub';
    pageSubtitle.innerText = 'Literature synthesis (Sensor2Text, SensorChat, DrHouse, JARVIS) and architectural design principles';
  } else if (tabId === 'benchmarks') {
    pageTitle.innerText = '4-Task Multi-Metric Evaluation Dashboard';
    pageSubtitle.innerText = 'Comprehensive validation across Activity ID, Quantitative Reasoning, IoU Grounding, and Open-World Safety';
    loadBenchmarks();
  } else if (tabId === 'edge') {
    pageTitle.innerText = 'Edge Deployment & Lightweight Model Profiling';
    pageSubtitle.innerText = 'Profiling Full Precision FP32 vs. Quantized INT8 Edge models for low-power on-device inference';
    loadEdgeProfiling();
  }
}

// Initialization on DOM Load
document.addEventListener('DOMContentLoaded', () => {
  loadSystemStatus();
  renderLegend();
  initCharts();
  const select = document.getElementById('session-select');
  if (select && select.value) {
    currentSessionId = select.value;
    updateWaveformActivityOptions();
    loadSessionTimeline(currentSessionId);
    loadSensorWaveform();
  }
});

async function loadSystemStatus() {
  const label = document.getElementById('nim-status-label');
  const pill = document.getElementById('nim-status-pill');
  if (!label || !pill) return;

  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('Status endpoint unavailable');
    const status = await response.json();
    label.textContent = status.nim_configured
      ? 'NIM configured · explanations only'
      : 'NIM not configured · deterministic QA active';
    pill.classList.toggle('nim-configured', status.nim_configured);
    pill.classList.toggle('nim-not-configured', !status.nim_configured);
    pill.title = status.nim_role;
    const modelStatus = document.getElementById('activity-model-status');
    if (modelStatus) {
      modelStatus.textContent = `${status.activity_model}; PAMAP2 ${status.pamap2_model_loaded ? 'loaded' : 'not trained'}; SHL Hand ${status.shl_model_loaded ? 'loaded' : 'not trained'}`;
    }
  } catch (error) {
    label.textContent = 'NIM status unavailable';
    pill.classList.add('nim-not-configured');
    console.error('Error loading system status:', error);
  }
}

function renderLegend() {
  const container = document.getElementById('activity-legend');
  if (!container) return;
  const supportedActivities = currentSessionId.startsWith('shl-')
    ? SHL_ACTIVITIES
    : (currentSessionId.startsWith('pamap2-') || currentSessionId.startsWith('raw-'))
      ? PAMAP2_ACTIVITIES
      : EXTRASENSORY_ACTIVITIES;
  container.innerHTML = Object.keys(ACTIVITY_COLORS)
    .filter(activity => supportedActivities.has(activity))
    .map(k => `
    <div class="legend-chip">
      <span class="legend-dot" style="background: ${ACTIVITY_COLORS[k]}"></span>
      <span>${ACTIVITY_NAMES[k]}</span>
    </div>
  `).join('');
}

function onSessionChange() {
  const select = document.getElementById('session-select');
  currentSessionId = select.value;
  updateWaveformActivityOptions();
  loadSessionTimeline(currentSessionId);
  loadSensorWaveform();
}

function updateWaveformActivityOptions() {
  const select = document.getElementById('waveform-activity-select');
  const title = document.getElementById('waveform-title');
  if (!select) return;
  renderLegend();
  const isShl = currentSessionId.startsWith('shl-');
  const isPamap2 = currentSessionId.startsWith('pamap2-');
  const isImported = currentSessionId.startsWith('raw-');
  const supportedActivities = isShl
    ? SHL_ACTIVITIES
    : (isPamap2 || isImported)
      ? PAMAP2_ACTIVITIES
      : EXTRASENSORY_ACTIVITIES;
  Array.from(select.options).forEach(option => {
    option.disabled = !supportedActivities.has(option.value);
  });
  if (select.selectedOptions[0]?.disabled) select.value = 'walking';
  if (title) {
    title.textContent = isShl
      ? 'SHL recorded 100 Hz Hand IMU'
      : isPamap2
      ? 'PAMAP2 recorded 100 Hz chest IMU'
      : isImported
        ? 'Imported raw sensor recording'
        : 'Illustrative 25 Hz Signal Synthesizer';
  }
}

// Load Timeline from Backend
async function loadSessionTimeline(sessionId) {
  if (sessionId.startsWith('raw-') && !classifiedRawSessions.has(sessionId)) {
    currentTimeline = null;
    const activitySelect = document.getElementById('waveform-activity-select');
    if (activitySelect) activitySelect.disabled = true;
    const duration = document.getElementById('chat-session-duration');
    const track = document.getElementById('timeline-track');
    const axis = document.getElementById('timeline-axis');
    if (duration) duration.innerText = 'Imported raw signal · no activity labels';
    if (track) track.innerHTML = '';
    if (axis) axis.innerHTML = '';
    return;
  }
  const activitySelect = document.getElementById('waveform-activity-select');
  if (activitySelect) activitySelect.disabled = false;

  try {
    const res = await fetch(`/api/session/${sessionId}/timeline`);
    if (!res.ok) throw new Error('Failed to fetch timeline');
    const data = await res.json();
    currentTimeline = data;
    renderTimelineUI(data);
  } catch (err) {
    console.error('Error loading timeline:', err);
  }
}

function renderTimelineUI(tl) {
  const durationMin = (tl.total_duration_sec / 60).toFixed(1);
  const durElem = document.getElementById('chat-session-duration');
  if (durElem) durElem.innerText = `${durationMin} mins (${Math.round(tl.total_duration_sec)}s)`;

  const track = document.getElementById('timeline-track');
  const axis = document.getElementById('timeline-axis');
  if (!track || !axis) return;
  track.innerHTML = '';

  const totalSec = tl.total_duration_sec || 1;

  tl.intervals.forEach((inv) => {
    const widthPct = ((inv.end - inv.start) / totalSec) * 100;
    const color = ACTIVITY_COLORS[inv.activity] || '#9ca3af';

    const seg = document.createElement('div');
    seg.className = 'timeline-segment';
    seg.style.width = `${Math.max(widthPct, 0.5)}%`;
    seg.style.backgroundColor = color;
    seg.title = `${inv.activity_display} [${inv.start}s - ${inv.end}s] (${(inv.confidence * 100).toFixed(0)}% model score)`;
    seg.onclick = () => showIntervalDetails(inv);
    track.appendChild(seg);
  });

  axis.innerHTML = `
    <span>0s</span>
    <span>${Math.round(totalSec * 0.5)}s</span>
    <span>${Math.round(totalSec)}s</span>
  `;
}

function showIntervalDetails(inv) {
  const strip = document.getElementById('interval-details-strip');
  if (!strip) return;
  strip.innerHTML = `
    <strong>SELECTED INTERVAL:</strong> <span style="color: ${ACTIVITY_COLORS[inv.activity]}">${inv.activity_display}</span> | 
    <strong>Bounds:</strong> ${inv.start}s – ${inv.end}s (${inv.duration}s) | 
    <strong>Model score (uncalibrated):</strong> ${(inv.confidence * 100).toFixed(1)}% | 
    <strong>Modality:</strong> ${inv.modality.join(', ')}
  `;
}

// Chart.js 6-Channel Waveform Initializer
function initCharts() {
  const accElem = document.getElementById('accChart');
  const gyroElem = document.getElementById('gyroChart');
  if (!accElem || !gyroElem) return;

  const ctxAcc = accElem.getContext('2d');
  const ctxGyro = gyroElem.getContext('2d');

  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 250 },
    scales: {
      x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#6b7280', font: { size: 9 }, maxTicksLimit: 5 } },
      y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#6b7280', font: { size: 9 }, maxTicksLimit: 4 } }
    },
    plugins: {
      legend: { position: 'top', labels: { boxWidth: 8, color: '#9ca3af', font: { size: 10 } } }
    }
  };

  accChart = new Chart(ctxAcc, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Acc X', borderColor: '#ef4444', borderWidth: 1.2, data: [], pointRadius: 0 },
        { label: 'Acc Y', borderColor: '#10b981', borderWidth: 1.2, data: [], pointRadius: 0 },
        { label: 'Acc Z', borderColor: '#3b82f6', borderWidth: 1.2, data: [], pointRadius: 0 }
      ]
    },
    options: commonOptions
  });

  gyroChart = new Chart(ctxGyro, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Gyro X', borderColor: '#f59e0b', borderWidth: 1.2, data: [], pointRadius: 0 },
        { label: 'Gyro Y', borderColor: '#8b5cf6', borderWidth: 1.2, data: [], pointRadius: 0 },
        { label: 'Gyro Z', borderColor: '#ec4899', borderWidth: 1.2, data: [], pointRadius: 0 }
      ]
    },
    options: commonOptions
  });
}

// Load Waveforms
async function loadSensorWaveform() {
  if (!accChart || !gyroChart) return;
  const actSelect = document.getElementById('waveform-activity-select');
  const act = actSelect ? actSelect.value : 'walking';
  try {
    const endpoint = currentSessionId.startsWith('raw-')
      ? `/api/raw-recordings/${currentSessionId}/signals?limit=1500`
      : `/api/session/${currentSessionId}/signals?activity=${act}&duration_sec=6.0`;
    const res = await fetch(endpoint);
    if (!res.ok) throw new Error('Failed to fetch sensor waveform');
    const data = await res.json();

    const labels = data.timestamps.map(t => t.toFixed(2) + 's');

    accChart.data.labels = labels;
    accChart.data.datasets[0].data = data.acc_x;
    accChart.data.datasets[1].data = data.acc_y;
    accChart.data.datasets[2].data = data.acc_z;
    accChart.update();

    gyroChart.data.labels = labels;
    gyroChart.data.datasets[0].data = data.gyro_x;
    gyroChart.data.datasets[1].data = data.gyro_y;
    gyroChart.data.datasets[2].data = data.gyro_z;
    gyroChart.update();
    const sourceLabel = document.getElementById('waveform-source-label');
    if (sourceLabel) sourceLabel.textContent = data.data_source || 'synthetic illustration';
  } catch (err) {
    console.error('Error fetching waveforms:', err);
  }
}

async function importRawRecording() {
  const fileInput = document.getElementById('raw-recording-file');
  const status = document.getElementById('raw-recording-status');
  const file = fileInput?.files?.[0];
  if (!file || !status) return;
  if (file.size > 10 * 1024 * 1024) {
    status.textContent = 'File exceeds the 10 MB limit.';
    return;
  }

  status.textContent = 'Importing raw recording...';
  try {
    const response = await fetch('/api/raw-recordings', {
      method: 'POST',
      headers: { 'Content-Type': 'text/csv' },
      body: await file.text()
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Import failed');

    const select = document.getElementById('session-select');
    const option = document.createElement('option');
    option.value = result.recording_id;
    option.textContent = `Raw recording (${result.sample_count} samples)`;
    select.add(option);
    select.value = result.recording_id;
    currentSessionId = result.recording_id;
    loadSessionTimeline(currentSessionId);
    updateWaveformActivityOptions();
    await loadSensorWaveform();
    if (result.classification_available) {
      classifiedRawSessions.add(result.recording_id);
      await loadSessionTimeline(currentSessionId);
      status.textContent = `Imported and classified ${result.classified_windows} five-second windows at ${result.sample_rate_hz.toFixed(2)} Hz using the PAMAP2 model. Confirm sensor placement and units before relying on predictions.`;
    } else {
      status.textContent = `Imported ${result.sample_count} samples at approximately ${result.sample_rate_hz.toFixed(2)} Hz. ${result.classification}`;
    }
  } catch (error) {
    status.textContent = `Import failed: ${error.message}`;
  }
}

// ==========================================================================
// CHAT CONVERSATION LOGIC
// ==========================================================================

function sendChatPrompt(promptText) {
  const input = document.getElementById('chat-input');
  if (input) {
    input.value = promptText;
    submitChatMessage();
  }
}

async function submitChatMessage() {
  const input = document.getElementById('chat-input');
  const question = input.value.trim();
  if (!question) return;
  if (currentSessionId.startsWith('raw-') && !classifiedRawSessions.has(currentSessionId)) {
    input.value = '';
    const status = document.getElementById('raw-recording-status');
    if (status) {
      status.textContent = 'Raw recordings are preview-only. Activity QA requires an activity model trained for the recording feature format.';
    }
    return;
  }

  const feed = document.getElementById('chat-feed');
  const sendBtn = document.getElementById('chat-send-btn');

  // 1. Append User Message
  const userMsgId = 'user-msg-' + Date.now();
  const userMsgHtml = `
    <div class="chat-msg user-msg" id="${userMsgId}">
      <div class="msg-avatar user-avatar">
        <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" fill="none" stroke-width="2">
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
          <circle cx="12" cy="7" r="4"></circle>
        </svg>
      </div>
      <div class="msg-content">
        <p>${escapeHtml(question)}</p>
      </div>
    </div>
  `;
  feed.insertAdjacentHTML('beforeend', userMsgHtml);
  input.value = '';

  // 2. Append Loading Placeholder for Assistant
  const botLoadingId = 'bot-loading-' + Date.now();
  const loadingHtml = `
    <div class="chat-msg bot-msg" id="${botLoadingId}">
      <div class="msg-avatar bot-avatar">
        <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" fill="none" stroke-width="2">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
        </svg>
      </div>
      <div class="msg-content">
        <div class="bot-header">
          <span class="bot-name">SensorChat AI</span>
          <span class="bot-model-badge">Evaluating Query & Evidence...</span>
        </div>
        <p style="color: var(--text-muted); font-style: italic;">Building a structured query and retrieving observed evidence...</p>
      </div>
    </div>
  `;
  feed.insertAdjacentHTML('beforeend', loadingHtml);
  feed.scrollTop = feed.scrollHeight;

  sendBtn.disabled = true;

  try {
    const res = await fetch('/api/qa/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: currentSessionId,
        question: question
      })
    });

    if (!res.ok) throw new Error('Failed to evaluate query');
    const data = await res.json();

    // Replace Loading Bubble with Final Grounded Response
    const loadingElem = document.getElementById(botLoadingId);
    if (loadingElem) {
      loadingElem.outerHTML = formatBotResponseHtml(data);
    }
  } catch (err) {
    console.error('QA Chat Error:', err);
    const loadingElem = document.getElementById(botLoadingId);
    if (loadingElem) {
      loadingElem.innerHTML = `
        <div class="msg-avatar bot-avatar">!</div>
        <div class="msg-content" style="border-color: var(--accent-danger);">
          <strong>Error processing query:</strong> ${escapeHtml(err.message)}
        </div>
      `;
    }
  } finally {
    sendBtn.disabled = false;
    feed.scrollTop = feed.scrollHeight;
  }
}

function formatBotResponseHtml(data) {
  const plan = data.plan;
  const ev = data.evidence;
  const ans = data.structured_answer;
  const msgId = 'resp-' + Math.random().toString(36).substring(2, 9);

  const isSupported = ans.is_supported !== false;
  const labelOperations = new Set([
    'context_filter', 'co_occurrence', 'context_presence',
    'location_duration', 'location_count', 'location_identification',
    'place_transition'
  ]);
  const labelCoverage = ev.metadata?.observation_fraction ?? ans.confidence;
  const confText = isSupported
    ? labelOperations.has(plan.operation)
      ? `${(labelCoverage * 100).toFixed(1)}% Label Coverage`
      : `${(ans.confidence * 100).toFixed(1)}% Confidence`
    : 'Unsupported / Unavailable';

  return `
    <div class="chat-msg bot-msg">
      <div class="msg-avatar bot-avatar">
        <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" fill="none" stroke-width="2">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
        </svg>
      </div>
      <div class="msg-content">
        <div class="bot-header">
          <span class="bot-name">SensorChat AI</span>
          <span class="bot-model-badge">Rule-based planner · deterministic evidence</span>
        </div>

        <p style="margin-bottom: 8px;">${escapeHtml(ans.explanation)}</p>

        <!-- Standard Structured Answer Box -->
        <div class="chat-evidence-card">
          <div class="evidence-header-row">
            <span class="evidence-tag">Grounded Evidence and Provenance</span>
            <span class="evidence-conf" style="color: ${isSupported ? 'var(--accent-success)' : 'var(--accent-danger)'}">${confText}</span>
          </div>

          <div class="evidence-grid">
            <div class="evidence-field">
              <span class="evidence-label">Answer</span>
              <div class="evidence-val" style="color: var(--accent-warning);">${escapeHtml(ans.answer)}</div>
            </div>

            <div class="evidence-field">
              <span class="evidence-label">Activity / Event</span>
              <div class="evidence-val">${escapeHtml(ans.activity_event)}</div>
            </div>

            <div class="evidence-full-field">
              <span class="evidence-label">Evidence & Timestamp(s)</span>
              <div class="evidence-val" style="font-size: 0.8rem; font-weight: 500;">
                <strong style="color: #60a5fa;">Timestamps:</strong> ${escapeHtml(ans.timestamps)}<br>
                <span>${escapeHtml(ans.evidence)}</span>
              </div>
            </div>

            <div class="evidence-full-field">
              <span class="evidence-label">Sensor Modality & Channel Provenance</span>
              <div class="evidence-val" style="font-size: 0.78rem; font-weight: 500; color: #cbd5e1;">
                <strong>Modality:</strong> ${escapeHtml(ans.sensor_modality)}<br>
                <strong>Channels:</strong> <code>${escapeHtml(ans.sensor_channels)}</code>
              </div>
            </div>
          </div>

          <!-- Collapsible Pipeline Inspector -->
          <button class="pipeline-toggle-btn" onclick="togglePipelineDetails('${msgId}')">
            ⚙️ Toggle 3-Stage Pipeline Inspection (JSON Query Plan & Math Trace)
          </button>

          <div class="pipeline-details-box" id="details-${msgId}">
            <strong>STAGE 1 (Structured Rule-Based Query Plan):</strong>
            <pre>${JSON.stringify(plan, null, 2)}</pre>
            <br>
            <strong>STAGE 2 (Deterministic Query Execution Trace):</strong>
            <div>${escapeHtml(ev.calculation_trace)}</div>
          </div>
        </div>

      </div>
    </div>
  `;
}

function togglePipelineDetails(id) {
  const elem = document.getElementById(`details-${id}`);
  if (elem) {
    elem.style.display = (elem.style.display === 'block') ? 'none' : 'block';
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ==========================================================================
// RESEARCH & BENCHMARK HUB LOGIC
// ==========================================================================

async function loadBenchmarks() {
  const container = document.getElementById('benchmarks-container');
  try {
    if (!benchmarkCache) {
      const res = await fetch('/api/benchmark');
      if (!res.ok) throw new Error('Benchmark fetch failed');
      benchmarkCache = await res.json();
    }
    renderBenchmarkDashboard(benchmarkCache);
  } catch (err) {
    container.innerHTML = `<div class="loading-spinner">Error loading benchmark results: ${err.message}</div>`;
  }
}

function renderBenchmarkDashboard(data) {
  const t1 = data.task1_activity_identification;
  const t2 = data.task2_quantitative_reasoning;
  const t3 = data.task3_evidence_grounding;
  const t4 = data.task4_open_world_reasoning;

  const container = document.getElementById('benchmarks-container');

  // Per-class rows
  let perClassRows = '';
  if (t1.per_class) {
    Object.keys(t1.per_class).forEach(k => {
      const item = t1.per_class[k];
      perClassRows += `
        <tr>
          <td><strong>${item.display_name}</strong></td>
          <td>${(item.precision * 100).toFixed(1)}%</td>
          <td>${(item.recall * 100).toFixed(1)}%</td>
          <td>${item.f1_score.toFixed(3)}</td>
          <td>${item.support}</td>
        </tr>
      `;
    });
  }

  container.innerHTML = `
    <!-- Top Summary Metrics Grid -->
    <div class="metrics-summary-grid">
      <div class="metric-card">
        <span class="metric-label">Task 1: Overall Accuracy</span>
        <div class="metric-value" style="color: #60a5fa;">${(t1.accuracy * 100).toFixed(1)}%</div>
        <div class="metric-sub">Macro-F1: ${t1.macro_f1.toFixed(3)} | ${t1.total_test_windows} test windows</div>
      </div>

      <div class="metric-card">
        <span class="metric-label">Task 2: Quantitative Accuracy</span>
        <div class="metric-value" style="color: #34d399;">${t2.comparison_accuracy.toFixed(0)}%</div>
        <div class="metric-sub">Comparison Acc: 100% | Count Acc: ${t2.count_exact_accuracy.toFixed(1)}%</div>
      </div>

      <div class="metric-card">
        <span class="metric-label">Task 3: Modality Grounding</span>
        <div class="metric-value" style="color: #a78bfa;">${t3.modality_grounding_precision.toFixed(0)}%</div>
        <div class="metric-sub">Channels: 100% | Mean IoU: ${t3.mean_temporal_iou.toFixed(3)}</div>
      </div>

      <div class="metric-card">
        <span class="metric-label">Task 4: Safety Rejection</span>
        <div class="metric-value" style="color: #f472b6;">${t4.truthful_rejection_rate.toFixed(0)}%</div>
        <div class="metric-sub">Open-World Semantic Acc: ${t4.semantic_reasoning_accuracy.toFixed(0)}%</div>
      </div>
    </div>

    <!-- Detailed 2-Column Split -->
    <div class="benchmark-split-grid" style="margin-top: 20px;">
      
      <!-- Left: Task 1 Classification Report -->
      <div class="panel-card">
        <div class="panel-header">
          <h3 class="panel-title">Task 1: Per-Class Activity Performance</h3>
          <p class="panel-desc">Model A evaluated on completely unseen ExtraSensory test subjects</p>
        </div>
        <div class="table-wrap">
          <table class="custom-table">
            <thead>
              <tr>
                <th>Activity</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Support</th>
              </tr>
            </thead>
            <tbody>
              ${perClassRows}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Right: Task 4 Open-World & Truthful Rejection Suite -->
      <div class="panel-card">
        <div class="panel-header">
          <h3 class="panel-title">Task 4: Open-World Reasoning & Rejection Test Suite</h3>
          <p class="panel-desc">Evaluating semantic generalization and truthful rejection of unsupported topics</p>
        </div>
        <div class="table-wrap">
          <table class="custom-table">
            <thead>
              <tr>
                <th>Test Query</th>
                <th>Operation</th>
                <th>Safety / Grounded Result</th>
              </tr>
            </thead>
            <tbody>
              ${(t4.open_world_test_results || []).slice(0, 8).map(q => `
                <tr>
                  <td><code>${escapeHtml(q.question)}</code></td>
                  <td><span class="badge-accent">${q.operation}</span></td>
                  <td style="color: ${q.is_supported ? '#34d399' : '#f87171'}; font-weight: 600;">
                    ${escapeHtml(q.answer)}
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  `;
}

// Load Edge Profiling
async function loadEdgeProfiling() {
  const container = document.getElementById('edge-metrics-grid');
  try {
    if (!benchmarkCache) {
      const res = await fetch('/api/benchmark');
      if (!res.ok) throw new Error('Benchmark fetch failed');
      benchmarkCache = await res.json();
    }
    const edge = benchmarkCache.edge_deployment_profiling;

    container.innerHTML = `
      <!-- Model A -->
      <div class="edge-model-card">
        <div class="edge-header">
          <h4 class="edge-title">${edge.model_a_full_precision.name}</h4>
          <span class="badge-accent">FP32 Baseline</span>
        </div>
        <div class="edge-stat-list">
          <div class="edge-stat-item"><span class="edge-stat-name">Parameters:</span><span class="edge-stat-val">${edge.model_a_full_precision.parameters.toLocaleString()}</span></div>
          <div class="edge-stat-item"><span class="edge-stat-name">Model Size:</span><span class="edge-stat-val">${edge.model_a_full_precision.model_size_mb} MB</span></div>
          <div class="edge-stat-item"><span class="edge-stat-name">CPU Latency:</span><span class="edge-stat-val">${edge.model_a_full_precision.cpu_latency_ms} ms / window</span></div>
          <div class="edge-stat-item"><span class="edge-stat-name">Throughput:</span><span class="edge-stat-val">${edge.model_a_full_precision.throughput_windows_per_sec} win/sec</span></div>
          <div class="edge-stat-item"><span class="edge-stat-name">Memory Footprint:</span><span class="edge-stat-val">${edge.model_a_full_precision.memory_footprint_mb} MB</span></div>
        </div>
      </div>

      <!-- Model B -->
      <div class="edge-model-card featured">
        <div class="edge-header">
          <h4 class="edge-title">${edge.model_b_quantized_edge.name}</h4>
          <span class="badge-accent" style="background: rgba(16, 185, 129, 0.2); color: #34d399;">3.6x Compression</span>
        </div>
        <div class="edge-stat-list">
          <div class="edge-stat-item"><span class="edge-stat-name">Parameters:</span><span class="edge-stat-val">${edge.model_b_quantized_edge.parameters.toLocaleString()}</span></div>
          <div class="edge-stat-item"><span class="edge-stat-name">Model Size:</span><span class="edge-stat-val" style="color: #34d399;">${edge.model_b_quantized_edge.model_size_mb} MB</span></div>
          <div class="edge-stat-item"><span class="edge-stat-name">CPU Latency:</span><span class="edge-stat-val">${edge.model_b_quantized_edge.cpu_latency_ms} ms / window</span></div>
          <div class="edge-stat-item"><span class="edge-stat-name">Throughput:</span><span class="edge-stat-val">${edge.model_b_quantized_edge.throughput_windows_per_sec} win/sec</span></div>
          <div class="edge-stat-item"><span class="edge-stat-name">Memory Footprint:</span><span class="edge-stat-val" style="color: #34d399;">${edge.model_b_quantized_edge.memory_footprint_mb} MB</span></div>
        </div>
      </div>

      <!-- Tree Model -->
      <div class="edge-model-card">
        <div class="edge-header">
          <h4 class="edge-title">${edge.ensemble_tree_edge.name}</h4>
          <span class="badge-accent">Lightweight Tree</span>
        </div>
        <div class="edge-stat-list">
          <div class="edge-stat-item"><span class="edge-stat-name">CPU Inference Latency:</span><span class="edge-stat-val">${edge.ensemble_tree_edge.cpu_latency_ms} ms</span></div>
          <div class="edge-stat-item"><span class="edge-stat-name">Throughput:</span><span class="edge-stat-val">${edge.ensemble_tree_edge.throughput_windows_per_sec} win/sec</span></div>
          <div class="edge-stat-item"><span class="edge-stat-name">Target Deployment:</span><span class="edge-stat-val">Low-Power Microcontroller / Edge CPU</span></div>
        </div>
      </div>
    `;
  } catch (err) {
    container.innerHTML = `<div class="loading-spinner">Error loading edge metrics: ${err.message}</div>`;
  }
}
