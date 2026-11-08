/**
 * DeepPCB Inspect AI — Frontend Logic & Visualizations
 * Pair-programmed for Machine Learning Minor Degree Final Project
 */

// State Container
const AppState = {
  currentTab: 'inspector',
  samples: [],
  selectedSample: null,
  uploadedFile: null,
  currentResult: null,
  viewMode: 'annotated', // 'annotated' or 'original'
  zoomLevel: 1.0,
  confidenceThreshold: 0.25,
  iouThreshold: 0.45,
  metricsData: null,
  charts: {}
};

// Defect definitions matching fine-tuned YOLOv8s model
const CLASS_COLORS = {
  missing_hole: '#06b6d4',
  mouse_bite: '#10b981',
  open_circuit: '#ef4444',
  short: '#3b82f6',
  spur: '#f59e0b',
  spurious_copper: '#8b5cf6'
};

const CLASS_NAMES = [
  'missing_hole',
  'mouse_bite',
  'open_circuit',
  'short',
  'spur',
  'spurious_copper'
];

const CLASS_DISPLAY_NAMES = {
  missing_hole: 'Missing Hole',
  mouse_bite: 'Mouse Bite',
  open_circuit: 'Open Circuit',
  short: 'Short Circuit',
  spur: 'Spur',
  spurious_copper: 'Spurious Copper'
};

// Document Ready
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  checkModelStatus();
  loadSamples();
  setupEventListeners();
  loadMetricsData();
});

// -------------------------------------------------------------
// Model Status & Reload
// -------------------------------------------------------------
async function checkModelStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    const badge = document.getElementById('modelStatusBadge');
    const statusText = document.getElementById('modelStatusText');
    const banner = document.getElementById('modelNoticeBanner');
    
    if (data.status === 'ready') {
      badge.className = 'status-badge active';
      statusText.textContent = `Live Model Active (${data.model_source} on ${data.device})`;
      if (banner) {
        banner.className = 'alert-banner success';
        banner.innerHTML = `
          <div class="alert-banner-content">
            <span style="font-size: 1.2rem;">✅</span>
            <div>
              <strong>Trained YOLOv8s Model Active:</strong> Running real-time PyTorch inference with <code>${data.weights_filename}</code> on <strong>${data.device}</strong> (Port 8000).
            </div>
          </div>
        `;
      }
    } else {
      badge.className = 'status-badge simulation';
      statusText.textContent = 'Simulation Mode (Awaiting best.pt)';
      if (banner) {
        banner.className = 'alert-banner';
        banner.innerHTML = `
          <div class="alert-banner-content">
            <span style="font-size: 1.2rem;">ℹ️</span>
            <div>
              <strong>Demo Simulation Mode:</strong> Place your trained <code>best.pt</code> into the <code>weights/</code> folder to activate live PyTorch inference. Pre-packaged DeepPCB test samples are fully interactive!
            </div>
          </div>
          <button class="btn btn-outline btn-sm" onclick="reloadModel()" id="reloadModelBtn">
            🔄 Reload Model
          </button>
        `;
      }
    }
  } catch (err) {
    console.error('Error checking status:', err);
  }
}

async function reloadModel() {
  const btn = document.getElementById('reloadModelBtn');
  if (btn) btn.innerHTML = '⏳ Loading...';
  try {
    const res = await fetch('/api/reload-model', { method: 'POST' });
    const data = await res.json();
    alert(data.message);
    checkModelStatus();
  } catch (err) {
    alert('Error reloading model: ' + err.message);
  }
}

// -------------------------------------------------------------
// Tabs Navigation
// -------------------------------------------------------------
function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.style.display = 'none';
      });

      const activePane = document.getElementById(`tab-${targetTab}`);
      if (activePane) {
        activePane.style.display = 'block';
        AppState.currentTab = targetTab;

        if (targetTab === 'performance' && AppState.metricsData) {
          renderCharts(AppState.metricsData);
        }
      }
    });
  });
}

// -------------------------------------------------------------
// Samples Loader
// -------------------------------------------------------------
async function loadSamples() {
  try {
    const res = await fetch('/api/samples');
    const samples = await res.json();
    AppState.samples = samples;
    const container = document.getElementById('sampleGrid');
    if (!container) return;
    container.innerHTML = '';

    samples.forEach((sample, idx) => {
      const card = document.createElement('div');
      card.className = `sample-card ${idx === 0 ? 'active' : ''}`;
      card.setAttribute('data-filename', sample.filename);

      const badgeClass = sample.status === 'PASS' ? 'pass' : (sample.status === 'CRITICAL' ? 'critical' : 'warning');
      const defCount = sample.expected_defects !== undefined ? `${sample.expected_defects} Def.` : (sample.defect_count !== undefined ? `${sample.defect_count} Def.` : '');

      card.innerHTML = `
        <img src="${sample.path}" class="sample-thumb" alt="${sample.title}">
        <div class="sample-info">
          <span class="sample-title" title="${sample.title}">${sample.title.split('—')[0]}</span>
          <span class="sample-badge ${badgeClass}">${defCount}</span>
        </div>
      `;

      card.addEventListener('click', () => {
        document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        AppState.selectedSample = sample.filename;
        AppState.uploadedFile = null;
        runInspection();
      });

      container.appendChild(card);
    });

    if (samples.length > 0) {
      AppState.selectedSample = samples[0].filename;
      runInspection();
    }
  } catch (err) {
    console.error('Error loading samples:', err);
  }
}

// -------------------------------------------------------------
// Event Listeners (Upload, Sliders, Buttons)
// -------------------------------------------------------------
function setupEventListeners() {
  // Confidence Slider
  const confSlider = document.getElementById('confSlider');
  const confVal = document.getElementById('confValue');
  if (confSlider && confVal) {
    confSlider.addEventListener('input', (e) => {
      confVal.textContent = parseFloat(e.target.value).toFixed(2);
      AppState.confidenceThreshold = parseFloat(e.target.value);
    });
    confSlider.addEventListener('change', () => runInspection());
  }

  // IoU Slider
  const iouSlider = document.getElementById('iouSlider');
  const iouVal = document.getElementById('iouValue');
  if (iouSlider && iouVal) {
    iouSlider.addEventListener('input', (e) => {
      iouVal.textContent = parseFloat(e.target.value).toFixed(2);
      AppState.iouThreshold = parseFloat(e.target.value);
    });
    iouSlider.addEventListener('change', () => runInspection());
  }

  // File Upload Dropzone
  const dropzone = document.getElementById('uploadDropzone');
  const fileInput = document.getElementById('fileInput');

  if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileUpload(e.target.files[0]);
      }
    });

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });
  }

  // View Mode Toggles (Annotated vs Original)
  const btnAnnotated = document.getElementById('btnViewAnnotated');
  const btnOriginal = document.getElementById('btnViewOriginal');
  if (btnAnnotated && btnOriginal) {
    btnAnnotated.addEventListener('click', () => {
      btnAnnotated.classList.add('active');
      btnOriginal.classList.remove('active');
      AppState.viewMode = 'annotated';
      updateCanvasView();
    });

    btnOriginal.addEventListener('click', () => {
      btnOriginal.classList.add('active');
      btnAnnotated.classList.remove('active');
      AppState.viewMode = 'original';
      updateCanvasView();
    });
  }

  // Zoom Controls
  const btnZoomIn = document.getElementById('btnZoomIn');
  const btnZoomOut = document.getElementById('btnZoomOut');
  const btnZoomReset = document.getElementById('btnZoomReset');
  const canvasImg = document.getElementById('pcbCanvasImg');

  if (btnZoomIn && canvasImg) {
    btnZoomIn.addEventListener('click', () => {
      AppState.zoomLevel = Math.min(AppState.zoomLevel + 0.25, 3.0);
      canvasImg.style.transform = `scale(${AppState.zoomLevel})`;
    });
  }
  if (btnZoomOut && canvasImg) {
    btnZoomOut.addEventListener('click', () => {
      AppState.zoomLevel = Math.max(AppState.zoomLevel - 0.25, 0.75);
      canvasImg.style.transform = `scale(${AppState.zoomLevel})`;
    });
  }
  if (btnZoomReset && canvasImg) {
    btnZoomReset.addEventListener('click', () => {
      AppState.zoomLevel = 1.0;
      canvasImg.style.transform = 'scale(1.0)';
    });
  }
}

function handleFileUpload(file) {
  AppState.uploadedFile = file;
  AppState.selectedSample = null;
  document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
  runInspection();
}

// -------------------------------------------------------------
// Run Inspection (/api/predict)
// -------------------------------------------------------------
async function runInspection() {
  const loadingIndicator = document.getElementById('canvasLoading');
  if (loadingIndicator) loadingIndicator.style.display = 'flex';

  const formData = new FormData();
  formData.append('confidence', AppState.confidenceThreshold);
  formData.append('iou', AppState.iouThreshold);

  if (AppState.uploadedFile) {
    formData.append('image', AppState.uploadedFile);
  } else if (AppState.selectedSample) {
    formData.append('sample_name', AppState.selectedSample);
  } else {
    if (loadingIndicator) loadingIndicator.style.display = 'none';
    return;
  }

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData
    });

    const result = await res.json();
    if (!result.success) {
      alert('Inspection error: ' + (result.error || 'Unknown error'));
      return;
    }

    AppState.currentResult = result;
    updateUIWithResult(result);
  } catch (err) {
    console.error('Error during inspection:', err);
  } finally {
    if (loadingIndicator) loadingIndicator.style.display = 'none';
  }
}

// -------------------------------------------------------------
// Update UI with Inspection Results
// -------------------------------------------------------------
function updateUIWithResult(result) {
  // Update Canvas Image
  updateCanvasView();

  // Update Verdict Banner
  const verdictCard = document.getElementById('verdictCard');
  const verdictTitle = document.getElementById('verdictTitle');
  const verdictDesc = document.getElementById('verdictDesc');
  const verdictCount = document.getElementById('verdictCount');

  if (verdictCard && verdictTitle && verdictDesc && verdictCount) {
    verdictCard.className = `verdict-card ${result.verdict_badge}`;
    const icon = result.verdict_badge === 'success' ? '✅' : (result.verdict_badge === 'warning' ? '⚠️' : '🚨');
    verdictTitle.innerHTML = `${icon} INSPECTION ${result.verdict}`;
    verdictDesc.textContent = result.verdict_desc;
    verdictCount.textContent = result.total_defects;
  }

  // Update Speed & Metadata
  const latencyBadge = document.getElementById('inspectionLatency');
  if (latencyBadge) {
    latencyBadge.textContent = `${result.inference_latency_ms} ms`;
  }

  // Update Class Defect Pills
  const pillsContainer = document.getElementById('defectPills');
  if (pillsContainer) {
    pillsContainer.innerHTML = '';
    CLASS_NAMES.forEach(c => {
      const count = result.class_counts[c] || 0;
      if (count > 0 || result.total_defects === 0) {
        const pill = document.createElement('div');
        pill.className = 'defect-pill';
        const disp = CLASS_DISPLAY_NAMES[c] || c.replace('_', ' ');
        pill.innerHTML = `
          <span class="pill-dot" style="background-color: ${CLASS_COLORS[c]}"></span>
          <span>${disp}</span>
          <span class="pill-count" style="color: ${CLASS_COLORS[c]}">${count}</span>
        `;
        pillsContainer.appendChild(pill);
      }
    });
  }

  // Populate Defect Ledger Table
  const tableBody = document.querySelector('#defectTable tbody');
  if (tableBody) {
    tableBody.innerHTML = '';

    if (result.detections.length === 0) {
      tableBody.innerHTML = `
        <tr>
          <td colspan="7" style="text-align: center; padding: 2rem; color: #166534; font-weight: 600;">
            🌟 Perfect Quality! No defects identified on this printed circuit board.
          </td>
        </tr>
      `;
    } else {
      result.detections.forEach(det => {
        const row = document.createElement('tr');
        const dispName = det.display_name || CLASS_DISPLAY_NAMES[det.class] || det.class.replace('_', ' ');
        row.innerHTML = `
          <td><strong>#${det.id}</strong></td>
          <td>
            <span class="badge-class" style="color: ${det.color}">
              <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${det.color};"></span>
              ${dispName}
            </span>
          </td>
          <td><strong>${det.confidence_pct}%</strong></td>
          <td><code>[${det.bbox.join(', ')}]</code></td>
          <td>${det.area} px²</td>
          <td><span class="badge-severity ${det.severity}">${det.severity}</span></td>
          <td style="font-size:0.75rem; color:#64748b;">${det.description}</td>
        `;
        tableBody.appendChild(row);
      });
    }
  }
}

function updateCanvasView() {
  const canvasImg = document.getElementById('pcbCanvasImg');
  if (!canvasImg || !AppState.currentResult) return;

  if (AppState.viewMode === 'annotated') {
    canvasImg.src = AppState.currentResult.annotated_image_base64;
  } else {
    canvasImg.src = AppState.currentResult.raw_image_base64;
  }
}

// -------------------------------------------------------------
// Load Metrics Data & Charts
// -------------------------------------------------------------
async function loadMetricsData() {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();
    AppState.metricsData = data;
    populateMetricsUI(data);
  } catch (err) {
    console.error('Error loading metrics:', err);
  }
}

function populateMetricsUI(data) {
  // Fill KPI cards
  const elMap50 = document.getElementById('kpiMap50');
  const elPrecision = document.getElementById('kpiPrecision');
  const elRecall = document.getElementById('kpiRecall');
  const elSpeed = document.getElementById('kpiSpeed');
  const elParams = document.getElementById('kpiParams');

  if (elMap50) elMap50.textContent = `${data.summary.mAP50_pct}%`;
  if (elPrecision) elPrecision.textContent = `${data.summary.precision_pct}%`;
  if (elRecall) elRecall.textContent = `${data.summary.recall_pct}%`;
  if (elSpeed) elSpeed.textContent = `${data.summary.inference_time_ms} ms`;
  if (elParams) elParams.textContent = data.summary.params_formatted;

  // Build confusion matrix table
  renderConfusionMatrix(data.confusion_matrix);

  // Render per-class performance table in Tab 2
  renderClassMetricsTable(data.per_class);

  // If already on performance tab, render charts immediately
  if (AppState.currentTab === 'performance') {
    renderCharts(data);
  }
}

function renderClassMetricsTable(classList) {
  const tbody = document.querySelector('#classMetricsTable tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  classList.forEach(cls => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>
        <span style="display:inline-flex; align-items:center; gap:6px; font-weight:700; color:${cls.color}">
          <span style="width:8px; height:8px; border-radius:50%; background:${cls.color}"></span>
          ${cls.display_name}
        </span>
      </td>
      <td><strong>${cls.ap50_pct}%</strong></td>
      <td>${(cls.precision * 100).toFixed(1)}%</td>
      <td>${(cls.recall * 100).toFixed(1)}%</td>
      <td>${cls.gt_total}</td>
      <td>${cls.images}</td>
      <td>${cls.missed}</td>
      <td>
        <div style="display:flex; align-items:center; gap:8px;">
          <div style="width:60px; height:6px; background:#e2e8f0; border-radius:3px; overflow:hidden;">
            <div style="width:${cls.miss_rate_pct * 3.5}%; height:100%; background:${cls.miss_rate_pct > 15 ? '#ef4444' : (cls.miss_rate_pct > 5 ? '#f59e0b' : '#10b981')}"></div>
          </div>
          <span style="font-size:0.75rem; font-weight:700;">${cls.miss_rate_pct}%</span>
        </div>
      </td>
    `;
    tbody.appendChild(row);
  });
}

function renderConfusionMatrix(matrixData) {
  const container = document.getElementById('confusionMatrixGrid');
  if (!container) return;
  container.innerHTML = '';

  // Top header row
  const emptyCorner = document.createElement('div');
  emptyCorner.className = 'matrix-cell matrix-header';
  emptyCorner.textContent = 'True \\ Pred';
  container.appendChild(emptyCorner);

  matrixData.classes.forEach(c => {
    const colHead = document.createElement('div');
    colHead.className = 'matrix-cell matrix-header';
    colHead.textContent = (CLASS_DISPLAY_NAMES[c] || c.replace('_', ' ')).slice(0, 5);
    colHead.title = c;
    container.appendChild(colHead);
  });

  // Rows
  matrixData.classes.forEach((rowClass, rIdx) => {
    const rowHead = document.createElement('div');
    rowHead.className = 'matrix-cell matrix-header';
    rowHead.textContent = (CLASS_DISPLAY_NAMES[rowClass] || rowClass.replace('_', ' ')).slice(0, 6);
    rowHead.title = rowClass;
    container.appendChild(rowHead);

    matrixData.matrix[rIdx].forEach((val, cIdx) => {
      const cell = document.createElement('div');
      cell.className = 'matrix-cell';
      const pct = (val * 100).toFixed(0);
      cell.textContent = `${pct}%`;

      // Color intensity based on accuracy
      if (rIdx === cIdx) {
        cell.style.backgroundColor = `rgba(37, 99, 235, ${Math.max(val, 0.15)})`;
        cell.style.color = val > 0.5 ? '#ffffff' : '#0f172a';
      } else {
        cell.style.backgroundColor = val > 0 ? `rgba(239, 68, 68, ${val * 1.5})` : '#f8fafc';
        cell.style.color = val > 0.05 ? '#991b1b' : '#94a3b8';
      }
      container.appendChild(cell);
    });
  });
}

// -------------------------------------------------------------
// Chart.js Visualizations
// -------------------------------------------------------------
function renderCharts(metrics) {
  // Chart 1: Per-Class AP@50 Bar Chart
  const ctxAp = document.getElementById('chartClassAp');
  if (ctxAp && !AppState.charts.classAp) {
    AppState.charts.classAp = new Chart(ctxAp, {
      type: 'bar',
      data: {
        labels: metrics.per_class.map(c => c.display_name),
        datasets: [{
          label: 'mAP@50 (%)',
          data: metrics.per_class.map(c => c.ap50_pct),
          backgroundColor: metrics.per_class.map(c => c.color),
          borderRadius: 6,
          barThickness: 28
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ` AP@50: ${ctx.raw}%`
            }
          }
        },
        scales: {
          y: {
            min: 90,
            max: 100,
            ticks: { callback: v => `${v}%` },
            grid: { color: '#f1f5f9' }
          },
          x: {
            grid: { display: false }
          }
        }
      }
    });
  }

  // Chart 2: Training Loss Progression
  const ctxLoss = document.getElementById('chartTrainingLoss');
  if (ctxLoss && !AppState.charts.trainingLoss) {
    const epochs = metrics.training_epochs.map(e => `Ep ${e.epoch}`);
    AppState.charts.trainingLoss = new Chart(ctxLoss, {
      type: 'line',
      data: {
        labels: epochs,
        datasets: [
          {
            label: 'Box Loss',
            data: metrics.training_epochs.map(e => e.box_loss),
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239, 68, 68, 0.05)',
            tension: 0.3,
            pointRadius: metrics.training_epochs.length > 50 ? 0 : 2,
            borderWidth: 2
          },
          {
            label: 'Class Loss',
            data: metrics.training_epochs.map(e => e.cls_loss),
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.05)',
            tension: 0.3,
            pointRadius: metrics.training_epochs.length > 50 ? 0 : 2,
            borderWidth: 2
          },
          {
            label: 'DFL Loss',
            data: metrics.training_epochs.map(e => e.dfl_loss),
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.05)',
            tension: 0.3,
            pointRadius: metrics.training_epochs.length > 50 ? 0 : 2,
            borderWidth: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top' }
        },
        scales: {
          y: { grid: { color: '#f1f5f9' } },
          x: { grid: { display: false } }
        }
      }
    });
  }

  // Chart 3: Validation mAP Progression
  const ctxMap = document.getElementById('chartMapProgression');
  if (ctxMap && !AppState.charts.mapProgression) {
    const epochs = metrics.training_epochs.map(e => `Ep ${e.epoch}`);
    AppState.charts.mapProgression = new Chart(ctxMap, {
      type: 'line',
      data: {
        labels: epochs,
        datasets: [
          {
            label: 'Validation mAP@50 (%)',
            data: metrics.training_epochs.map(e => (e.mAP50 * 100).toFixed(1)),
            borderColor: '#2563eb',
            backgroundColor: 'rgba(37, 99, 235, 0.08)',
            fill: true,
            tension: 0.3,
            pointRadius: metrics.training_epochs.map(e => (e.epoch === 100 || e.epoch === 1) ? 5 : (metrics.training_epochs.length > 50 ? 0 : 2)),
            pointBackgroundColor: metrics.training_epochs.map(e => e.epoch === 100 ? '#10b981' : '#2563eb'),
            borderWidth: 2.5
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top' },
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => {
                if (ctx.label.includes('100')) return '⭐ Final 100-Epoch Checkpoint (saved as best.pt)';
                return '';
              }
            }
          }
        },
        scales: {
          y: {
            min: 90,
            max: 100,
            ticks: { callback: v => `${v}%` },
            grid: { color: '#f1f5f9' }
          },
          x: { grid: { display: false } }
        }
      }
    });
  }
}

// -------------------------------------------------------------
// Image Zoom Modal for Plots
// -------------------------------------------------------------
function openImageModal(imgSrc, title) {
  const modal = document.getElementById('imageModal');
  const img = document.getElementById('imageModalImg');
  const modalTitle = document.getElementById('imageModalTitle');
  if (modal && img) {
    img.src = imgSrc;
    if (modalTitle) modalTitle.textContent = title || 'Colab Training Artifact';
    modal.classList.add('active');
  }
}

function closeImageModal(e) {
  if (e && e.target && e.target.closest && e.target.closest('.modal-box') && !e.target.classList.contains('btn')) {
    return;
  }
  const modal = document.getElementById('imageModal');
  if (modal) modal.classList.remove('active');
}

// -------------------------------------------------------------
// Report & Modal Functions
// -------------------------------------------------------------
function openReportModal() {
  const modal = document.getElementById('reportModal');
  const reportBody = document.getElementById('reportModalContent');
  if (!modal || !reportBody || !AppState.currentResult) {
    alert('Please run an inspection first.');
    return;
  }

  const res = AppState.currentResult;
  const dateStr = new Date().toLocaleString();

  reportBody.innerHTML = `
    <div style="border-bottom: 2px solid #e2e8f0; padding-bottom: 1rem; margin-bottom: 1rem;">
      <h2 style="font-size: 1.25rem; font-weight: 800; color: #0f172a;">PCB QUALITY ASSURANCE INSPECTION REPORT</h2>
      <p style="font-size: 0.8rem; color: #64748b;">Automated Optical Inspection (AOI) via Fine-Tuned YOLOv8s Model</p>
    </div>

    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1.25rem; font-size: 0.85rem;">
      <div><strong>Report Timestamp:</strong> ${dateStr}</div>
      <div><strong>Inspection Target:</strong> <code>${res.filename}</code></div>
      <div><strong>Model Engine:</strong> ${res.model_engine} (${res.device})</div>
      <div><strong>Inference Latency:</strong> ${res.inference_latency_ms} ms</div>
      <div><strong>Inspection Verdict:</strong> <span class="badge-severity ${res.verdict_badge === 'success' ? 'LOW' : (res.verdict_badge === 'warning' ? 'MEDIUM' : 'HIGH')}">${res.verdict}</span></div>
      <div><strong>Total Defects Identified:</strong> ${res.total_defects} anomalies</div>
    </div>

    <div style="margin-bottom: 1.25rem;">
      <h4 style="font-size: 0.9rem; font-weight: 700; margin-bottom: 0.5rem;">Defect Classification Ledger</h4>
      <table class="data-table" style="font-size: 0.78rem;">
        <thead>
          <tr>
            <th>#</th>
            <th>Type</th>
            <th>Confidence</th>
            <th>Coordinates [x1, y1, x2, y2]</th>
            <th>Severity</th>
          </tr>
        </thead>
        <tbody>
          ${res.detections.length === 0 ? '<tr><td colspan="5" style="text-align:center;">PASS: Zero defect anomalies detected.</td></tr>' :
            res.detections.map(d => `
              <tr>
                <td>#${d.id}</td>
                <td><strong style="color:${d.color}">${d.display_name || d.class}</strong></td>
                <td>${d.confidence_pct}%</td>
                <td><code>[${d.bbox.join(', ')}]</code></td>
                <td><span class="badge-severity ${d.severity}">${d.severity}</span></td>
              </tr>
            `).join('')
          }
        </tbody>
      </table>
    </div>

    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.85rem; font-size: 0.8rem; color: #475569;">
      <strong>Academic Note:</strong> Generated by the Automated PCB Surface Defect Inspection System. Developed for Machine Learning Minor Degree Capstone. Benchmark: DeepPCB with 98.40% mAP@50, 96.82% Precision, and 96.47% Recall (100 Epochs).
    </div>
  `;

  modal.classList.add('active');
}

function closeReportModal() {
  const modal = document.getElementById('reportModal');
  if (modal) modal.classList.remove('active');
}

function printReport() {
  window.print();
}

function downloadAnnotatedImage() {
  if (!AppState.currentResult) return;
  const link = document.createElement('a');
  link.download = `annotated_${AppState.currentResult.filename}`;
  link.href = AppState.currentResult.annotated_image_base64;
  link.click();
}
