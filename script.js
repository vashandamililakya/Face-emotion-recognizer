/* ============================================
   FACE EMOTION DETECTION SYSTEM — script.js
   ============================================ */

// ── State ──────────────────────────────────
let stream        = null;
let realtimeTimer = null;
let uploadFile    = null;
let history       = [];

// Emotion metadata
const EMOTIONS = {
  happy:    { emoji: '😊', label: 'Happy',    color: '#f9d84a' },
  sad:      { emoji: '😢', label: 'Sad',      color: '#5bacd8' },
  angry:    { emoji: '😠', label: 'Angry',    color: '#e05252' },
  surprise: { emoji: '😲', label: 'Surprise', color: '#f0a030' },
  fear:     { emoji: '😨', label: 'Fear',     color: '#9a7de0' },
  disgust:  { emoji: '🤢', label: 'Disgust',  color: '#6ab87a' },
  neutral:  { emoji: '😐', label: 'Neutral',  color: '#8fa8c0' },
};

// ── Camera ────────────────────────────────

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });

    const video   = document.getElementById('videoEl');
    video.srcObject = stream;
    video.classList.add('active');

    document.getElementById('videoOverlay').classList.add('hidden');
    document.getElementById('scanLine').classList.add('active');

    // Update UI
    setBtn('startCameraBtn', true);
    setBtn('captureBtn', false);
    setBtn('stopCameraBtn', false);
    setCameraStatus(true);
    showToast('Camera started successfully!', 'success');
  } catch (err) {
    console.error(err);
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      showToast('Camera access denied. Please allow camera permissions in your browser.', 'error');
    } else if (err.name === 'NotFoundError') {
      showToast('No camera found on this device.', 'error');
    } else {
      showToast('Could not start camera: ' + err.message, 'error');
    }
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  stopRealtime();

  const video = document.getElementById('videoEl');
  video.srcObject = null;
  video.classList.remove('active');

  document.getElementById('videoOverlay').classList.remove('hidden');
  document.getElementById('scanLine').classList.remove('active');
  document.getElementById('faceBox').style.display = 'none';

  setBtn('startCameraBtn', false);
  setBtn('captureBtn', true);
  setBtn('stopCameraBtn', true);
  setCameraStatus(false);
  showToast('Camera stopped.', 'warning');
}

function captureImage() {
  if (!stream) { showToast('Start the camera first.', 'warning'); return; }

  const video  = document.getElementById('videoEl');
  const canvas = document.getElementById('snapshotCanvas');
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(blob => {
    if (!blob) { showToast('Failed to capture image.', 'error'); return; }
    sendToAPI(blob, 'webcam');
  }, 'image/jpeg', 0.92);
}

// ── Realtime Detection ────────────────────

function toggleRealtime(checkbox) {
  if (checkbox.checked) {
    if (!stream) {
      showToast('Start the camera before enabling auto-detect.', 'warning');
      checkbox.checked = false;
      return;
    }
    startRealtime();
    showToast('Auto-detection enabled (every 3s)', 'success');
  } else {
    stopRealtime();
    showToast('Auto-detection disabled.', 'warning');
  }
}

function startRealtime() {
  stopRealtime();
  realtimeTimer = setInterval(() => {
    if (stream) captureImage();
    else stopRealtime();
  }, 3000);
}

function stopRealtime() {
  if (realtimeTimer) {
    clearInterval(realtimeTimer);
    realtimeTimer = null;
  }
  const toggle = document.getElementById('realtimeToggle');
  if (toggle) toggle.checked = false;
}

// ── File Upload ───────────────────────────

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
  if (!allowed.includes(file.type)) {
    showToast('Please upload a JPG, PNG, or WEBP image.', 'error');
    return;
  }

  uploadFile = file;

  const reader = new FileReader();
  reader.onload = e => {
    const preview   = document.getElementById('uploadPreview');
    const content   = document.getElementById('dropzoneContent');
    preview.src      = e.target.result;
    preview.style.display = 'block';
    content.style.display = 'none';
    setBtn('analyseUploadBtn', false);
  };
  reader.readAsDataURL(file);
}

// Drag & drop
const dropzone = document.getElementById('dropzone');
['dragenter', 'dragover'].forEach(evt => {
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.style.borderColor = '#4fa0d8';
    dropzone.style.background  = 'rgba(79,160,216,0.08)';
  });
});
['dragleave', 'drop'].forEach(evt => {
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.style.borderColor = '';
    dropzone.style.background  = '';
  });
});
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) {
    const fakeEvt = { target: { files: [file] } };
    handleFileUpload(fakeEvt);
  }
});

function analyseUpload() {
  if (!uploadFile) { showToast('Please select an image first.', 'warning'); return; }
  sendToAPI(uploadFile, 'upload');
}

// ── API Call ──────────────────────────────

async function sendToAPI(imageBlob, source) {
  showAnalysing();

  const formData = new FormData();
  formData.append('image', imageBlob, source === 'webcam' ? 'capture.jpg' : 'upload.jpg');

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.message || `Server error: ${response.status}`);
    }

    const data = await response.json();

    if (!data.emotion) throw new Error('No emotion detected in the response.');
    showResult(data);

  } catch (err) {
    console.error('API error:', err);

    // ── Demo fallback when no backend is connected ──
    if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError') || err.message.includes('404') || err.message.includes('Server error')) {
      const demoData = generateDemoResult();
      showResult(demoData);
      showToast('Demo mode: backend not connected. Showing simulated result.', 'warning');
    } else if (err.message.toLowerCase().includes('no face') || err.message.toLowerCase().includes('no emotion')) {
      hideAnalysing();
      showToast('No face detected in this image. Try another photo.', 'error');
    } else {
      hideAnalysing();
      showToast('Analysis failed: ' + err.message, 'error');
    }
  }
}

// ── Demo Result Generator ─────────────────

function generateDemoResult() {
  const keys   = Object.keys(EMOTIONS);
  const picked = keys[Math.floor(Math.random() * keys.length)];
  const conf   = (Math.random() * 0.38 + 0.60).toFixed(4); // 60–98%

  // Generate other scores
  const breakdown = {};
  let remaining = 1 - parseFloat(conf);
  keys.forEach((k, i) => {
    if (k === picked) { breakdown[k] = parseFloat(conf); return; }
    if (i === keys.length - 1) { breakdown[k] = parseFloat(remaining.toFixed(4)); return; }
    const slice = parseFloat((Math.random() * remaining * 0.6).toFixed(4));
    breakdown[k] = slice;
    remaining -= slice;
  });

  return { emotion: picked, confidence: parseFloat(conf), breakdown };
}

// ── Result Display ────────────────────────

function showResult(data) {
  hideAnalysing();

  // Backend returns "all_emotions" — support both key names
  const breakdown = data.all_emotions || data.breakdown || null;
  const { emotion, confidence } = data;
  const meta = EMOTIONS[emotion.toLowerCase()] || EMOTIONS['neutral'];
  const pct  = Math.round(confidence * 100);

  // Main result
  document.getElementById('emotionEmoji').textContent   = meta.emoji;
  document.getElementById('emotionLabel').textContent   = meta.label;
  document.getElementById('confidenceValue').textContent = pct + '%';

  const bar = document.getElementById('confidenceBarFill');
  bar.style.width = '0%';
  setTimeout(() => { bar.style.width = pct + '%'; }, 60);

  // Breakdown bars
  if (breakdown) {
    const bdEl = document.getElementById('emotionBreakdown');
    const sorted = Object.entries(breakdown).sort((a, b) => b[1] - a[1]);

    bdEl.innerHTML = sorted.map(([key, val]) => {
      const m = EMOTIONS[key] || EMOTIONS['neutral'];
      const p = Math.round(val * 100);
      return `
        <div class="breakdown-row">
          <span class="breakdown-emoji">${m.emoji}</span>
          <span class="breakdown-name">${m.label}</span>
          <div class="breakdown-bar-track">
            <div class="breakdown-bar-fill" style="width:${p}%"></div>
          </div>
          <span class="breakdown-pct">${p}%</span>
        </div>`;
    }).join('');
  }

  // Face box (cosmetic)
  showFaceBox();

  // Show panels
  document.getElementById('resultEmpty').style.display   = 'none';
  document.getElementById('resultContent').style.display = 'flex';

  // Add to history
  addHistory(meta, pct);

  showToast(`${meta.emoji} ${meta.label} detected (${pct}% confidence)`, 'success');
}

function showAnalysing() {
  document.getElementById('resultEmpty').style.display    = 'none';
  document.getElementById('resultContent').style.display  = 'none';
  document.getElementById('analysingState').style.display = 'flex';
}

function hideAnalysing() {
  document.getElementById('analysingState').style.display = 'none';
}

// ── Face Box Overlay ──────────────────────

function showFaceBox() {
  const frame = document.getElementById('videoFrame');
  const box   = document.getElementById('faceBox');

  const fw = frame.clientWidth;
  const fh = frame.clientHeight;

  // Randomise position slightly for demo realism
  const margin  = 0.15;
  const bw = fw * 0.32;
  const bh = fh * 0.55;
  const bx = fw * margin + (Math.random() * fw * (0.5 - margin));
  const by = fh * 0.1 + (Math.random() * fh * 0.15);

  box.style.display = 'block';
  box.style.left    = bx + 'px';
  box.style.top     = by + 'px';
  box.style.width   = bw + 'px';
  box.style.height  = bh + 'px';

  setTimeout(() => { box.style.display = 'none'; }, 4000);
}

// ── History ───────────────────────────────

function addHistory(meta, pct) {
  const now = new Date();
  const time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  history.unshift({ emoji: meta.emoji, label: meta.label, pct, time });
  if (history.length > 5) history.pop();
  renderHistory();
}

function renderHistory() {
  const list = document.getElementById('historyList');
  if (history.length === 0) {
    list.innerHTML = '<li class="history-empty">No detections yet</li>';
    return;
  }
  list.innerHTML = history.map(h => `
    <li class="history-item">
      <span class="history-emoji">${h.emoji}</span>
      <div class="history-info">
        <div class="history-emotion">${h.label}</div>
        <div class="history-confidence">${h.pct}% confidence</div>
      </div>
      <span class="history-time">${h.time}</span>
    </li>`).join('');
}

function clearHistory() {
  history = [];
  renderHistory();
  showToast('History cleared.', 'warning');
}

// ── Helpers ───────────────────────────────

function setBtn(id, disabled) {
  const el = document.getElementById(id);
  if (el) el.disabled = disabled;
}

function setCameraStatus(online) {
  const badge = document.getElementById('cameraStatus');
  badge.textContent = online ? 'Live' : 'Offline';
  badge.classList.toggle('online', online);
}

// ── Toast Notifications ───────────────────

function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');

  const icons = {
    success: '✓',
    error:   '✕',
    warning: '⚠',
    info:    'ℹ',
  };

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || 'ℹ'}</span><span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('toast-fade-out');
    toast.addEventListener('animationend', () => toast.remove());
  }, 4000);
}

// ── Init ──────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  renderHistory();
  showToast('Welcome! Start your camera or upload an image.', 'info');
});
