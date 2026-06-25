// ============================================================
// Skin Disease Classification Web App
// ============================================================

let classificationCtx = null;  // Store classification context for chat
let activeAbortController = null;  // Cancel previous SSE streams
let lastUploadedFile = null;  // Keep file reference for Grad-CAM re-fetch

// Color map
const RISK_COLORS = { HIGH: '#ef5350', MEDIUM: '#ff9800', LOW: '#66bb6a' };
const BAR_COLORS = ['#4fc3f7', '#81c784', '#ce93d8', '#ffb74d', '#e57373'];

function abortActiveStreams() {
  if (activeAbortController) {
    activeAbortController.abort();
    activeAbortController = null;
  }
}

// ============================================================
// Image Upload
// ============================================================

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('previewImage');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const btnUpload = document.getElementById('btnUpload');
const btnClear = document.getElementById('btnClear');

btnUpload.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('click', () => fileInput.click());

// Clear button
btnClear.addEventListener('click', (e) => {
  e.stopPropagation();
  resetAll();
});

uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadArea.classList.add('drag-over');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('drag-over');
  const files = e.dataTransfer.files;
  if (files.length) handleFile(files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

btnClear.addEventListener('click', resetAll);

function handleFile(file) {
  if (!file.type.startsWith('image/')) return;

  lastUploadedFile = file;

  // Cancel any active SSE streams from previous classification
  abortActiveStreams();

  // Reset Grad-CAM
  document.getElementById('camResult').style.display = 'none';
  document.getElementById('chkGradCAM').checked = false;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewImage.classList.remove('preview-hidden');
    uploadPlaceholder.style.display = 'none';
    btnClear.style.display = 'inline-block';
    document.getElementById('toggleAttn').style.display = '';
    classifyImage(file);
  };
  reader.readAsDataURL(file);
  fileInput.value = '';
}

function resetAll() {
  previewImage.classList.add('preview-hidden');
  previewImage.src = '';
  uploadPlaceholder.style.display = '';
  btnClear.style.display = 'none';
  fileInput.value = '';
  classificationCtx = null;
  document.getElementById('resultContent').style.display = 'none';
  document.getElementById('resultPlaceholder').style.display = '';
  document.getElementById('aiReport').style.display = 'none';
  document.getElementById('aiReportContent').textContent = '';
  document.getElementById('aiLoading').style.display = 'none';
  document.getElementById('chatMessages').innerHTML = '<div class="chat-placeholder">分类完成后可在此咨询AI</div>';
  document.getElementById('chatInput').disabled = true;
  document.getElementById('btnSend').disabled = true;
  document.getElementById('camResult').style.display = 'none';
  document.getElementById('toggleAttn').style.display = 'none';
  document.getElementById('chkGradCAM').checked = false;
  lastUploadedFile = null;
}


// ============================================================
// Classification
// ============================================================

async function classifyImage(file) {
  // Show loading
  document.getElementById('resultPlaceholder').style.display = 'none';
  document.getElementById('resultContent').style.display = 'none';
  document.getElementById('aiReport').style.display = 'none';

  const formData = new FormData();
  formData.append('image', file);

  try {
    const resp = await fetch('/api/classify', { method: 'POST', body: formData });
    const data = await resp.json();

    if (data.error) {
      alert(data.error);
      return;
    }

    renderClassification(data.predictions);
    classificationCtx = data.predictions[0];

    // Enable chat
    document.getElementById('chatInput').disabled = false;
    document.getElementById('btnSend').disabled = false;

    // Trigger AI report
    streamAIReport(data.predictions[0]);

  } catch (err) {
    alert('分类失败: ' + err.message);
  }
}

function renderClassification(predictions) {
  const top = predictions[0];
  const risk = top.risk;

  // Risk banner
  const banner = document.getElementById('riskBanner');
  const riskLabels = { HIGH: '🚨 高风险警告', MEDIUM: '⚠️ 中等风险', LOW: '✅ 低风险' };
  const riskMsgs = {
    HIGH: '建议立即就医，由皮肤科医生进行专业诊断。',
    MEDIUM: '建议安排临床随访，进一步评估。',
    LOW: '倾向良性表现，建议常规观察。'
  };
  banner.className = 'risk-banner ' + risk.toLowerCase();
  banner.innerHTML = `<div>${riskLabels[risk]}: ${top.class_zh}（${top.class_en}）</div>
    <div style="font-size:11px;font-weight:400;margin-top:3px;color:#bbb;">${riskMsgs[risk]}</div>`;

  // Prediction bars
  const barsDiv = document.getElementById('predictionBars');
  barsDiv.innerHTML = predictions.map((p, i) => `
    <div class="prediction-bar">
      <span class="name">${i === 0 ? '→ ' : ''}${p.class_zh}<br><small style="color:#888">${p.class_en}</small></span>
      <div class="track"><div class="fill" style="width:${p.confidence*100}%;background:${BAR_COLORS[i] || '#888'}"></div></div>
      <span class="conf">${(p.confidence*100).toFixed(1)}%</span>
    </div>
  `).join('');

  // Disease KB
  const kb = top.kb || {};
  document.getElementById('diseaseInfo').innerHTML = `
    <h4>📋 疾病知识</h4>
    <p><b>概述:</b> ${kb.overview || '—'}</p>
    <p><b>症状:</b> ${kb.symptoms || '—'}</p>
    <p><b>治疗:</b> ${kb.treatment || '—'}</p>
    <p><b>注意:</b> ${kb.precautions || '—'}</p>
  `;

  document.getElementById('resultContent').style.display = '';
}


// ============================================================
// AI Report (SSE streaming)
// ============================================================

async function streamAIReport(topPrediction) {
  const reportDiv = document.getElementById('aiReport');
  const contentDiv = document.getElementById('aiReportContent');
  const loadingDiv = document.getElementById('aiLoading');

  reportDiv.style.display = '';
  contentDiv.textContent = '';
  loadingDiv.style.display = '';

  // Use AbortController to cancel previous stream
  activeAbortController = new AbortController();

  try {
    const resp = await fetch('/api/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ top: topPrediction }),
      signal: activeAbortController.signal,
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            loadingDiv.style.display = 'none';
            return;
          }
          try {
            const parsed = JSON.parse(data);
            if (parsed.token) {
              contentDiv.textContent += parsed.token;
            }
          } catch (e) { /* skip */ }
        }
      }
    }
    activeAbortController = null;
  } catch (err) {
    if (err.name === 'AbortError') return;  // Intentionally cancelled
    contentDiv.textContent = '[AI建议生成失败: ' + err.message + ']';
  }
  loadingDiv.style.display = 'none';
}


// ============================================================
// Chat
// ============================================================

const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const btnSend = document.getElementById('btnSend');

let chatHistory = [];  // [{role, content}, ...]

btnSend.addEventListener('click', sendChat);
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendChat();
});

async function sendChat() {
  const message = chatInput.value.trim();
  if (!message || !classificationCtx) return;

  chatInput.value = '';
  chatInput.disabled = true;
  btnSend.disabled = true;

  // Cancel previous chat stream
  abortActiveStreams();

  // Add user message
  addChatMsg('user', message);
  chatHistory.push({ role: 'user', content: message });

  // Add assistant placeholder
  const assistantDiv = addChatMsg('assistant', '');
  chatHistory.push({ role: 'assistant', content: '' });

  activeAbortController = new AbortController();

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: message,
        history: chatHistory.slice(0, -1),
        context: classificationCtx,
      }),
      signal: activeAbortController.signal,
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') break;
          try {
            const parsed = JSON.parse(data);
            if (parsed.token) {
              chatHistory[chatHistory.length - 1].content += parsed.token;
              assistantDiv.textContent += parsed.token;
            }
          } catch (e) { /* skip */ }
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') return;
    assistantDiv.textContent = '[回复失败: ' + err.message + ']';
    chatHistory[chatHistory.length - 1].content = assistantDiv.textContent;
  }

  activeAbortController = null;
  chatInput.disabled = false;
  btnSend.disabled = false;
  chatInput.focus();

  // Remove placeholder if present
  const placeholder = chatMessages.querySelector('.chat-placeholder');
  if (placeholder) placeholder.remove();
}

function addChatMsg(role, content) {
  const placeholder = chatMessages.querySelector('.chat-placeholder');
  if (placeholder) placeholder.remove();

  const div = document.createElement('div');
  div.className = 'chat-msg ' + role;
  div.textContent = content;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return div;
}

// ============================================================
// Grad-CAM Toggle
// ============================================================

const chkGradCAM = document.getElementById('chkGradCAM');
const camResult = document.getElementById('camResult');
const camImage = document.getElementById('camImage');
const camLabel = document.getElementById('camLabel');

chkGradCAM.addEventListener('change', async () => {
  if (chkGradCAM.checked && lastUploadedFile) {
    camResult.style.display = '';
    camLabel.textContent = '正在生成注意力热力图...';
    await fetchGradCAM(lastUploadedFile);
  } else {
    camResult.style.display = 'none';
  }
});

async function fetchGradCAM(file) {
  const formData = new FormData();
  formData.append('image', file);
  try {
    const resp = await fetch('/api/gradcam_compare', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.heatmap) {
      camImage.src = data.heatmap;
      camLabel.innerHTML = `🔥 注意力对比 &nbsp;|&nbsp; <span style="color:#888">基线: ${data.baseline_class}</span> &nbsp;vs&nbsp; <span style="color:#ff9800">改进: ${data.improved_class}</span>`;
    } else {
      camLabel.textContent = '生成失败: ' + (data.error || '未知错误');
    }
  } catch (err) {
    camLabel.textContent = '生成失败: ' + err.message;
  }
}
