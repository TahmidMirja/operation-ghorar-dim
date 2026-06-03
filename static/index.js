/**
 * Aetheris Vision AI – Frontend Engine v3
 *
 * Architecture:
 *  - One analysis request at a time (no duplicates / stampede)
 *  - AbortController cancels in-flight request on stop
 *  - requestAnimationFrame drives bounding-box overlay renders
 *  - Pre-allocated, reused canvas (no GC churn)
 *  - Bounding box interpolation (lerp) for smooth tracking
 *  - Rate-limit auto-recovery with 60-second back-off
 *  - Full ARIA + accessibility wiring
 */

'use strict';

// ── DOM References ────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// Camera
const webcam        = $('webcam');
const overlayCanvas = $('overlay-canvas');
const captureCanvas = $('capture-canvas');
const camOffline    = $('cam-offline');
const viewport      = $('viewport');
const scanLine      = $('scan-line');
const recBadge      = $('rec-badge');
const cameraSelect  = $('camera-select');
const intervalSlider= $('interval-slider');
const intervalLabel = $('interval-label');
const startBtn      = $('start-btn');
const stopBtn       = $('stop-btn');

// Status
const systemPill    = $('system-pill');
const pillDot       = $('pill-dot');
const pillText      = $('pill-text');
const latencyBadge  = $('latency-badge');
const latencyMs     = $('latency-ms');

// Widgets
const statusCard    = $('status-card');
const statusIcon    = $('status-icon');
const statusTitle   = $('status-title');
const statusSub     = $('status-sub');

const emotionEmoji  = $('emotion-emoji');
const emotionVal    = $('emotion-val');
const gestureEmoji  = $('gesture-emoji');
const gestureVal    = $('gesture-val');

const restlessBar   = $('restless-bar');
const restlessVal   = $('restless-val');
const confBadge     = $('conf-badge');
const confBar       = $('conf-bar');

const insightExp    = $('insight-explanation');
const insightRec    = $('insight-recommendation');

const logList       = $('log-list');
const logEmpty      = $('log-empty');
const clearLogBtn   = $('clear-log-btn');

// Settings modal
const settingsBtn   = $('settings-btn');
const modalBackdrop = $('modal-backdrop');
const modalClose    = $('modal-close');
const modalCancel   = $('modal-cancel');
const modalSave     = $('modal-save');
const inputApiKey   = $('input-api-key');
const selectModel   = $('select-model');

// Manual / reference guide
const manualToggle  = $('manual-toggle');
const manualBody    = $('manual-body');

// Camera HUD overlay
const camHud           = $('cam-hud');
const hudEmotionEmoji  = $('hud-emotion-emoji');
const hudEmotionLabel  = $('hud-emotion-label');
const hudGestureEmoji  = $('hud-gesture-emoji');
const hudGestureLabel  = $('hud-gesture-label');
const hudStatusDot     = $('hud-status-dot');
const hudStatusLabel   = $('hud-status-label');
const hudSpinner       = $('hud-spinner');

// Canvas contexts (allocated once)
const captureCtx    = captureCanvas.getContext('2d', { willReadFrequently: false });
const overlayCtx    = overlayCanvas.getContext('2d');

// ── State ─────────────────────────────────────────────────────────────────────
let localStream         = null;
let analysisTimer       = null;
let isMonitoring        = false;
let requestInFlight     = false;    // Prevents duplicate concurrent requests
let abortCtrl           = null;     // AbortController for in-flight fetch
let cooldownTimer       = null;     // Rate-limit back-off timer
let isCoolingDown       = false;    // True while waiting for rate-limit recovery
let previousNeed        = null;
let intervalSeconds     = 2;

// Server config (fetched on load)
let serverHasKey        = false;
let serverModel         = null;

// Bounding box lerp targets (for smooth overlay animation)
let faceLerp  = { y1:0, x1:0, y2:0, x2:0, active:false };
let handLerp  = { y1:0, x1:0, y2:0, x2:0, active:false };
let rafHandle = null;

// ── Maps ──────────────────────────────────────────────────────────────────────
const EMOTION_MAP = {
    'Happy':            { e:'😊', l:'Happy / Smiling' },
    'Sad':              { e:'😢', l:'Sad / Sorrow' },
    'Crying':           { e:'😭', l:'Crying / Distress' },
    'Pain':             { e:'😫', l:'Pain / Grimacing' },
    'Angry':            { e:'😠', l:'Angry / Agitated' },
    'Scared':           { e:'😨', l:'Scared / Fearful' },
    'Surprised':        { e:'😲', l:'Surprised / Shocked' },
    'Neutral':          { e:'😐', l:'Neutral / Calm' },
    'Tired / Sleepy':   { e:'🥱', l:'Tired / Sleepy' },
    'Disgusted':        { e:'🤢', l:'Disgusted' },
    'Anxious / Nervous':{ e:'😰', l:'Anxious / Nervous' },
    'Confused':         { e:'😕', l:'Confused / Puzzled' },
};

const GESTURE_MAP = {
    'Holding Stomach':          { e:'🤢', l:'Holding Stomach' },
    'Raising Index Finger':     { e:'☝️', l:'Raising Index Finger' },
    'Holding Head':             { e:'💆', l:'Holding Head' },
    'Waving Hand (Hello)':      { e:'👋', l:'Waving Hello' },
    'Waving Goodbye':           { e:'🚶', l:'Waving Goodbye' },
    'Holding Throat / Coughing':{ e:'😷', l:'Holding Throat' },
    'Rubbing Eyes':             { e:'🥱', l:'Rubbing Eyes' },
    'Thumbs Up':                { e:'👍', l:'Thumbs Up' },
    'Thumbs Down':              { e:'👎', l:'Thumbs Down' },
    'Shushing Finger on Lips':  { e:'🤫', l:'Shushing' },
    'Covering Ears':            { e:'🙉', l:'Covering Ears' },
    'None':                     { e:'🙌', l:'No Active Gesture' },
};

const STATUS_MAP = {
    'Normal':              { icon:'💚', title:'Normal Status',             sub:'No active distress detected. Individual appears calm.',                              cls:'status-normal',  log:'log-normal',  vig:'',             snd:'info' },
    'Hungry or Pain':      { icon:'🚨', title:'Stomach Pain / Hunger',     sub:'Possible stomach pain or hunger. Individual is holding their abdomen.',             cls:'status-danger',  log:'log-danger',  vig:'vignette-danger',snd:'critical' },
    'Needs Bathroom':      { icon:'🚻', title:'Bathroom Urgency',          sub:'Individual raised an index finger – a universal signal for bathroom need.',         cls:'status-bath',    log:'log-bath',    vig:'vignette-bath', snd:'warning' },
    'Headache or Tired':   { icon:'⚠️', title:'Headache / Fatigue',        sub:'Individual is holding their head, forehead, or shows eyes-closed fatigue.',        cls:'status-warning', log:'log-warning', vig:'vignette-warn', snd:'warning' },
    'Crying':              { icon:'😭', title:'Active Crying Alert',       sub:'Individual is actively weeping or showing deep emotional distress.',                 cls:'status-danger',  log:'log-danger',  vig:'vignette-danger',snd:'critical' },
    'Greeting':            { icon:'👋', title:'Greeting Gesture',          sub:'Individual is waving their hand to say Hello.',                                     cls:'status-normal',  log:'log-normal',  vig:'',             snd:'info' },
    'Leaving':             { icon:'🚪', title:'Goodbye / Leaving',         sub:'Individual is waving goodbye or pointing away to signal departure.',                cls:'status-warning', log:'log-warning', vig:'vignette-warn', snd:'warning' },
    'Distressed':          { icon:'😰', title:'Emotional Distress Alert',  sub:'Individual is showing expressions of extreme anger, panic, or acute fear.',         cls:'status-danger',  log:'log-danger',  vig:'vignette-danger',snd:'critical' },
    'Choking or Throat Pain':{ icon:'🚨', title:'Choking / Throat Distress',sub:'Individual is holding throat or coughing severely. Immediate attention required.', cls:'status-danger',  log:'log-danger',  vig:'vignette-danger',snd:'critical' },
    'Approval':            { icon:'👍', title:'Approval Sign',             sub:'Individual gave a Thumbs Up gesture.',                                             cls:'status-normal',  log:'log-normal',  vig:'',             snd:'info' },
    'Disapproval':         { icon:'👎', title:'Disapproval Sign',          sub:'Individual gave a Thumbs Down gesture.',                                           cls:'status-warning', log:'log-warning', vig:'vignette-warn', snd:'warning' },
    'Request Quiet':       { icon:'🤫', title:'Request Silence',           sub:'Individual is gesturing for quiet or silence.',                                     cls:'status-normal',  log:'log-normal',  vig:'',             snd:'info' },
    'Sensory Overload':    { icon:'🙉', title:'Sensory Overload',          sub:'Individual is covering ears. Possible loud noises or sensory disturbance.',         cls:'status-warning', log:'log-warning', vig:'vignette-warn', snd:'warning' },
    'Tired or Sleepy':     { icon:'🥱', title:'Tired / Sleepy Alert',      sub:'Individual is yawning, rubbing eyes, or displaying signs of deep fatigue.',        cls:'status-warning', log:'log-warning', vig:'vignette-warn', snd:'warning' },
};

// ── Audio Synthesiser ─────────────────────────────────────────────────────────
function playSound(type) {
    try {
        const AC = window.AudioContext || window.webkitAudioContext;
        if (!AC) return;
        const ctx = new AC();

        const gain = ctx.createGain();
        gain.connect(ctx.destination);

        if (type === 'critical') {
            const o1 = ctx.createOscillator();
            const o2 = ctx.createOscillator();
            o1.type = 'sine'; o1.frequency.setValueAtTime(880, ctx.currentTime);
            o1.frequency.exponentialRampToValueAtTime(1100, ctx.currentTime + .15);
            o2.type = 'sine'; o2.frequency.setValueAtTime(440, ctx.currentTime);
            o2.frequency.exponentialRampToValueAtTime(550, ctx.currentTime + .15);
            gain.gain.setValueAtTime(.11, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(.001, ctx.currentTime + .38);
            o1.connect(gain); o2.connect(gain);
            o1.start(); o2.start();
            o1.stop(ctx.currentTime + .38); o2.stop(ctx.currentTime + .38);
        } else if (type === 'warning') {
            const o = ctx.createOscillator();
            o.type = 'triangle';
            o.frequency.setValueAtTime(523, ctx.currentTime);
            o.frequency.setValueAtTime(659, ctx.currentTime + .12);
            gain.gain.setValueAtTime(.07, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(.001, ctx.currentTime + .28);
            o.connect(gain); o.start(); o.stop(ctx.currentTime + .28);
        } else {
            const o = ctx.createOscillator();
            o.type = 'sine'; o.frequency.setValueAtTime(620, ctx.currentTime);
            gain.gain.setValueAtTime(.025, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(.001, ctx.currentTime + .08);
            o.connect(gain); o.start(); o.stop(ctx.currentTime + .08);
        }
    } catch (e) { /* ignore audio context errors */ }
}

// ── System Status Helpers ─────────────────────────────────────────────────────
function setStatus(state, msg) {
    pillDot.className = 'pill-dot';
    pillText.textContent = msg;
    if (state === 'active')  pillDot.classList.add('state-active');
    else if (state === 'error')   pillDot.classList.add('state-error');
    else if (state === 'warning') pillDot.classList.add('state-warning');
}

function showLatency(ms) {
    latencyBadge.classList.remove('hidden');
    latencyMs.textContent = ms;
}

// ── Initialisation ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);

async function init() {
    loadStoredSettings();
    await fetchServerConfig();
    await enumerateCameras();
    bindEvents();
    setStatus('', 'System Ready');
}

function bindEvents() {
    // Monitoring
    startBtn.addEventListener('click', startMonitoring);
    stopBtn.addEventListener('click',  stopMonitoring);

    // Interval slider
    intervalSlider.addEventListener('input', e => {
        intervalSeconds = Number(e.target.value);
        intervalLabel.textContent = `${intervalSeconds}s`;
        if (isMonitoring) restartLoop();
    });

    // Settings modal
    settingsBtn.addEventListener('click',   () => openModal());
    modalClose.addEventListener('click',    () => closeModal());
    modalCancel.addEventListener('click',   () => closeModal());
    modalSave.addEventListener('click',     saveSettings);
    modalBackdrop.addEventListener('click', e => { if (e.target === modalBackdrop) closeModal(); });

    // Clear log
    clearLogBtn.addEventListener('click', clearLog);

    // Reference manual accordion
    manualToggle.addEventListener('click', toggleManual);

    // Manual tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', e => {
            e.stopPropagation();
            switchTab(btn.dataset.pane);
        });
    });

    // Overlay canvas resizing
    webcam.addEventListener('loadedmetadata', syncCanvas);
    webcam.addEventListener('play',           syncCanvas);
    window.addEventListener('resize',         syncCanvas);

    // Keyboard: close modal on Escape
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') closeModal();
    });
}

// ── Settings ──────────────────────────────────────────────────────────────────
function loadStoredSettings() {
    inputApiKey.value = localStorage.getItem('ae_api_key') || '';
    const m = localStorage.getItem('ae_model');
    if (m) selectModel.value = m;
}

async function fetchServerConfig() {
    try {
        const res = await fetch('/api/config');
        if (!res.ok) return;
        const cfg = await res.json();
        serverHasKey = cfg.has_server_key;
        serverModel  = cfg.default_model;
        if (serverHasKey && !localStorage.getItem('ae_api_key')) {
            setStatus('active', 'Server Key Configured');
        }
        if (!localStorage.getItem('ae_model') && serverModel) {
            selectModel.value = serverModel;
        }
    } catch (_) { /* server not reachable yet */ }
}

// ── Camera enumeration ────────────────────────────────────────────────────────
async function enumerateCameras() {
    try {
        // Request permission first so labels are populated
        const tmp = await navigator.mediaDevices.getUserMedia({ video: true });
        tmp.getTracks().forEach(t => t.stop());

        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameras = devices.filter(d => d.kind === 'videoinput');

        cameraSelect.innerHTML = '';
        if (!cameras.length) {
            cameraSelect.innerHTML = '<option>No cameras found</option>';
            return;
        }
        cameras.forEach((d, i) => {
            const opt = document.createElement('option');
            opt.value = d.deviceId;
            opt.textContent = d.label || `Camera ${i + 1}`;
            cameraSelect.appendChild(opt);
        });
    } catch (err) {
        cameraSelect.innerHTML = '<option>Permission denied</option>';
        setStatus('error', 'Camera Permission Denied');
    }
}

// ── Monitoring Control ────────────────────────────────────────────────────────
async function startMonitoring() {
    const key = localStorage.getItem('ae_api_key');
    if (!key && !serverHasKey) {
        openModal();
        return;
    }
    try {
        const deviceId = cameraSelect.value;
        localStream = await navigator.mediaDevices.getUserMedia({
            video: deviceId ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } } : { width: { ideal: 1280 }, height: { ideal: 720 } }
        });
        webcam.srcObject = localStream;

        // Hide offline overlay
        camOffline.classList.add('hidden');

        // Show HUD
        camHud.hidden = false;
        scanLine.classList.add('active');
        recBadge.classList.add('visible');
        startBtn.disabled    = true;
        stopBtn.disabled     = false;
        cameraSelect.disabled= true;
        isMonitoring         = true;
        isCoolingDown        = false;

        setStatus('active', 'Active Monitoring');
        playSound('info');

        setTimeout(syncCanvas, 400);
        startRAF();
        startLoop();

    } catch (err) {
        console.error('Camera error:', err);
        setStatus('error', 'Camera Access Failed');
        alert('Could not open camera. Please check permissions.');
    }
}

function stopMonitoring() {
    isMonitoring = false;

    // Cancel in-flight request
    if (abortCtrl) { abortCtrl.abort(); abortCtrl = null; }

    // Clear timers
    if (analysisTimer) { clearTimeout(analysisTimer); analysisTimer = null; }
    clearTimeout(cooldownTimer);  cooldownTimer = null;
    isCoolingDown    = false;
    requestInFlight  = false;
    previousNeed     = null;

    // Stop RAF overlay loop
    if (rafHandle) { cancelAnimationFrame(rafHandle); rafHandle = null; }

    // Stop webcam
    if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
    webcam.srcObject = null;

    // UI reset
    camHud.hidden = true;
    camOffline.classList.remove('hidden');
    scanLine.classList.remove('active');
    recBadge.classList.remove('visible');
    startBtn.disabled    = false;
    stopBtn.disabled     = true;
    cameraSelect.disabled= false;

    clearOverlay();
    resetDashboard();
    setStatus('', 'Monitoring Stopped');
}

// ── Analysis Loop ─────────────────────────────────────────────────────────────
function startLoop() {
    analyzeFrame();          // Fire immediately
}

function restartLoop() {
    if (analysisTimer) {
        clearTimeout(analysisTimer);
        analysisTimer = setTimeout(analyzeFrame, intervalSeconds * 1000);
    }
}

function scheduleNextAnalysis() {
    if (!isMonitoring) return;
    if (analysisTimer) clearTimeout(analysisTimer);
    analysisTimer = setTimeout(analyzeFrame, intervalSeconds * 1000);
}

async function analyzeFrame() {
    if (!isMonitoring || !localStream)      return;
    if (requestInFlight || isCoolingDown)   return;
    if (!webcam.videoWidth || webcam.readyState < 2) return;

    // ── Capture frame ──────────────────────────────────────
    const MAX = 480;
    let w = webcam.videoWidth, h = webcam.videoHeight;
    if (w > MAX || h > MAX) {
        if (w >= h) { h = Math.round(h * MAX / w); w = MAX; }
        else        { w = Math.round(w * MAX / h); h = MAX; }
    }
    captureCanvas.width  = w;
    captureCanvas.height = h;
    captureCtx.drawImage(webcam, 0, 0, w, h);

    // JPEG 0.82 – best balance of payload vs accuracy
    const b64 = captureCanvas.toDataURL('image/jpeg', 0.82);

    const apiKey = localStorage.getItem('ae_api_key') || '';
    const model  = localStorage.getItem('ae_model') || serverModel || 'google/gemini-2.5-flash';

    // ── Fetch ──────────────────────────────────────────────
    requestInFlight = true;
    abortCtrl       = new AbortController();
    const t0        = performance.now();
    hudSpinner.classList.add('active');    // show spinner while in-flight

    try {
        const res = await fetch('/api/analyze', {
            method:  'POST',
            signal:  abortCtrl.signal,
            headers: { 'Content-Type': 'application/json', 'X-Gemini-API-Key': apiKey },
            body:    JSON.stringify({ image: b64, model }),
        });

        const latency = Math.round(performance.now() - t0);
        showLatency(latency);

        if (res.status === 429) {
            const body = await safeJson(res);
            handleRateLimit(body?.detail || 'Rate limit reached (429).');
            return;
        }

        if (!res.ok) {
            const body = await safeJson(res);
            const msg  = body?.detail || `API error ${res.status}`;
            if (res.status === 401 || res.status === 403) {
                showWidgetError('Authentication Failed', msg);
                setStatus('error', 'Auth Error – Check API Key');
            } else {
                showWidgetError('Analysis Error', msg);
                setStatus('error', `Error ${res.status}`);
            }
            return;
        }

        const data = await res.json();

        // Extra 429 guard (some APIs return 200 + error body)
        if (data?.detail?.includes('429')) {
            handleRateLimit(data.detail);
            return;
        }

        // ✅ Success
        setStatus('active', 'Active Monitoring');
        updateDashboard(data);

    } catch (err) {
        if (err.name === 'AbortError') return; // User clicked Stop – ignore

        const msg = err.message || 'Network error';
        if (msg.includes('429') || msg.includes('quota') || msg.includes('limit')) {
            handleRateLimit(msg);
        } else {
            showWidgetError('Connection Error', msg);
            setStatus('error', 'Connection Failed – Retrying…');
        }
    } finally {
        requestInFlight = false;
        abortCtrl       = null;
        hudSpinner.classList.remove('active'); // always hide spinner
        if (isMonitoring && !isCoolingDown) {
            scheduleNextAnalysis();
        }
    }
}

// Safe JSON parser that doesn't throw on non-JSON bodies
async function safeJson(res) {
    try { return await res.json(); } catch (_) { return null; }
}

// ── Rate Limit Handler ────────────────────────────────────────────────────────
function handleRateLimit(msg) {
    const WAIT = 60;
    isCoolingDown = true;
    if (analysisTimer) { clearTimeout(analysisTimer); analysisTimer = null; }

    showWidgetError('Rate Limited (429)', `${msg} — Auto-resuming in ${WAIT}s…`);
    setStatus('warning', `Rate Limited – resuming in ${WAIT}s`);
    playSound('warning');
    clearOverlay();

    clearTimeout(cooldownTimer);
    cooldownTimer = setTimeout(() => {
        if (!isMonitoring) return;
        isCoolingDown = false;
        setStatus('active', 'Active Monitoring');
        statusCard.className = 'widget status-widget status-normal';
        statusIcon.textContent = '💚';
        statusTitle.textContent = 'Resuming Analysis…';
        statusSub.textContent  = 'Rate limit cleared. Monitoring resumed.';
        analyzeFrame();
    }, WAIT * 1000);
}

// ── Dashboard Reset ───────────────────────────────────────────────────────────
function resetDashboard() {
    statusCard.className    = 'widget status-widget status-idle';
    statusIcon.textContent  = '💤';
    statusTitle.textContent = 'Awaiting Signal…';
    statusSub.textContent   = 'Start monitoring to analyse behavioural cues.';
    viewport.className      = 'viewport';

    emotionEmoji.textContent = '😐';  emotionVal.textContent = 'Neutral';
    gestureEmoji.textContent = '🙌';  gestureVal.textContent = 'None';

    restlessBar.style.height          = '20%';
    restlessBar.style.background      = 'var(--green)';
    restlessBar.style.boxShadow       = '0 0 6px var(--green)';
    restlessVal.textContent           = 'Low';

    confBar.style.width               = '0%';
    confBadge.textContent             = '--';

    insightExp.textContent = 'No active stream. Start monitoring for real-time analysis.';
    insightRec.textContent = 'N/A';

    latencyBadge.classList.add('hidden');
}

function showWidgetError(title, sub) {
    if (!isMonitoring) return;
    statusCard.className    = 'widget status-widget status-danger';
    statusIcon.textContent  = '❌';
    statusTitle.textContent = title;
    statusSub.textContent   = sub;
    viewport.className      = 'viewport';
    clearOverlay();
}

// ── Dashboard Update ──────────────────────────────────────────────────────────
function updateDashboard(data) {
    if (!isMonitoring) return;

    // 1. Status card
    const need    = data.Predicted_Need_or_Status || 'Normal';
    const mapping = STATUS_MAP[need] || STATUS_MAP['Normal'];

    statusCard.className    = `widget status-widget ${mapping.cls}`;
    statusTitle.textContent = mapping.title;
    statusSub.textContent   = mapping.sub;

    // Animate icon only on change
    if (need !== previousNeed) {
        statusIcon.textContent = mapping.icon;
        statusIcon.classList.remove('pulse');
        requestAnimationFrame(() => statusIcon.classList.add('pulse'));
        playSound(mapping.snd);
        appendLog(need, data.Detected_Emotion || 'Neutral', mapping);
        previousNeed = need;
    }

    // 2. Vignette
    viewport.className = 'viewport';
    if (mapping.vig) viewport.classList.add(mapping.vig);

    // 3. Emotion
    const em = EMOTION_MAP[data.Detected_Emotion] || EMOTION_MAP['Neutral'];
    emotionEmoji.textContent = em.e;
    emotionVal.textContent   = em.l;

    // 4. Gesture
    const gm = GESTURE_MAP[data.Physical_Gesture] || GESTURE_MAP['None'];
    gestureEmoji.textContent = gm.e;
    gestureVal.textContent   = gm.l;

    // 5. Restlessness
    const rl = data.Restlessness_Level || 'Low';
    restlessVal.textContent = rl;
    if (rl === 'High') {
        restlessBar.style.height     = '92%';
        restlessBar.style.background = 'var(--red)';
        restlessBar.style.boxShadow  = '0 0 8px var(--red)';
    } else if (rl === 'Medium') {
        restlessBar.style.height     = '55%';
        restlessBar.style.background = 'var(--amber)';
        restlessBar.style.boxShadow  = '0 0 8px var(--amber)';
    } else {
        restlessBar.style.height     = '20%';
        restlessBar.style.background = 'var(--green)';
        restlessBar.style.boxShadow  = '0 0 6px var(--green)';
    }

    // 6. Confidence
    const cf = data.Confidence_Score || 'Low';
    confBadge.textContent = cf;
    if (cf === 'High') {
        confBar.style.width      = '92%';
        confBar.style.background = 'linear-gradient(90deg, var(--teal), var(--green))';
    } else if (cf === 'Medium') {
        confBar.style.width      = '60%';
        confBar.style.background = 'linear-gradient(90deg, var(--teal), var(--amber))';
    } else {
        confBar.style.width      = '28%';
        confBar.style.background = 'linear-gradient(90deg, var(--teal), var(--red))';
    }

    // 7. Insights
    insightExp.textContent = data.Detailed_Explanation      || 'No anomalies detected.';
    insightRec.textContent = data.Actionable_Recommendation || 'Continue standard observation.';

    // 8. Update bounding box lerp targets
    setBBoxTarget(faceLerp, data.Face_Bounding_Box);
    setBBoxTarget(handLerp, data.Hand_Bounding_Box);

    // 9. Update camera HUD
    updateHUD(data, mapping);
}

// ── Camera HUD Update ───────────────────────────────────────────────
function updateHUD(data, mapping) {
    const em = EMOTION_MAP[data.Detected_Emotion] || EMOTION_MAP['Neutral'];
    const gm = GESTURE_MAP[data.Physical_Gesture] || GESTURE_MAP['None'];

    hudEmotionEmoji.textContent = em.e;
    hudEmotionLabel.textContent = em.l;
    hudGestureEmoji.textContent = gm.e;
    hudGestureLabel.textContent = gm.l;
    hudStatusLabel.textContent  = mapping.title;

    // Status dot colour
    hudStatusDot.className = 'hud-dot';
    if (mapping.cls === 'status-danger')  hudStatusDot.classList.add('danger');
    else if (mapping.cls === 'status-warning') hudStatusDot.classList.add('warning');
    else if (mapping.cls === 'status-bath')    hudStatusDot.classList.add('bath');
    else                                       hudStatusDot.classList.add('normal');
}

// ── Bounding Box Overlay (RAF loop) ──────────────────────────────────────────
function setBBoxTarget(lerp, box) {
    if (!box || box.length < 4 || box.every(v => v === 0)) {
        lerp.active = false;
    } else {
        const wasActive = lerp.active;
        lerp.active = true;
        lerp.ty1 = box[0]; lerp.tx1 = box[1]; lerp.ty2 = box[2]; lerp.tx2 = box[3];
        // Snap to target on first appearance (no slide-from-zero glitch)
        if (!wasActive) {
            lerp.y1 = lerp.ty1; lerp.x1 = lerp.tx1;
            lerp.y2 = lerp.ty2; lerp.x2 = lerp.tx2;
        }
    }
}

function lerpVal(a, b, t) { return a + (b - a) * t; }

function startRAF() {
    function frame() {
        if (!isMonitoring) return;
        rafHandle = requestAnimationFrame(frame);
        renderOverlay();
    }
    rafHandle = requestAnimationFrame(frame);
}

function renderOverlay() {
    const W = overlayCanvas.width;
    const H = overlayCanvas.height;
    overlayCtx.clearRect(0, 0, W, H);

    const LERP_T = 0.35; // Smoothing factor (lower = smoother but laggier)

    // Lerp face box
    if (faceLerp.active) {
        faceLerp.y1 = lerpVal(faceLerp.y1 ?? faceLerp.ty1, faceLerp.ty1, LERP_T);
        faceLerp.x1 = lerpVal(faceLerp.x1 ?? faceLerp.tx1, faceLerp.tx1, LERP_T);
        faceLerp.y2 = lerpVal(faceLerp.y2 ?? faceLerp.ty2, faceLerp.ty2, LERP_T);
        faceLerp.x2 = lerpVal(faceLerp.x2 ?? faceLerp.tx2, faceLerp.tx2, LERP_T);
        drawBox(W, H, faceLerp, '#14b8a6',
                `Face: ${document.getElementById('emotion-val').textContent}`);
    }

    // Lerp hand box
    if (handLerp.active) {
        handLerp.y1 = lerpVal(handLerp.y1 ?? handLerp.ty1, handLerp.ty1, LERP_T);
        handLerp.x1 = lerpVal(handLerp.x1 ?? handLerp.tx1, handLerp.tx1, LERP_T);
        handLerp.y2 = lerpVal(handLerp.y2 ?? handLerp.ty2, handLerp.ty2, LERP_T);
        handLerp.x2 = lerpVal(handLerp.x2 ?? handLerp.tx2, handLerp.tx2, LERP_T);
        drawBox(W, H, handLerp, '#0284c7',
                `Gesture: ${document.getElementById('gesture-val').textContent}`);
    }
}

function drawBox(W, H, lerp, color, label) {
    // Mirror X axis (video is CSS-flipped with scaleX(-1))
    const x1 = Math.floor((1 - lerp.x2 / 1000) * W);
    const x2 = Math.floor((1 - lerp.x1 / 1000) * W);
    const y1 = Math.floor((lerp.y1 / 1000) * H);
    const y2 = Math.floor((lerp.y2 / 1000) * H);
    const bw = x2 - x1, bh = y2 - y1;
    if (bw < 4 || bh < 4) return;

    // ── Corner-bracket style box (more visible than plain rect) ────────────
    overlayCtx.save();
    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth   = 2.5;
    overlayCtx.shadowColor = color;
    overlayCtx.shadowBlur  = 10;
    overlayCtx.lineCap     = 'round';

    const arm = Math.min(bw, bh) * 0.22; // corner arm length

    overlayCtx.beginPath();
    // top-left
    overlayCtx.moveTo(x1, y1 + arm); overlayCtx.lineTo(x1, y1); overlayCtx.lineTo(x1 + arm, y1);
    // top-right
    overlayCtx.moveTo(x2 - arm, y1); overlayCtx.lineTo(x2, y1); overlayCtx.lineTo(x2, y1 + arm);
    // bottom-left
    overlayCtx.moveTo(x1, y2 - arm); overlayCtx.lineTo(x1, y2); overlayCtx.lineTo(x1 + arm, y2);
    // bottom-right
    overlayCtx.moveTo(x2 - arm, y2); overlayCtx.lineTo(x2, y2); overlayCtx.lineTo(x2, y2 - arm);
    overlayCtx.stroke();

    // Semi-transparent fill
    overlayCtx.shadowBlur = 0;
    overlayCtx.fillStyle  = color + '14'; // ~8% opacity fill
    overlayCtx.fillRect(x1, y1, bw, bh);

    // ── Label badge (always visible, clamp inside canvas) ──────────────────
    overlayCtx.font = 'bold 11px Outfit, sans-serif';
    overlayCtx.shadowBlur = 0;
    const tw  = overlayCtx.measureText(label).width;
    const bPad= 6;
    const bW  = tw + bPad * 2;
    const bH  = 18;
    const bX  = Math.max(0, Math.min(x1, W - bW));
    // Place label above box if room, else inside top of box
    const bY  = y1 >= bH + 4 ? y1 - bH - 2 : y1 + 2;

    overlayCtx.fillStyle = color + 'DD';
    overlayCtx.fillRect(bX, bY, bW, bH);
    overlayCtx.fillStyle = '#ffffff';
    overlayCtx.fillText(label, bX + bPad, bY + 13);

    overlayCtx.restore();
}

function clearOverlay() {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    faceLerp.active = false;
    handLerp.active = false;
}

// ── Canvas Sync ───────────────────────────────────────────────────────────────
function syncCanvas() {
    const vW = viewport.clientWidth  || 640;
    const vH = viewport.clientHeight || 360;
    if (overlayCanvas.width !== vW || overlayCanvas.height !== vH) {
        overlayCanvas.width  = vW;
        overlayCanvas.height = vH;
    }
}

// ResizeObserver keeps canvas in sync when panel resizes (no manual call needed)
const _ro = new ResizeObserver(() => syncCanvas());
_ro.observe(viewport);

// ── Activity Log ──────────────────────────────────────────────────────────────
function appendLog(need, emotion, mapping) {
    if (logEmpty) logEmpty.style.display = 'none';

    const t    = new Date().toLocaleTimeString([], { hour:'2-digit', minute:'2-digit', second:'2-digit' });
    const item = document.createElement('div');
    item.className = `log-item ${mapping.log}`;
    item.innerHTML = `
        <span class="log-dot"></span>
        <span class="log-time">${t}</span>
        <span class="log-text">Status: <strong>${need}</strong></span>
        <span class="log-emotion">${emotion}</span>`;
    logList.insertBefore(item, logList.firstChild);

    // Cap at 35 entries
    const items = logList.querySelectorAll('.log-item');
    if (items.length > 35) items[items.length - 1].remove();
}

function clearLog() {
    logList.innerHTML = '';
    const ph = document.createElement('div');
    ph.className = 'log-empty';
    ph.id        = 'log-empty';
    ph.textContent = 'No events yet. Start monitoring to record alerts here.';
    logList.appendChild(ph);
}

// ── Reference Manual ──────────────────────────────────────────────────────────
function toggleManual() {
    const open = manualToggle.getAttribute('aria-expanded') === 'true';
    manualToggle.setAttribute('aria-expanded', String(!open));
    if (open) {
        manualBody.hidden = true;
    } else {
        manualBody.hidden = false;
    }
}

// Settings modal saved
function saveSettings() {
    const key   = inputApiKey.value.trim();
    const model = selectModel.value;
    localStorage.setItem('ae_api_key', key);
    localStorage.setItem('ae_model',   model);
    closeModal();
    setStatus(key || serverHasKey ? 'active' : 'error',
              key ? 'API Key Updated' : serverHasKey ? 'Server Key Active' : 'API Key Required');
}

function switchTab(paneId) {
    document.querySelectorAll('.tab-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.pane === paneId);
        b.setAttribute('aria-selected', String(b.dataset.pane === paneId));
    });
    document.querySelectorAll('.tab-pane').forEach(p => {
        p.hidden = (p.id !== paneId);
        p.classList.toggle('active', p.id === paneId);
    });
}
