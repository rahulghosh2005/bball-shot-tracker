// ─── Shot Tracker — Basketball Analytics ─────────────────────────────────────

const video      = document.getElementById('webcam');
const canvas     = document.getElementById('canvas');
const ctx        = canvas.getContext('2d');
const loader     = document.getElementById('loader');
const loaderTxt  = document.getElementById('loader-text');
const madeEl     = document.getElementById('made-count');
const missEl     = document.getElementById('miss-count');
const pctEl      = document.getElementById('pct-count');
const feedbackEl = document.getElementById('shot-feedback');
const fpsEl      = document.getElementById('fps-display');
const detectEl   = document.getElementById('detect-display');
const angleEl    = document.getElementById('angle-display');
const hoopLabel  = document.getElementById('hoop-mode-label');
const setHoopBtn = document.getElementById('set-hoop-btn');

// ─── Settings ────────────────────────────────────────────────────────────────
const settings = {
  showSkeleton: true,
  showHands:    true,
  showBall:     true,
  showTrails:   true,
  videoOpacity: 0.70,
};

// ─── Models ──────────────────────────────────────────────────────────────────
let poseDetector  = null;
let handDetector  = null;
let cocoModel     = null;

let currentPoses  = [];
let currentHands  = [];
let currentBalls  = []; // COCO sports ball detections

// ─── Shot tracking state ──────────────────────────────────────────────────────
let hoopPoint     = null;   // {x, y} in canvas px — user-defined hoop position
const HOOP_RADIUS = 55;     // px — how close ball must get to count as "through hoop"

const ballHistory = [];     // [{x, y, t}] canvas coords of ball center, last ~90 frames
const MAX_BALL_HIST = 90;
let   shotInFlight = false;
let   shotPeakY    = Infinity;
let   lastBallSeen = -999;

const shotTrails   = [];    // [{points, made, ts}] last 8 shots
const MAX_TRAILS   = 8;

const stats = { made: 0, missed: 0 };

let settingHoop   = false;
let frameCount    = 0;
let fps           = 0;
let fpsTimer      = 0;

// ─── MoveNet skeleton connections ────────────────────────────────────────────
// keypoint order: nose(0) left_eye(1) right_eye(2) left_ear(3) right_ear(4)
//   left_shoulder(5) right_shoulder(6) left_elbow(7) right_elbow(8)
//   left_wrist(9) right_wrist(10) left_hip(11) right_hip(12)
//   left_knee(13) right_knee(14) left_ankle(15) right_ankle(16)
const POSE_LINES = [
  [5,6],[5,7],[7,9],[6,8],[8,10],   // arms
  [5,11],[6,12],[11,12],             // torso
  [11,13],[13,15],[12,14],[14,16],   // legs
  [0,1],[0,2],[1,3],[2,4],           // head
];

// ─── Hand connections ─────────────────────────────────────────────────────────
const HAND_LINES = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

// ─── Draw body skeleton (MoveNet) ─────────────────────────────────────────────
function drawSkeleton(poses) {
  if (!settings.showSkeleton || !poses.length) return;
  const sx = canvas.width  / video.videoWidth;
  const sy = canvas.height / video.videoHeight;

  for (const pose of poses) {
    const kp = pose.keypoints;

    // Connections
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.75)';
    ctx.lineWidth   = 2;
    ctx.lineCap     = 'round';
    for (const [a, b] of POSE_LINES) {
      if (kp[a].score < 0.3 || kp[b].score < 0.3) continue;
      ctx.beginPath();
      ctx.moveTo(kp[a].x * sx, kp[a].y * sy);
      ctx.lineTo(kp[b].x * sx, kp[b].y * sy);
      ctx.stroke();
    }

    // Keypoints
    for (const k of kp) {
      if (k.score < 0.3) continue;
      ctx.beginPath();
      ctx.arc(k.x * sx, k.y * sy, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    }
    ctx.restore();
  }
}

// ─── Draw hand skeleton (MediaPipe) ───────────────────────────────────────────
function drawHands(hands) {
  if (!settings.showHands || !hands.length) return;
  const sx = canvas.width  / video.videoWidth;
  const sy = canvas.height / video.videoHeight;

  for (const hand of hands) {
    const kp = hand.keypoints;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.lineWidth   = 1.5;
    ctx.lineCap     = 'round';
    for (const [a, b] of HAND_LINES) {
      if (!kp[a] || !kp[b]) continue;
      ctx.beginPath();
      ctx.moveTo(kp[a].x * sx, kp[a].y * sy);
      ctx.lineTo(kp[b].x * sx, kp[b].y * sy);
      ctx.stroke();
    }
    for (const k of kp) {
      const x = k.x * sx, y = k.y * sy;
      const isTip = [4,8,12,16,20].includes(kp.indexOf(k));
      ctx.beginPath();
      ctx.arc(x, y, isTip ? 5 : 3, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    }
    ctx.restore();
  }
}

// ─── Draw detected objects (non-person COCO) ──────────────────────────────────
function drawObjects(predictions) {
  if (!predictions.length) return;
  const sx = canvas.width  / video.videoWidth;
  const sy = canvas.height / video.videoHeight;

  for (const pred of predictions) {
    if (pred.class === 'person') continue; // skeleton handles persons
    const [x, y, w, h] = pred.bbox;
    const isBall = pred.class === 'sports ball';

    ctx.save();
    ctx.strokeStyle = isBall ? '#f59e0b' : 'rgba(255,255,255,0.6)';
    ctx.lineWidth   = isBall ? 2.5 : 1.5;
    ctx.strokeRect(x * sx, y * sy, w * sx, h * sy);

    ctx.fillStyle = isBall ? 'rgba(245,158,11,0.12)' : 'rgba(255,255,255,0.05)';
    ctx.fillRect(x * sx, y * sy, w * sx, h * sy);

    ctx.fillStyle = isBall ? '#f59e0b' : 'rgba(255,255,255,0.7)';
    ctx.font = `${isBall ? '600' : '400'} 11px "Helvetica Neue", Arial`;
    ctx.fillText(
      `${pred.class}  ${Math.round(pred.score * 100)}%`,
      x * sx + 5,
      y * sy - 5
    );
    ctx.restore();
  }
}

// ─── Ball tracking + shot detection ──────────────────────────────────────────
function getBallCanvasPos(balls) {
  if (!balls.length) return null;
  // Use highest-confidence sports ball
  const ball = balls.reduce((a, b) => b.score > a.score ? b : a);
  const [x, y, w, h] = ball.bbox;
  const sx = canvas.width  / video.videoWidth;
  const sy = canvas.height / video.videoHeight;
  return {
    x: (x + w / 2) * sx,
    y: (y + h / 2) * sy,
    w: w * sx,
    h: h * sy,
  };
}

function trackBall(ballPos) {
  const now = frameCount;

  if (!ballPos) {
    // Ball has disappeared
    if (shotInFlight && now - lastBallSeen > 15 && ballHistory.length >= 8) {
      evaluateShot();
    }
    if (now - lastBallSeen > 45) {
      ballHistory.length = 0;
      shotInFlight = false;
    }
    return;
  }

  lastBallSeen = now;
  ballHistory.push({ x: ballPos.x, y: ballPos.y, f: now });
  if (ballHistory.length > MAX_BALL_HIST) ballHistory.shift();

  // Detect upward motion = shot started
  if (ballHistory.length >= 6) {
    const recent = ballHistory.slice(-6);
    const dy = recent[5].y - recent[0].y; // negative = moving up on screen
    if (dy < -20 && !shotInFlight) {
      shotInFlight = true;
      shotPeakY    = ballPos.y;
    }
    if (shotInFlight) {
      shotPeakY = Math.min(shotPeakY, ballPos.y);
      // Ball now moving down after peak = shot completed
      if (dy > 20 && ballPos.y > shotPeakY + 30) {
        evaluateShot();
      }
    }
  }
}

function evaluateShot() {
  if (!hoopPoint) {
    shotInFlight = false;
    return;
  }
  if (ballHistory.length < 8) { shotInFlight = false; return; }

  // Find peak of shot arc
  const peakIdx = ballHistory.reduce((pi, p, i) => p.y < ballHistory[pi].y ? i : pi, 0);

  // Check if ball passed through hoop on downward arc
  let made = false;
  for (let i = peakIdx; i < ballHistory.length; i++) {
    const dist = Math.hypot(ballHistory[i].x - hoopPoint.x, ballHistory[i].y - hoopPoint.y);
    if (dist < HOOP_RADIUS) { made = true; break; }
  }

  stats[made ? 'made' : 'missed']++;
  updateStats();

  // Calculate release angle from first part of trajectory
  const releaseAngle = calcReleaseAngle(ballHistory, peakIdx);
  showFeedback(made, releaseAngle);

  // Store trail
  shotTrails.push({ points: ballHistory.slice(), made, ts: Date.now(), angle: releaseAngle });
  if (shotTrails.length > MAX_TRAILS) shotTrails.shift();

  // Reset
  ballHistory.length = 0;
  shotInFlight = false;
  shotPeakY    = Infinity;
}

function calcReleaseAngle(history, peakIdx) {
  // Use the first portion of the arc (before peak) to estimate angle
  const end   = Math.min(peakIdx, history.length - 1);
  const start = Math.max(0, end - 6);
  if (end <= start) return null;
  const dx = Math.abs(history[end].x - history[start].x);
  const dy = history[start].y - history[end].y; // positive = upward
  if (dx < 2) return 90;
  return Math.round(Math.atan2(dy, dx) * 180 / Math.PI);
}

function updateStats() {
  const total = stats.made + stats.missed;
  madeEl.textContent = stats.made;
  missEl.textContent = stats.missed;
  pctEl.textContent  = total ? `${Math.round((stats.made / total) * 100)}%` : '--%';
}

let feedbackTimer = null;
function showFeedback(made, angle) {
  feedbackEl.className = made ? 'made show' : 'missed show';
  feedbackEl.textContent = made ? 'IN' : 'MISS';

  if (angle !== null) {
    const tip = angle < 40 ? 'Arc too flat'
              : angle > 65 ? 'Arc too steep'
              : 'Good arc';
    angleEl.textContent = `Release ${angle}°  ·  ${tip}`;
  }

  clearTimeout(feedbackTimer);
  feedbackTimer = setTimeout(() => {
    feedbackEl.classList.remove('show');
    setTimeout(() => { feedbackEl.className = ''; feedbackEl.textContent = ''; }, 200);
  }, 1200);
}

// ─── Draw shot trails ─────────────────────────────────────────────────────────
function drawShotTrails() {
  if (!settings.showTrails || !shotTrails.length) return;
  const now = Date.now();

  for (const trail of shotTrails) {
    const age     = (now - trail.ts) / 1000;  // seconds old
    const opacity = Math.max(0, 1 - age / 12); // fade over 12 seconds
    if (opacity <= 0 || trail.points.length < 2) continue;

    ctx.save();
    ctx.globalAlpha = opacity * 0.7;
    ctx.strokeStyle = trail.made ? '#22c55e' : '#ef4444';
    ctx.lineWidth   = 2;
    ctx.lineCap     = 'round';
    ctx.lineJoin    = 'round';
    ctx.beginPath();
    ctx.moveTo(trail.points[0].x, trail.points[0].y);
    for (let i = 1; i < trail.points.length; i++) {
      ctx.lineTo(trail.points[i].x, trail.points[i].y);
    }
    ctx.stroke();

    // Dot at end of trail
    const last = trail.points.at(-1);
    ctx.beginPath();
    ctx.arc(last.x, last.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = trail.made ? '#22c55e' : '#ef4444';
    ctx.fill();
    ctx.restore();
  }
}

// ─── Draw current ball trail (in progress) ───────────────────────────────────
function drawBallTrail() {
  if (ballHistory.length < 2) return;
  ctx.save();
  ctx.strokeStyle = 'rgba(245,158,11,0.5)';
  ctx.lineWidth   = 2;
  ctx.lineCap     = 'round';
  ctx.beginPath();
  ctx.moveTo(ballHistory[0].x, ballHistory[0].y);
  for (let i = 1; i < ballHistory.length; i++) {
    ctx.lineTo(ballHistory[i].x, ballHistory[i].y);
  }
  ctx.stroke();
  ctx.restore();
}

// ─── Draw hoop indicator ──────────────────────────────────────────────────────
function drawHoop() {
  if (!hoopPoint) return;
  ctx.save();
  ctx.strokeStyle = '#3b82f6';
  ctx.lineWidth   = 2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.arc(hoopPoint.x, hoopPoint.y, HOOP_RADIUS, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  // Center dot
  ctx.beginPath();
  ctx.arc(hoopPoint.x, hoopPoint.y, 4, 0, Math.PI * 2);
  ctx.fillStyle = '#3b82f6';
  ctx.fill();
  // Label
  ctx.fillStyle = 'rgba(147,197,253,0.8)';
  ctx.font = '11px "Helvetica Neue", Arial';
  ctx.fillText('HOOP', hoopPoint.x + HOOP_RADIUS + 6, hoopPoint.y + 4);
  ctx.restore();
}

// ─── Run models async (non-blocking) ─────────────────────────────────────────
async function runModels() {
  if (video.readyState < 2) return;

  // Pose every frame
  if (poseDetector) {
    try { currentPoses = await poseDetector.estimatePoses(video); } catch (_) {}
  }

  // Hands every 2 frames
  if (handDetector && frameCount % 2 === 0) {
    try { currentHands = await handDetector.estimateHands(video, { flipHorizontal: false }); } catch (_) {}
  }

  // COCO every 4 frames (filtered to relevant objects)
  if (cocoModel && frameCount % 4 === 0) {
    try {
      const all  = await cocoModel.detect(video, 10, 0.4);
      currentBalls = all.filter(p => p.class === 'sports ball');

      // Store all non-person objects for display
      window._lastObjects = all.filter(p => p.class !== 'person');
    } catch (_) {}
  }
}

// ─── Main loop ────────────────────────────────────────────────────────────────
async function loop() {
  if (video.readyState < 2) { requestAnimationFrame(loop); return; }

  const now = performance.now();
  frameCount++;
  if (now - fpsTimer > 1000) { fps = frameCount; frameCount = 0; fpsTimer = now; }

  if (canvas.width !== window.innerWidth || canvas.height !== window.innerHeight) {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  runModels(); // async, non-blocking

  // Track ball and shot
  const ballPos = getBallCanvasPos(currentBalls);
  if (settings.showBall) trackBall(ballPos);

  // ── Render ──────────────────────────────────────────────────────────────────
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Shot trails (behind everything)
  drawShotTrails();
  drawBallTrail();

  // Body + hands
  drawSkeleton(currentPoses);
  drawHands(currentHands);

  // Other detected objects
  if (settings.showBall && window._lastObjects) drawObjects(window._lastObjects);

  // Hoop
  drawHoop();

  // Info
  const detected = [];
  if (currentPoses.length)  detected.push(`${currentPoses.length} body`);
  if (currentHands.length)  detected.push(`${currentHands.length} hand`);
  if (currentBalls.length)  detected.push('ball');
  detectEl.textContent = detected.join('  ·  ');
  fpsEl.textContent    = `${fps} fps`;

  requestAnimationFrame(loop);
}

// ─── Hoop placement ───────────────────────────────────────────────────────────
function toggleHoopMode() {
  settingHoop = !settingHoop;
  setHoopBtn.classList.toggle('active', settingHoop);
  hoopLabel.classList.toggle('hidden', !settingHoop);
  document.body.classList.toggle('hoop-mode', settingHoop);
}

canvas.addEventListener('click', e => {
  if (!settingHoop) return;
  const rect = canvas.getBoundingClientRect();
  hoopPoint = { x: e.clientX - rect.left, y: e.clientY - rect.top };
  toggleHoopMode();
});

// ─── Camera ───────────────────────────────────────────────────────────────────
async function startCamera(deviceId) {
  if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        width:  { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
      }
    });
    await video.play();
  } catch (e) { loaderTxt.textContent = 'Camera access denied.'; }
}

async function populateCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const sel     = document.getElementById('camera-select');
  sel.innerHTML  = '';
  devices.filter(d => d.kind === 'videoinput').forEach((cam, i) => {
    const opt       = document.createElement('option');
    opt.value       = cam.deviceId;
    opt.textContent = cam.label || `Camera ${i + 1}`;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', () => startCamera(sel.value));
}

// ─── Controls wiring ─────────────────────────────────────────────────────────
setHoopBtn.addEventListener('click', toggleHoopMode);

document.getElementById('reset-btn').addEventListener('click', () => {
  stats.made = stats.missed = 0;
  updateStats();
  shotTrails.length = 0;
  ballHistory.length = 0;
  shotInFlight = false;
  angleEl.textContent = '';
});

document.getElementById('toggle-panel-btn').addEventListener('click', () =>
  document.getElementById('settings-panel').classList.toggle('hidden'));

document.getElementById('show-skeleton').addEventListener('change', e =>
  settings.showSkeleton = e.target.checked);
document.getElementById('show-hands').addEventListener('change', e =>
  settings.showHands = e.target.checked);
document.getElementById('show-ball').addEventListener('change', e =>
  settings.showBall = e.target.checked);
document.getElementById('show-trails').addEventListener('change', e =>
  settings.showTrails = e.target.checked);

document.getElementById('opacity-slider').addEventListener('input', e => {
  video.style.opacity = +e.target.value / 100;
});

document.getElementById('fullscreen-btn').addEventListener('click', () =>
  document.fullscreenElement ? document.exitFullscreen()
                              : document.documentElement.requestFullscreen());

// ─── Init ─────────────────────────────────────────────────────────────────────
(async () => {
  try {
    await startCamera(null);
    await populateCameras();

    loaderTxt.textContent = 'Loading pose model...';
    await tf.setBackend('webgl');
    poseDetector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
    );

    loaderTxt.textContent = 'Loading hand model...';
    handDetector = await handPoseDetection.createDetector(
      handPoseDetection.SupportedModels.MediaPipeHands,
      {
        runtime:      'mediapipe',
        solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915',
        modelType:    'full',
        maxHands:     2,
      }
    );

    loaderTxt.textContent = 'Loading object model...';
    cocoModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });

    loader.classList.add('hidden');
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    requestAnimationFrame(loop);
  } catch (e) {
    loaderTxt.textContent = `Error: ${e.message}`;
    console.error(e);
  }
})();
