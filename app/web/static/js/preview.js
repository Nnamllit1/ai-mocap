(() => {
  const canvas = document.getElementById("pose3d-canvas");
  if (!canvas) return;
  const token = canvas.dataset.token;
  const ctx = canvas.getContext("2d");
  const camGrid = document.getElementById("cam-grid");
  const runtimeRoot = document.getElementById("preview-runtime");

  const COCO_EDGES = [
    [5, 6],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [5, 11],
    [6, 12],
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
  ];

  function confColor(conf) {
    if (conf == null || Number.isNaN(conf)) return "#9aa5b1";
    if (conf >= 0.75) return "#22c55e";
    if (conf >= 0.45) return "#f59e0b";
    return "#ef4444";
  }

  async function req(path) {
    const resp = await fetch(path, { headers: { "x-access-token": token } });
    if (!resp.ok) {
      throw new Error(`${path}: ${resp.status}`);
    }
    return await resp.json();
  }

  function drawPose(pointsByIndex, confidences = {}) {
    const points = Object.entries(pointsByIndex || {}).map(([idx, xyz]) => ({
      idx: Number(idx),
      xyz,
      conf: confidences[idx] == null ? null : Number(confidences[idx]),
    }));

    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw bones first.
    ctx.lineWidth = 2;
    for (const [a, b] of COCO_EDGES) {
      const pa = points.find((p) => p.idx === a);
      const pb = points.find((p) => p.idx === b);
      if (!pa || !pb) continue;
      const ax = canvas.width * 0.5 + Number(pa.xyz[0]) * 100;
      const ay = canvas.height * 0.7 + Number(pa.xyz[1]) * 100;
      const bx = canvas.width * 0.5 + Number(pb.xyz[0]) * 100;
      const by = canvas.height * 0.7 + Number(pb.xyz[1]) * 100;
      const c = confColor(
        pa.conf == null || pb.conf == null ? null : Math.min(pa.conf, pb.conf)
      );
      ctx.strokeStyle = c;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }

    // Draw joints over the bones.
    for (const p of points) {
      const sx = canvas.width * 0.5 + Number(p.xyz[0]) * 100;
      const sy = canvas.height * 0.7 + Number(p.xyz[1]) * 100;
      ctx.fillStyle = confColor(p.conf);
      ctx.beginPath();
      ctx.arc(sx, sy, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function renderRuntime(metrics = {}) {
    if (!runtimeRoot) return;
    const running = !!metrics.running;
    const fps = Number(metrics.loop_fps || 0);
    const loopMs = Number(metrics.loop_ms || 0);
    const cams = Number(metrics.active_cameras || 0);
    const valid = Number(metrics.valid_joints || 0);
    const dropped = Number(metrics.dropped_cycles || 0);
    const conf = metrics.joint_conf_avg == null ? "n/a" : Number(metrics.joint_conf_avg).toFixed(2);

    runtimeRoot.innerHTML = `
      <span class="diag-pill ${running ? "ok-pill" : "bad-pill"}">running: ${running}</span>
      <span class="diag-pill">schema: COCO-17</span>
      <span class="diag-pill">cameras: ${cams}</span>
      <span class="diag-pill">valid joints: ${valid}</span>
      <span class="diag-pill">fps: ${fps.toFixed(1)}</span>
      <span class="diag-pill">loop: ${loopMs.toFixed(1)} ms</span>
      <span class="diag-pill">joint conf avg: ${conf}</span>
      <span class="diag-pill">dropped cycles: ${dropped}</span>
    `;
  }

  async function refreshPose() {
    try {
      const data = await req("/api/preview/pose3d");
      drawPose(data.joints || {}, data.confidences || {});
      renderRuntime(data.metrics || {});
    } catch (_) {}
  }

  function ageText(lastSeen) {
    if (!lastSeen) return "n/a";
    const ageMs = Math.max(0, Date.now() - Number(lastSeen) * 1000);
    return `${Math.round(ageMs)} ms`;
  }

  async function refreshCameraSnapshots() {
    if (!camGrid) return;
    let cameras = [];
    try {
      cameras = await req("/api/cameras/roster");
    } catch (_) {
      return;
    }

    camGrid.innerHTML = cameras
      .map(
        (cam) => `
      <div class="cam-item">
        <div class="cam-meta">
          <p>${cam.label} <span class="muted">(${cam.camera_id})</span></p>
          <p class="muted">
            status: ${cam.connected ? "connected" : "offline"} |
            seq: ${cam.seq || 0} |
            age: ${ageText(cam.last_seen)}
          </p>
        </div>
        <img id="cam-${cam.camera_id}" alt="${cam.camera_id}">
      </div>
    `
      )
      .join("");

    const ts = Date.now();
    cameras.forEach((cam) => {
      const img = document.getElementById(`cam-${cam.camera_id}`);
      if (!img) return;
      img.src = `/api/preview/camera/${cam.camera_id}?token=${encodeURIComponent(token)}&t=${ts}`;
    });
  }

  setInterval(refreshPose, 180);
  setInterval(refreshCameraSnapshots, 900);
  refreshPose();
  refreshCameraSnapshots();
})();
