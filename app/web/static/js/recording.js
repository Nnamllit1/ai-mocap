(() => {
  const canvas = document.getElementById("recording-pose3d-canvas");
  if (!canvas) return;
  const token = canvas.dataset.token || "";
  const ctx = canvas.getContext("2d");
  const statusRoot = document.getElementById("recording-status");
  const noteEl = document.getElementById("recording-note");
  const startBtn = document.getElementById("recording-start");
  const stopBtn = document.getElementById("recording-stop");
  const clipHistory = document.getElementById("clip-history");
  const camGrid = document.getElementById("recording-cam-grid");

  const COCO_EDGES = [
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16], [0, 1], [0, 2], [1, 3], [2, 4],
  ];

  function confColor(conf) {
    if (conf == null || Number.isNaN(conf)) return "#9aa5b1";
    if (conf >= 0.75) return "#22c55e";
    if (conf >= 0.45) return "#f59e0b";
    return "#ef4444";
  }

  async function req(path, method = "GET", body = null) {
    const opts = { method, headers: { "x-access-token": token } };
    if (body != null) {
      opts.headers["content-type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const resp = await fetch(path, opts);
    const payload = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(payload.detail || `${path}: ${resp.status}`);
    }
    return payload;
  }

  function drawPose(jointStates) {
    const points = Object.entries(jointStates || {}).map(([idx, entry]) => ({
      idx: Number(idx),
      xyz: entry.xyz || [0, 0, 0],
      conf: entry.confidence == null ? null : Number(entry.confidence),
      state: entry.state || "measured",
    }));

    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;

    for (const [a, b] of COCO_EDGES) {
      const pa = points.find((p) => p.idx === a);
      const pb = points.find((p) => p.idx === b);
      if (!pa || !pb) continue;
      const ax = canvas.width * 0.5 + Number(pa.xyz[0]) * 100;
      const ay = canvas.height * 0.7 + Number(pa.xyz[1]) * 100;
      const bx = canvas.width * 0.5 + Number(pb.xyz[0]) * 100;
      const by = canvas.height * 0.7 + Number(pb.xyz[1]) * 100;
      const held = pa.state === "held" || pb.state === "held";
      ctx.strokeStyle = held ? "#94a3b8" : confColor(Math.min(pa.conf ?? 0, pb.conf ?? 0));
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }

    for (const p of points) {
      const sx = canvas.width * 0.5 + Number(p.xyz[0]) * 100;
      const sy = canvas.height * 0.7 + Number(p.xyz[1]) * 100;
      ctx.fillStyle = p.state === "held" ? "#94a3b8" : confColor(p.conf);
      ctx.beginPath();
      ctx.arc(sx, sy, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function ageText(lastSeen) {
    if (!lastSeen) return "n/a";
    const ageMs = Math.max(0, Date.now() - Number(lastSeen) * 1000);
    return `${Math.round(ageMs)} ms`;
  }

  function elapsedText(clip) {
    if (!clip || !clip.started_at) return "0.0 s";
    const stoppedAt = clip.stopped_at || (Date.now() / 1000);
    const elapsed = Math.max(0, Number(stoppedAt) - Number(clip.started_at));
    return `${elapsed.toFixed(1)} s`;
  }

  function renderRecordingStatus(status) {
    if (!statusRoot) return;
    const activeClip = status.active_clip;
    const state = status.state || "idle";
    const frames = Number(activeClip?.frame_count || 0);
    const samples = Number(activeClip?.joint_samples || 0);
    statusRoot.innerHTML = `
      <span class="diag-pill ${state === "recording" ? "ok-pill" : "bad-pill"}">state: ${state}</span>
      <span class="diag-pill">clip: ${activeClip?.clip_id || "n/a"}</span>
      <span class="diag-pill">elapsed: ${elapsedText(activeClip)}</span>
      <span class="diag-pill">frames: ${frames}</span>
      <span class="diag-pill">samples: ${samples}</span>
    `;
  }

  function renderClipHistory(clips) {
    if (!clipHistory) return;
    if (!clips || clips.length === 0) {
      clipHistory.innerHTML = `<p class="muted">No clips yet.</p>`;
      return;
    }
    clipHistory.innerHTML = clips.map((clip) => `
      <div class="roster-card">
        <span>
          <strong>${clip.clip_id}</strong>
          <span class="muted">status: ${clip.status} | frames: ${clip.frame_count} | samples: ${clip.joint_samples}</span>
        </span>
        <button class="clip-export-btn" data-clip-id="${clip.clip_id}" ${clip.status === "recording" ? "disabled" : ""}>Export</button>
      </div>
    `).join("");
    clipHistory.querySelectorAll(".clip-export-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const clipId = btn.dataset.clipId;
        if (!clipId) return;
        try {
          const out = await req(`/api/recordings/${encodeURIComponent(clipId)}/export`, "POST");
          noteEl.textContent = `Exported ${clipId} to ${out.paths?.json || "json"} and ${out.paths?.csv || "csv"}.`;
          await refreshRecordings();
        } catch (err) {
          noteEl.textContent = String(err.message || err);
        }
      });
    });
  }

  async function refreshPose() {
    try {
      const data = await req("/api/preview/pose3d");
      drawPose(data.joint_states || {});
    } catch (_) {}
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
      .map((cam) => `
        <div class="cam-item">
          <div class="cam-meta">
            <p>${cam.label} <span class="muted">(${cam.camera_id})</span></p>
            <p class="muted">status: ${cam.connected ? "connected" : "offline"} | seq: ${cam.seq || 0} | age: ${ageText(cam.last_seen)}</p>
          </div>
          <img id="cam-${cam.camera_id}" alt="${cam.camera_id}">
        </div>
      `)
      .join("");
    const ts = Date.now();
    cameras.forEach((cam) => {
      const img = document.getElementById(`cam-${cam.camera_id}`);
      if (!img) return;
      img.src = `/api/preview/camera/${cam.camera_id}?token=${encodeURIComponent(token)}&t=${ts}`;
    });
  }

  async function refreshRecordings() {
    try {
      const [status, clips] = await Promise.all([
        req("/api/recordings/status"),
        req("/api/recordings"),
      ]);
      renderRecordingStatus(status);
      renderClipHistory(clips);
    } catch (_) {}
  }

  startBtn?.addEventListener("click", async () => {
    try {
      const out = await req("/api/recordings/start", "POST");
      noteEl.textContent = out.warning || out.message || "";
    } catch (err) {
      noteEl.textContent = String(err.message || err);
    }
    await refreshRecordings();
  });

  stopBtn?.addEventListener("click", async () => {
    try {
      const out = await req("/api/recordings/stop", "POST");
      noteEl.textContent = out.message || "";
    } catch (err) {
      noteEl.textContent = String(err.message || err);
    }
    await refreshRecordings();
  });

  setInterval(refreshPose, 180);
  setInterval(refreshCameraSnapshots, 900);
  setInterval(refreshRecordings, 500);
  refreshPose();
  refreshCameraSnapshots();
  refreshRecordings();
})();
