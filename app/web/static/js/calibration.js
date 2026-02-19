(() => {
  const root = document.getElementById("calibration-root");
  if (!root) return;
  const token = root.dataset.token;
  const output = document.getElementById("cal-output");
  const captureBtn = document.getElementById("cal-capture");
  const globalReadiness = document.getElementById("cal-global");
  const camReadiness = document.getElementById("cal-cam-readiness");
  const scoreHead = document.getElementById("cal-score-head");
  const scoreTips = document.getElementById("cal-score-tips");

  async function post(path, body) {
    const opts = {
      method: "POST",
      headers: { "x-access-token": token, "content-type": "application/json" },
    };
    if (body) opts.body = JSON.stringify(body);
    const resp = await fetch(path, opts);
    return await resp.json();
  }

  async function get(path) {
    const resp = await fetch(path, { headers: { "x-access-token": token } });
    return await resp.json();
  }

  function statusDot(ok) {
    return ok ? "ok-pill" : "bad-pill";
  }

  function renderReadiness(data) {
    if (!data || !data.active) {
      globalReadiness.textContent = "No active calibration session.";
      camReadiness.innerHTML = "";
      captureBtn.disabled = true;
      return;
    }
    globalReadiness.innerHTML = `
      <span class="diag-pill ${statusDot(data.all_cameras_ready)}">all ready: ${data.all_cameras_ready}</span>
      <span class="diag-pill">effective latency: ${Math.round(data.effective_latency_ms)} ms</span>
      <span class="diag-pill">sync skew: ${Math.round(data.sync_skew_ms)} ms</span>
      <span class="diag-pill">recommended fps: ${data.recommended_fps_cap ?? "n/a"}</span>
      <span class="diag-pill">captures: ${data.captures}/${data.required}</span>
    `;
    const entries = Object.entries(data.per_camera || {});
    camReadiness.innerHTML = entries
      .map(([id, cam]) => {
        const age = cam.latest_frame_age_ms == null ? "n/a" : `${Math.round(cam.latest_frame_age_ms)} ms`;
        return `
          <div class="roster-card">
            <strong>${id}</strong>
            <span class="diag-pill ${statusDot(cam.connected)}">connected: ${cam.connected}</span>
            <span class="diag-pill ${statusDot(cam.in_sync)}">in sync: ${cam.in_sync}</span>
            <span class="diag-pill ${statusDot(cam.checkerboard_detected)}">board: ${cam.checkerboard_detected}</span>
            <span class="diag-pill">age: ${age}</span>
          </div>
        `;
      })
      .join("");
    captureBtn.disabled = !data.all_cameras_ready;
  }

  function renderScore(data) {
    const score = data?.result?.calibration_score;
    if (!score) {
      scoreHead.textContent = "Run solve to generate a score.";
      scoreTips.innerHTML = "";
      return;
    }
    scoreHead.innerHTML = `
      <span class="diag-pill">score: ${score.score}/100</span>
      <span class="diag-pill">rating: ${score.rating}</span>
      <span class="diag-pill">verdict: ${score.verdict}</span>
      <span class="diag-pill">intrinsic rms avg: ${score.intrinsic_rms_avg == null ? "n/a" : Number(score.intrinsic_rms_avg).toFixed(3)}</span>
      <span class="diag-pill">pair rms avg: ${score.pair_rms_avg == null ? "n/a" : Number(score.pair_rms_avg).toFixed(3)}</span>
    `;
    const tips = Array.isArray(score.tips) ? score.tips : [];
    scoreTips.innerHTML = tips.map((tip) => `<li class="wizard-step">${tip}</li>`).join("");
  }

  async function refreshReadiness() {
    const data = await get("/api/calibration/readiness");
    renderReadiness(data);
  }

  async function refreshScore() {
    const data = await get("/api/calibration/report");
    renderScore(data);
  }

  document.getElementById("cal-start")?.addEventListener("click", async () => {
    const ids = document
      .getElementById("camera-ids")
      .value.split(",")
      .map((v) => v.trim())
      .filter(Boolean);
    const data = await post("/api/calibration/start", { camera_ids: ids });
    output.textContent = JSON.stringify(data, null, 2);
    await refreshReadiness();
    await refreshScore();
  });

  document.getElementById("cal-load-cameras")?.addEventListener("click", async () => {
    const resp = await fetch("/api/cameras/roster", { headers: { "x-access-token": token } });
    const cams = await resp.json();
    const connected = cams.filter((c) => c.connected).map((c) => c.camera_id);
    document.getElementById("camera-ids").value = connected.join(",");
    output.textContent = JSON.stringify({ connected }, null, 2);
    await refreshReadiness();
    await refreshScore();
  });

  captureBtn?.addEventListener("click", async () => {
    const data = await post("/api/calibration/capture");
    output.textContent = JSON.stringify(data, null, 2);
    await refreshReadiness();
    await refreshScore();
  });

  document.getElementById("cal-solve")?.addEventListener("click", async () => {
    const data = await post("/api/calibration/solve");
    output.textContent = JSON.stringify(data, null, 2);
    await refreshReadiness();
    await refreshScore();
  });

  setInterval(refreshReadiness, 900);
  refreshReadiness();
  refreshScore();
})();
