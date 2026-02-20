(() => {
  const root = document.getElementById("calibration-root");
  if (!root) return;
  const token = root.dataset.token;
  const output = document.getElementById("cal-output");
  const captureBtn = document.getElementById("cal-capture");
  const autoToggleBtn = document.getElementById("cal-auto-toggle");
  const globalReadiness = document.getElementById("cal-global");
  const autoStatus = document.getElementById("cal-auto-status");
  const camReadiness = document.getElementById("cal-cam-readiness");
  const scoreHead = document.getElementById("cal-score-head");
  const scoreTips = document.getElementById("cal-score-tips");
  let autoEnabled = false;
  let autoInFlight = false;
  let lastAttemptTs = 0;
  let autoPollMs = 400;
  let latestReadiness = null;

  async function post(path, body) {
    const opts = {
      method: "POST",
      headers: { "x-access-token": token, "content-type": "application/json" },
    };
    if (body) opts.body = JSON.stringify(body);
    const resp = await fetch(path, opts);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `${path}: ${resp.status}`);
    }
    return await resp.json();
  }

  async function get(path) {
    const resp = await fetch(path, { headers: { "x-access-token": token } });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `${path}: ${resp.status}`);
    }
    return await resp.json();
  }

  function statusDot(ok) {
    return ok ? "ok-pill" : "bad-pill";
  }

  function renderReadiness(data) {
    latestReadiness = data;
    if (!data || !data.active) {
      globalReadiness.textContent = "No active calibration session.";
      camReadiness.innerHTML = "";
      captureBtn.disabled = true;
      if (autoStatus) {
        autoStatus.textContent = autoEnabled
          ? "Auto capture is on. Waiting for an active session."
          : "Auto capture is off.";
      }
      return;
    }
    globalReadiness.innerHTML = `
      <span class="diag-pill ${statusDot(data.all_cameras_ready)}">all ready: ${data.all_cameras_ready}</span>
      <span class="diag-pill">effective latency: ${Math.round(data.effective_latency_ms)} ms</span>
      <span class="diag-pill">sync skew: ${Math.round(data.sync_skew_ms)} ms</span>
      <span class="diag-pill">recommended fps: ${data.recommended_fps_cap ?? "n/a"}</span>
      <span class="diag-pill">captures: ${data.captures}/${data.required}</span>
    `;
    const board = data.board_metrics || {};
    const poseDelta = board.pose_delta == null ? "n/a" : Number(board.pose_delta).toFixed(3);
    const stable = board.stable_ms == null ? "0" : Math.round(board.stable_ms);
    const area = board.board_area_norm == null ? "n/a" : Number(board.board_area_norm).toFixed(4);
    const reason = data.capture_block_reason || "ready";
    if (autoStatus) {
      autoStatus.innerHTML = `
        <span class="diag-pill ${statusDot(Boolean(board.quality_ok))}">quality: ${Boolean(board.quality_ok)}</span>
        <span class="diag-pill">pose delta: ${poseDelta}</span>
        <span class="diag-pill">stable: ${stable} ms</span>
        <span class="diag-pill">area: ${area}</span>
        <span class="diag-pill">auto block: ${reason}</span>
      `;
    }
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
    const boardDetected = Object.values(data.per_camera || {}).every((cam) => Boolean(cam.checkerboard_detected));
    captureBtn.disabled = !(data.all_cameras_ready && boardDetected);
    if (autoEnabled && Number(data.captures || 0) >= Number(data.required || 0)) {
      autoEnabled = false;
      if (autoToggleBtn) autoToggleBtn.textContent = "Auto Capture: Off";
      if (autoStatus) {
        autoStatus.innerHTML += ' <span class="diag-pill">auto stopped: minimum captures reached</span>';
      }
    }
  }

  function extractScorePayload(data) {
    if (!data || typeof data !== "object") return null;
    if (data.calibration_score) return data.calibration_score; // /api/calibration/solve response
    if (data.result?.calibration_score) return data.result.calibration_score; // /api/calibration/report response
    return null;
  }

  function renderScore(data) {
    const score = extractScorePayload(data);
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
    try {
      const data = await get("/api/calibration/readiness");
      renderReadiness(data);
    } catch (err) {
      globalReadiness.textContent = String(err.message || err);
    }
  }

  async function refreshScore() {
    try {
      const data = await get("/api/calibration/report");
      renderScore(data);
    } catch (err) {
      scoreHead.textContent = String(err.message || err);
      scoreTips.innerHTML = "";
    }
  }

  document.getElementById("cal-start")?.addEventListener("click", async () => {
    const ids = document
      .getElementById("camera-ids")
      .value.split(",")
      .map((v) => v.trim())
      .filter(Boolean);
    const data = await post("/api/calibration/start", { camera_ids: ids });
    output.textContent = JSON.stringify(data, null, 2);
    autoEnabled = false;
    if (autoToggleBtn) autoToggleBtn.textContent = "Auto Capture: Off";
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
    const data = await post("/api/calibration/capture", { mode: "manual" });
    output.textContent = JSON.stringify(data, null, 2);
    await refreshReadiness();
    await refreshScore();
  });

  autoToggleBtn?.addEventListener("click", () => {
    autoEnabled = !autoEnabled;
    autoToggleBtn.textContent = autoEnabled ? "Auto Capture: On" : "Auto Capture: Off";
    if (autoStatus) {
      autoStatus.textContent = autoEnabled
        ? "Auto capture is on."
        : "Auto capture is off.";
    }
    if (output) {
      output.textContent = JSON.stringify(
        { ok: true, event: "auto_toggle", enabled: autoEnabled, at_ms: Date.now() },
        null,
        2
      );
    }
  });

  document.getElementById("cal-solve")?.addEventListener("click", async () => {
    try {
      const data = await post("/api/calibration/solve");
      output.textContent = JSON.stringify(data, null, 2);
      // Show score/tips immediately from solve response.
      renderScore(data);
      await refreshReadiness();
      await refreshScore();
    } catch (err) {
      output.textContent = JSON.stringify({ ok: false, error: String(err.message || err) }, null, 2);
    }
  });

  async function maybeAutoCapture() {
    if (!autoEnabled || autoInFlight) return;
    if (!latestReadiness || !latestReadiness.active) return;
    const now = Date.now();
    if (now - lastAttemptTs < autoPollMs) return;
    autoInFlight = true;
    lastAttemptTs = now;
    try {
      const data = await post("/api/calibration/capture", { mode: "auto" });
      output.textContent = JSON.stringify(data, null, 2);
      if (data?.ok === false && data?.rejection_reason) {
        if (autoStatus) autoStatus.innerHTML = `<span class="diag-pill bad-pill">auto reject: ${data.rejection_reason}</span>`;
      }
    } catch (err) {
      if (autoStatus) autoStatus.innerHTML = `<span class="diag-pill bad-pill">auto error: ${String(err.message || err)}</span>`;
    } finally {
      autoInFlight = false;
      await refreshReadiness();
      await refreshScore();
    }
  }

  setInterval(async () => {
    await refreshReadiness();
    await maybeAutoCapture();
  }, 900);
  refreshReadiness();
  refreshScore();
})();
