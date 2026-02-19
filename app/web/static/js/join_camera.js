(() => {
  const STATES = {
    IDLE: "idle",
    DIAGNOSE: "diagnose",
    BLOCKED_HTTP: "blocked_http",
    WIZARD: "wizard",
    RETRYING: "retrying",
    PERMISSION: "permission",
    REGISTERING: "registering",
    STREAMING: "streaming",
    ERROR: "error",
  };

  const video = document.getElementById("video");
  const statusEl = document.getElementById("ws-status");
  const startBtn = document.getElementById("start-btn");
  const stopBtn = document.getElementById("stop-btn");
  const retryBtn = document.getElementById("retry-btn");
  const copyOriginBtn = document.getElementById("copy-origin-btn");
  const copyFlagBtn = document.getElementById("copy-flag-btn");
  const copyChromiumOriginBtn = document.getElementById("copy-chromium-origin-btn");
  const wizardTitle = document.getElementById("wizard-title");
  const wizardStepsAndroid = document.getElementById("wizard-steps-android");
  const wizardStepsFirefox = document.getElementById("wizard-steps-firefox");
  const wizardStepsChromium = document.getElementById("wizard-steps-chromium");
  const chromiumFlagsLink = document.getElementById("chromium-flags-link");
  const originCopyRow = document.getElementById("origin-copy-row");
  const chromiumFlagRow = document.getElementById("chromium-flag-row");
  const chromiumFlagInput = document.getElementById("chromium-flag-input");
  const chromiumOriginRow = document.getElementById("chromium-origin-row");
  const chromiumOriginInput = document.getElementById("chromium-origin-input");
  const chromiumNote = document.getElementById("chromium-note");
  const originInput = document.getElementById("origin-input");
  const diagSummary = document.getElementById("diag-summary");
  const wizardSection = document.getElementById("join-wizard");
  const helpSection = document.getElementById("join-help");
  const helpMessage = document.getElementById("help-message");
  const wizardNote = document.getElementById("wizard-note");
  const labelInput = document.getElementById("camera-label");
  const fpsInput = document.getElementById("fps");
  const qualityInput = document.getElementById("quality");
  const ticketId = window.JOIN_TICKET;
  const validTicket = window.VALID_TICKET === "1";
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  const DEVICE_KEY = "mocap_device_uid";
  const LABEL_KEY = "mocap_device_label";

  let ws = null;
  let stream = null;
  let timer = null;
  let heart = null;
  let assignedCameraId = null;

  function setState(state, message) {
    statusEl.textContent = `${state}${message ? `: ${message}` : ""}`;
  }

  function hideWizard() {
    wizardSection.classList.add("hidden");
  }

  function showWizard() {
    wizardSection.classList.remove("hidden");
  }

  function showHelp(msg) {
    helpSection.classList.remove("hidden");
    helpMessage.textContent = msg;
  }

  function hideHelp() {
    helpSection.classList.add("hidden");
    helpMessage.textContent = "";
  }

  function renderDiag(diagnostics) {
    const items = [
      { name: "Android", value: diagnostics.isAndroid },
      { name: "Firefox", value: diagnostics.isFirefox },
      { name: "Chromium", value: diagnostics.isChromiumFamily },
      { name: "Desktop", value: diagnostics.isDesktop },
      { name: "Secure Context", value: diagnostics.secureContext },
      { name: "mediaDevices", value: diagnostics.hasMediaDevices },
      { name: "getUserMedia", value: diagnostics.hasGetUserMedia },
    ];
    diagSummary.innerHTML = items
      .map(
        (item) =>
          `<span class="diag-pill ${item.value ? "ok-pill" : "bad-pill"}">${item.name}: ${
            item.value ? "OK" : "NO"
          }</span>`
      )
      .join("");
  }

  function currentOrigin() {
    return window.location.origin;
  }

  function isLocalHost(hostname) {
    return hostname === "localhost" || hostname === "127.0.0.1";
  }

  function isAndroidBrowser() {
    return /android/i.test(navigator.userAgent || "");
  }

  function isFirefoxBrowser() {
    const ua = navigator.userAgent || "";
    return /firefox/i.test(ua) && !/seamonkey/i.test(ua);
  }

  function isDesktopBrowser() {
    const ua = navigator.userAgent || "";
    return !/android|iphone|ipad|ipod|mobile/i.test(ua);
  }

  function isChromiumFamilyBrowser() {
    const ua = navigator.userAgent || "";
    if (/firefox|seamonkey/i.test(ua)) return false;
    return /chrome|chromium|edg\/|opr\/|opera|brave/i.test(ua);
  }

  function isEdgeBrowser() {
    return /edg\//i.test(navigator.userAgent || "");
  }

  function configureWizard(kind, origin) {
    wizardStepsAndroid.classList.add("hidden");
    wizardStepsFirefox.classList.add("hidden");
    wizardStepsChromium.classList.add("hidden");
    originCopyRow.classList.add("hidden");
    chromiumOriginRow.classList.add("hidden");
    chromiumFlagRow.classList.add("hidden");
    chromiumNote.classList.add("hidden");

    if (kind === "android_http") {
      wizardTitle.textContent = "Android HTTP Setup Wizard";
      wizardStepsAndroid.classList.remove("hidden");
      originCopyRow.classList.remove("hidden");
      originInput.value = origin;
      wizardNote.textContent =
        "After enabling the flag in Chrome, restart Chrome and tap retry.";
      return;
    }

    if (kind === "chromium_desktop_http") {
      const edge = isEdgeBrowser();
      const scheme = edge ? "edge" : "chrome";
      const browserName = edge ? "Edge" : "Chrome";
      wizardTitle.textContent = `Desktop ${browserName} HTTP Setup Wizard`;
      wizardStepsChromium.classList.remove("hidden");
      chromiumOriginRow.classList.remove("hidden");
      chromiumFlagRow.classList.remove("hidden");
      chromiumNote.classList.remove("hidden");
      chromiumFlagsLink.textContent = `${scheme}://flags/#unsafely-treat-insecure-origin-as-secure`;
      chromiumOriginInput.value = origin;
      chromiumFlagInput.value = `--unsafely-treat-insecure-origin-as-secure="${origin}"`;
      wizardNote.textContent =
        `Preferred: use ${scheme}://flags/#unsafely-treat-insecure-origin-as-secure. Alternative: launch with the copied flag.`;
      return;
    }

    if (kind === "firefox_desktop_http") {
      wizardTitle.textContent = "Firefox Desktop HTTP Setup Wizard";
      wizardStepsFirefox.classList.remove("hidden");
      wizardNote.textContent =
        "These preferences are for local development only.";
      return;
    }

    wizardTitle.textContent = "Camera Access Wizard";
    wizardNote.textContent = "";
  }

  function getDiagnostics() {
    const hasMediaDevices = !!navigator.mediaDevices;
    const hasGetUserMedia =
      (hasMediaDevices && typeof navigator.mediaDevices.getUserMedia === "function") ||
      !!(navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia);
    const secureContext = window.isSecureContext || isLocalHost(window.location.hostname);
    return {
      isAndroid: isAndroidBrowser(),
      isFirefox: isFirefoxBrowser(),
      isChromiumFamily: isChromiumFamilyBrowser(),
      isDesktop: isDesktopBrowser(),
      secureContext,
      hasMediaDevices,
      hasGetUserMedia,
      blockedByHttp: !secureContext && !hasGetUserMedia,
    };
  }

  async function getClientEnv() {
    try {
      const res = await fetch("/api/client/env");
      if (!res.ok) return { origin: currentOrigin() };
      const data = await res.json();
      return { origin: data.origin || currentOrigin() };
    } catch (_) {
      return { origin: currentOrigin() };
    }
  }

  function getDeviceUid() {
    let uid = localStorage.getItem(DEVICE_KEY);
    if (!uid) {
      uid = (crypto?.randomUUID?.() || `device_${Date.now()}_${Math.random()}`).replace(/[^a-zA-Z0-9_-]/g, "");
      localStorage.setItem(DEVICE_KEY, uid);
    }
    return uid;
  }

  async function getMediaStream(constraints) {
    if (navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === "function") {
      return navigator.mediaDevices.getUserMedia(constraints);
    }
    const legacyGetUserMedia =
      navigator.getUserMedia ||
      navigator.webkitGetUserMedia ||
      navigator.mozGetUserMedia ||
      navigator.msGetUserMedia;
    if (legacyGetUserMedia) {
      return new Promise((resolve, reject) => {
        legacyGetUserMedia.call(navigator, constraints, resolve, reject);
      });
    }
    throw new Error("camera_api_unavailable");
  }

  async function registerCamera() {
    if (!validTicket) throw new Error("invite_invalid");
    const payload = {
      ticket_id: ticketId,
      device_uid: getDeviceUid(),
      device_name: navigator.userAgentData?.platform || navigator.platform || "browser",
      platform: /iphone|ipad|ios/i.test(navigator.userAgent)
        ? "ios"
        : /android/i.test(navigator.userAgent)
        ? "android"
        : "web",
      preferred_label: labelInput?.value || null,
    };
    const response = await fetch("/api/cameras/register", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "register_failed");
    }
    const data = await response.json();
    assignedCameraId = data.camera_id;
    if (labelInput?.value) localStorage.setItem(LABEL_KEY, labelInput.value);
    return data;
  }

  function openSocket(registration) {
    ws = new WebSocket(registration.ws_url);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      setState(STATES.STREAMING, assignedCameraId || "connected");
      const fps = Math.max(1, Number(fpsInput.value || 15));
      const quality = Math.max(30, Math.min(95, Number(qualityInput.value || 70))) / 100;
      timer = setInterval(() => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
          if (!blob || !ws || ws.readyState !== WebSocket.OPEN) return;
          blob.arrayBuffer().then((buf) => ws.send(buf));
        }, "image/jpeg", quality);
      }, Math.floor(1000 / fps));
      heart = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "heartbeat", ts: Date.now() }));
        }
      }, 1000);
    };
    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "hint") {
          qualityInput.value = msg.jpeg_quality;
          if (msg.recommended_fps_cap && Number(msg.recommended_fps_cap) > 0) {
            fpsInput.value = String(Number(msg.recommended_fps_cap));
            setState(STATES.STREAMING, `${assignedCameraId || "connected"} | fps hint ${msg.recommended_fps_cap}`);
          }
        }
      } catch (_) {}
    };
    ws.onclose = () => setState(STATES.ERROR, "reconnecting/disconnected");
    ws.onerror = () => setState(STATES.ERROR, "websocket");
  }

  function cleanupStream() {
    if (timer) clearInterval(timer);
    if (heart) clearInterval(heart);
    timer = null;
    heart = null;
    if (ws && ws.readyState <= 1) ws.close();
    ws = null;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
  }

  async function diagnoseAndMaybeWizard() {
    setState(STATES.DIAGNOSE);
    hideHelp();
    const diagnostics = getDiagnostics();
    renderDiag(diagnostics);
    const env = await getClientEnv();

    if (!diagnostics.hasGetUserMedia) {
      if (diagnostics.isAndroid && !diagnostics.secureContext) {
        setState(STATES.BLOCKED_HTTP, "camera blocked over HTTP");
        showWizard();
        configureWizard("android_http", env.origin);
        return false;
      }
      if (
        diagnostics.isChromiumFamily &&
        diagnostics.isDesktop &&
        !diagnostics.secureContext
      ) {
        setState(STATES.BLOCKED_HTTP, "camera blocked over HTTP");
        showWizard();
        configureWizard("chromium_desktop_http", env.origin);
        return false;
      }
      if (diagnostics.isFirefox && diagnostics.isDesktop && !diagnostics.secureContext) {
        setState(STATES.BLOCKED_HTTP, "camera blocked over HTTP");
        showWizard();
        configureWizard("firefox_desktop_http", env.origin);
        return false;
      }
      showWizard();
      configureWizard("generic", env.origin);
      setState(STATES.ERROR, "camera API unavailable");
      showHelp("This browser/device requires HTTPS for camera access.");
      return false;
    }

    hideWizard();
    return true;
  }

  async function startStreaming() {
    setState(STATES.PERMISSION);
    stream = await getMediaStream({ video: { facingMode: "environment" }, audio: false });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    setState(STATES.REGISTERING);
    const registration = await registerCamera();
    openSocket(registration);
  }

  async function startJoinFlow() {
    if (!validTicket) {
      setState(STATES.ERROR, "invite invalid");
      return;
    }
    const canContinue = await diagnoseAndMaybeWizard();
    if (!canContinue) {
      setState(STATES.WIZARD, "follow steps below");
      return;
    }
    try {
      await startStreaming();
    } catch (err) {
      const msg = String(err?.message || err || "");
      if (msg.includes("NotAllowedError")) {
        setState(STATES.ERROR, "camera permission denied");
        showHelp("Allow camera access in Android Chrome site settings and retry.");
      } else {
        setState(STATES.ERROR, msg);
      }
    }
  }

  async function retryAfterWizard() {
    setState(STATES.RETRYING, "checking browser again");
    const ok = await diagnoseAndMaybeWizard();
    if (!ok) {
      setState(STATES.WIZARD, "still blocked");
      return;
    }
    await startJoinFlow();
  }

  labelInput.value = localStorage.getItem(LABEL_KEY) || labelInput.value || "";

  startBtn?.addEventListener("click", async () => {
    await startJoinFlow();
  });

  stopBtn?.addEventListener("click", () => {
    cleanupStream();
    hideWizard();
    hideHelp();
    setState(STATES.IDLE, "stopped");
  });

  retryBtn?.addEventListener("click", async () => {
    await retryAfterWizard();
  });

  copyOriginBtn?.addEventListener("click", async () => {
    if (!originInput.value) return;
    try {
      await navigator.clipboard.writeText(originInput.value);
      wizardNote.textContent = "Origin copied. Paste it into the Chrome flag field.";
    } catch (_) {
      originInput.select();
      wizardNote.textContent = "Copy failed automatically. Please copy the text manually.";
    }
  });

  copyFlagBtn?.addEventListener("click", async () => {
    if (!chromiumFlagInput.value) return;
    try {
      await navigator.clipboard.writeText(chromiumFlagInput.value);
      wizardNote.textContent = "Launch flag copied. Start browser with it, then retry.";
    } catch (_) {
      chromiumFlagInput.select();
      wizardNote.textContent = "Copy failed automatically. Copy the launch flag manually.";
    }
  });

  copyChromiumOriginBtn?.addEventListener("click", async () => {
    if (!chromiumOriginInput.value) return;
    try {
      await navigator.clipboard.writeText(chromiumOriginInput.value);
      wizardNote.textContent = "Origin copied. Paste it into the flags value field.";
    } catch (_) {
      chromiumOriginInput.select();
      wizardNote.textContent = "Copy failed automatically. Copy the origin manually.";
    }
  });

  setState(STATES.IDLE);
})();
