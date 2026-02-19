(() => {
  const root = document.getElementById("session-status");
  const token = root?.dataset.token || "";
  const startBtn = document.getElementById("start-session");
  const stopBtn = document.getElementById("stop-session");
  const inviteBtn = document.getElementById("invite-create");
  const inviteOutput = document.getElementById("invite-output");
  const rosterRoot = document.getElementById("roster-list");

  async function req(path, method = "GET", body = null) {
    const opts = { method, headers: { "x-access-token": token } };
    if (body) {
      opts.headers["content-type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const resp = await fetch(path, opts);
    return await resp.json();
  }

  function render(status) {
    root.textContent = JSON.stringify(status, null, 2);
  }

  function renderRoster(items) {
    if (!rosterRoot) return;
    if (!items || items.length === 0) {
      rosterRoot.innerHTML = `<p class="muted">No cameras registered yet.</p>`;
      return;
    }
    rosterRoot.innerHTML = items
      .map(
        (cam) => `
      <div class="roster-card">
        <strong>${cam.label}</strong>
        <span class="muted">${cam.camera_id}</span>
        <span class="${cam.connected ? "ok" : "warn"}">${cam.connected ? "connected" : "idle"}</span>
      </div>
    `
      )
      .join("");
  }

  async function refresh() {
    const [status, roster] = await Promise.all([
      req("/api/session/status"),
      req("/api/cameras/roster"),
    ]);
    render(status);
    renderRoster(roster);
  }

  startBtn?.addEventListener("click", async () => {
    await req("/api/session/start", "POST");
    await refresh();
  });

  stopBtn?.addEventListener("click", async () => {
    await req("/api/session/stop", "POST");
    await refresh();
  });

  inviteBtn?.addEventListener("click", async () => {
    const ttl = Number(document.getElementById("invite-ttl")?.value || 120);
    const presetLabel = document.getElementById("invite-label")?.value || "";
    const data = await req("/api/cameras/invites", "POST", {
      ttl_s: ttl,
      preset_label: presetLabel || null,
    });
    const link = data.join_url || "";
    const qrSrc = `https://api.qrserver.com/v1/create-qr-code/?size=260x260&data=${encodeURIComponent(link)}`;
    inviteOutput.innerHTML = `
      <p><strong>Invite Code:</strong> <code>${data.ticket_id}</code></p>
      <p><a href="${link}">${link}</a></p>
      <img class="invite-qr" src="${qrSrc}" alt="Invite QR code" />
      <p class="muted">Expires at: ${new Date((data.expires_at || 0) * 1000).toLocaleTimeString()}</p>
    `;
  });

  setInterval(refresh, 1000);
  refresh();
})();
