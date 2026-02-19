(() => {
  const roster = document.getElementById("camera-roster");
  if (!roster) return;
  const token = roster.dataset.token;

  async function api(path, method = "GET", body = null) {
    const opts = { method, headers: { "x-access-token": token } };
    if (body) {
      opts.headers["content-type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const resp = await fetch(path, opts);
    return await resp.json();
  }

  async function refresh() {
    const data = await api("/api/cameras/roster");
    roster.innerHTML = data
      .map((cam) => {
        const last = cam.last_seen ? new Date(cam.last_seen * 1000).toLocaleTimeString() : "n/a";
        return `
      <div class="camera-card" data-camera-id="${cam.camera_id}">
        <div class="camera-main">
          <strong>${cam.label}</strong>
          <span class="muted">${cam.camera_id}</span>
          <span>${cam.device_name} (${cam.platform})</span>
          <span class="${cam.connected ? "ok" : "warn"}">${cam.connected ? "connected" : "offline"}</span>
          <span class="muted">Last: ${last} | Seq: ${cam.seq || 0}</span>
        </div>
        <div class="camera-actions">
          <input class="label-input" value="${cam.label}" />
          <button data-act="rename">Rename</button>
          <button data-act="toggle">${cam.enabled ? "Disable" : "Enable"}</button>
          <button data-act="remove">Remove</button>
        </div>
      </div>
      `;
      })
      .join("");

    roster.querySelectorAll(".camera-card").forEach((card) => {
      const cameraId = card.dataset.cameraId;
      const labelInput = card.querySelector(".label-input");
      card.querySelector('[data-act="rename"]')?.addEventListener("click", async () => {
        await api(`/api/cameras/${encodeURIComponent(cameraId)}`, "PATCH", {
          label: labelInput.value,
        });
        refresh();
      });
      card.querySelector('[data-act="toggle"]')?.addEventListener("click", async (e) => {
        const enabled = e.target.textContent === "Enable";
        await api(`/api/cameras/${encodeURIComponent(cameraId)}`, "PATCH", { enabled });
        refresh();
      });
      card.querySelector('[data-act="remove"]')?.addEventListener("click", async () => {
        await api(`/api/cameras/${encodeURIComponent(cameraId)}`, "DELETE");
        refresh();
      });
    });
  }

  setInterval(refresh, 1000);
  refresh();
})();
