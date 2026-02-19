(() => {
  const root = document.getElementById("config-root");
  if (!root) return;
  const token = root.dataset.token;
  const output = document.getElementById("config-output");

  function num(id) {
    return Number(document.getElementById(id).value);
  }
  function txt(id) {
    return document.getElementById(id).value;
  }

  document.getElementById("config-save")?.addEventListener("click", async () => {
    const payload = {
      model: {
        path: txt("model-path"),
        conf: num("model-conf"),
        iou: num("model-iou"),
        device: txt("model-device"),
      },
      runtime: {
        target_fps: num("runtime-fps"),
        max_latency_ms: num("runtime-latency"),
        ema_alpha: num("runtime-ema"),
      },
      triangulation: {
        min_views: num("tri-min-views"),
        pair_conf_threshold: num("tri-conf"),
        reproj_error_max: num("tri-reproj"),
      },
      osc: {
        host: txt("osc-host"),
        port: num("osc-port"),
      },
    };
    const resp = await fetch("/api/config", {
      method: "PUT",
      headers: { "x-access-token": token, "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    output.textContent = JSON.stringify(data, null, 2);
  });
})();
