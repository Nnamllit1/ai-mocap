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
  function checked(id) {
    return Boolean(document.getElementById(id)?.checked);
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
        missing_joint_hold_ms: num("runtime-missing-hold"),
        max_joint_jump_m: num("runtime-max-jump"),
        jump_reject_conf: num("runtime-jump-conf"),
        bone_length_guard_enabled: checked("runtime-bone-guard-enabled"),
        bone_length_soft_rel_tol: num("runtime-bone-soft-tol"),
        bone_length_hard_rel_tol: num("runtime-bone-hard-tol"),
        bone_length_ema_alpha: num("runtime-bone-ema"),
        bone_length_learn_conf: num("runtime-bone-learn-conf"),
      },
      triangulation: {
        min_views: num("tri-min-views"),
        pair_conf_threshold: num("tri-conf"),
        reproj_error_max: num("tri-reproj"),
        allow_single_view_fallback: checked("tri-single-fallback"),
        single_view_conf_scale: num("tri-single-scale"),
        single_view_max_age_ms: num("tri-single-age"),
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
