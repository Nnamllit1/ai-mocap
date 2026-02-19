(() => {
  if (!('wakeLock' in navigator) || !navigator.wakeLock?.request) {
    return;
  }

  let sentinel = null;
  let shouldKeepAwake = true;

  async function requestWakeLock() {
    if (!shouldKeepAwake || document.visibilityState !== 'visible') {
      return;
    }
    try {
      sentinel = await navigator.wakeLock.request('screen');
      sentinel.addEventListener('release', () => {
        sentinel = null;
        if (shouldKeepAwake && document.visibilityState === 'visible') {
          requestWakeLock();
        }
      });
    } catch (_) {
      // Ignore transient failures (battery saver, OS policy, background tab)
    }
  }

  async function releaseWakeLock() {
    if (!sentinel) {
      return;
    }
    try {
      await sentinel.release();
    } catch (_) {
      // no-op
    }
    sentinel = null;
  }

  function refreshWakeLock() {
    if (document.visibilityState === 'visible') {
      requestWakeLock();
      return;
    }
    releaseWakeLock();
  }

  document.addEventListener('visibilitychange', refreshWakeLock);
  window.addEventListener('pageshow', refreshWakeLock);
  window.addEventListener('focus', refreshWakeLock);

  const userGestureEvents = ['click', 'touchstart', 'keydown'];
  for (const ev of userGestureEvents) {
    window.addEventListener(
      ev,
      () => {
        requestWakeLock();
      },
      { passive: true }
    );
  }

  requestWakeLock();

  window.mocapWakeLock = {
    enable: () => {
      shouldKeepAwake = true;
      requestWakeLock();
    },
    disable: async () => {
      shouldKeepAwake = false;
      await releaseWakeLock();
    },
  };
})();
