# Mocap Web Portal

## Run
1. Install dependencies:
   - `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
2. Configure token and settings in `configs/default.yaml`.
3. Start:
   - `.\scripts\run_dev.ps1`
4. Open:
   - `http://<host>:8000/login`

## Camera Join
- In Dashboard, create a camera invite.
- Open the generated `/join?ticket=...` link on each phone.
- Grant camera access and tap `Join + Stream`.

## Preview Runtime Monitor
- The `Preview` tab now shows live mocap runtime diagnostics:
  - running state
  - active cameras
  - valid 3D joints
  - loop FPS / loop time
  - dropped cycles
- 3D preview uses COCO-17 joints and simple bone links for easier quality checks.

## Checkerboard PDF
- In Dashboard, use `Download Checkerboard PDF (A4)` in the `Calibration Board` card.
- The PDF uses your current calibration config from `configs/default.yaml`:
  - `calibration.chessboard`
  - `calibration.square_size_m`
- Print with **Actual Size / 100% scale** (disable fit-to-page) for correct calibration dimensions.

## Calibration Tutorial
1. Open `Dashboard` and download the checkerboard PDF.
2. Print at **100% scale** and mount the paper on a flat surface.
3. Join all cameras and confirm they are connected in `Cameras`.
4. Open `Calibration`, click `Use connected cameras`, then `Start`.
5. Wait for readiness to show all cameras green:
   - connected = true
   - in sync = true
   - board = true
6. Move the board through different poses:
   - center, left/right/top/bottom
   - near and far
   - tilted angles
7. Press `Capture` repeatedly until `captures >= required`.
8. Press `Solve` and verify the RMS report.

### Common calibration errors
- `not_all_cameras_ready`:
  - one camera is missing/lagging/out of sync.
  - wait for readiness panel to become fully green before capture.
- `checkerboard_not_found:<camera_id>`:
  - that camera does not detect the board right now.
  - improve lighting, reduce motion blur, keep full board visible.
- `not_enough_captures`:
  - continue capturing until required count is reached.

### Calibration score
- After `Solve`, the Calibration page shows a score and rating (`A` to `E`) with tips.
- The score uses:
  - intrinsic RMS
  - stereo pair RMS
  - distortion sanity checks
- Quick interpretation:
  - `A-B`: good to excellent
  - `C`: usable, but can improve
  - `D-E`: recalibrate before production use

### Tuning tips
- If `not_all_cameras_ready` appears often:
  - reduce phone camera FPS in Join page.
  - keep phones on stable Wi-Fi close to router.
  - calibration now adapts sync latency automatically and may recommend lower FPS.
- For a fast first run:
  - calibrate with 2 cameras first, then expand.
  - temporarily reduce `calibration.min_captures` (e.g. 20 -> 10).

## Android HTTP Workaround
- If camera is blocked on HTTP, the Join page shows a built-in Android wizard.
- In Chrome on Android, open:
  - `chrome://flags`
- Enable the flag, paste the origin shown in the wizard, restart Chrome, then tap retry in the portal.
- iOS Safari still requires HTTPS for camera access.

## Desktop Browser HTTP Workarounds
- Chromium family (Chrome, Edge, Chromium, Brave, Opera):
  - Preferred: open `chrome://flags` (or `edge://flags`) and set
    - `unsafely treat insecure origin as secure`
    - add your portal origin, e.g. `http://YOUR_IP:8000`
  - Alternative: start browser with:
    - `--unsafely-treat-insecure-origin-as-secure="http://YOUR_IP:8000"`
- Firefox Desktop:
  - `about:config`
  - `media.getusermedia.insecure.enabled=true`
  - optionally `media.devices.insecure.enabled=true`

## Core Endpoints
- `GET /api/session/status`
- `POST /api/session/start`
- `POST /api/session/stop`
- `POST /api/calibration/start`
- `POST /api/calibration/capture`
- `POST /api/calibration/solve`
- `POST /api/offline/export`
