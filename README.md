# Mocap Web Portal

## Quickstart
1. Install dependencies:
   - `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
2. Configure token/settings in `configs/default.yaml`.
3. Start:
   - `.\scripts\run_dev.ps1`
4. Open login:
   - `http://<host>:8000/login`

## Guides
- In-app page: `/guides`
- Setup guide doc: `docs/guides/setup.md`
- Calibration guide doc: `docs/guides/calibration.md`

## Main Pages
- `/` Dashboard: invites, session start/stop, checkerboard download.
- `/cameras`: camera roster and status.
- `/calibration`: calibration wizard, readiness, capture/solve, score and tips.
- `/preview`: runtime monitor + 3D preview + snapshots.
- `/recording`: recording controls, wireframe preview, clip list/export.
- `/guides`: setup + calibration checklists.

## Camera Join
1. In Dashboard, create a camera invite.
2. Open `/join?ticket=...` on each phone.
3. Grant camera access and tap `Join + Stream`.

## Calibration Quick Notes
- Download checkerboard PDF from Dashboard and print at 100% scale.
- In `/calibration`, click `Use connected cameras`, then `Start`.
- Capture diverse checkerboard poses until required count.
- Click `Solve` and review score/rating/tips.

## Common Calibration Errors
- `not_all_cameras_ready`:
  - One camera is missing, lagging, or out of sync.
- `checkerboard_not_found:<camera_id>`:
  - That camera does not detect the board.
- `not_enough_captures`:
  - Continue capturing until the required count is reached.

## Browser Camera Access Notes
- Join page includes built-in HTTP camera setup wizards for Android/Chromium/Firefox.
- iOS Safari requires HTTPS for camera access.

## Core Endpoints
- `GET /api/session/status`
- `POST /api/session/start`
- `POST /api/session/stop`
- `POST /api/calibration/start`
- `POST /api/calibration/capture`
- `POST /api/calibration/solve`
- `GET /api/recordings/status`
- `POST /api/recordings/start`
- `POST /api/recordings/stop`
- `GET /api/recordings`
- `POST /api/recordings/{clip_id}/export`
- `POST /api/offline/export`
