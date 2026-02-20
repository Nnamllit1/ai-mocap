# Setup Guide

This guide is for first-time setup on Windows.

## Who This Is For
- You are starting the mocap portal for the first time.
- You want to reach a live view quickly before calibration.
- You are using one Windows PC and one or more phone cameras.

## What You Need Before Starting
- A Windows PC on the same Wi-Fi network as your phones.
- Python installed.
- One or more phones with a modern browser.
- Repo files on your PC.
- Optional now, required later for calibration: printed checkerboard.

## 5-Minute Quick Path
1. Install dependencies.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

2. Check token in `configs/default.yaml`.
   - Default is `server.token: "1234"`.
   - Change it before non-local use.
3. Start server.

```powershell
.\scripts\run_dev.ps1
```

4. Open login page on the PC:
   - `http://localhost:8000/login`
   - You should see: `Portal Token` form.
5. Log in with your Portal token.
   - You should see: Dashboard page.
6. Create camera invite(s) in Dashboard and open each join link on phone.
7. On each phone, allow camera and tap `Join + Stream`.
8. Open `/cameras`.
   - You should see: your phone camera(s) as `connected`.
9. Start session from Dashboard and open `/preview`.
   - You should see: active camera count and moving joints.

## Detailed Step-by-Step

### 1) Install Python dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Expected result:
- Command completes without install errors.

### 2) Configure token and base settings
1. Open `configs/default.yaml`.
2. Check `server.token`.
3. Default token is `1234`.
4. For anything except private local testing, set your own token.

Expected result:
- You know exactly which token to enter on `/login`.

### 3) Start the server

```powershell
.\scripts\run_dev.ps1
```

Expected result:
- Terminal shows Uvicorn running on port `8000`.

### 4) Log in
1. On the PC browser, open `http://localhost:8000/login`.
2. Enter Portal token and submit.

Expected result:
- Dashboard opens (not login page again).

### 5) Connect phone cameras
1. In Dashboard, create an invite.
2. Open invite URL on each phone.
3. Grant camera permission.
4. Tap `Join + Stream`.

Expected result:
- `/cameras` shows each camera as connected and updating.

### 6) Start live tracking
1. In Dashboard, start session.
2. Open `/preview`.

Expected result:
- `active_cameras` is above `0`.
- Joints update when you move.

## Troubleshooting by Symptom

### I can open the site on PC, but not on phone
- Make sure phone and PC are on the same Wi-Fi.
- Use PC LAN IP on phone, not `localhost`.
  - Example: `http://192.168.x.x:8000/login`
- Check Windows firewall allows Python/Uvicorn on private network.

### Login keeps returning to token page
- Token in browser does not match `configs/default.yaml`.
- Re-enter the exact `server.token` value.

### Phone join page cannot access camera
- Accept camera permission in browser settings.
- Use the browser setup hints shown on `/join`.
- On iOS Safari, use HTTPS.

### Invite opened and joined, but stream does not appear
- Verify camera is shown on `/cameras`.
- Re-open join link and tap `Join + Stream` again.
- Keep the phone screen awake and browser tab active.

### I see websocket `403` in logs
- This usually means stale/invalid camera websocket token.
- Re-open join link and let it re-register.
- If needed, create a fresh invite from Dashboard.

## Next Steps
- Calibration guide: `docs/guides/calibration.md`
- Blender live sync guide: `docs/guides/blender_live_sync.md`
