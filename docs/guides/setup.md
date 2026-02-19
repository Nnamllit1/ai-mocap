# Setup Guide

Use this checklist for first-time portal setup.

## Steps
1. Install dependencies:
   - `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
2. Configure server token and settings:
   - Edit `configs/default.yaml`
   - Set `server.token` to your chosen value.
3. Start the application:
   - `.\scripts\run_dev.ps1`
4. Open login page:
   - `http://<host>:8000/login`
5. Authenticate with your token.
6. In Dashboard, create camera invites.
7. Open each `/join?ticket=...` URL on phone cameras.
8. Grant camera permission and tap `Join + Stream`.
9. Confirm camera connectivity on `/cameras`.
10. Start session on Dashboard.
11. Validate runtime activity on `/preview` and `/recording`.

## Common Mistakes
- Wrong token in browser vs config:
  - Re-enter token on `/login` and verify `configs/default.yaml`.
- Camera blocked on HTTP:
  - Use the join-page browser wizard instructions.
- Cameras connect intermittently:
  - Reduce phone FPS, improve Wi-Fi stability, and keep phones charged.
