# Calibration Guide

Use this checklist for reliable multi-camera calibration.

## Steps
1. Download checkerboard PDF from Dashboard (`/api/calibration/checkerboard.pdf` link).
2. Print at `100%` scale (no fit-to-page).
3. Ensure all cameras are connected and streaming.
4. Open `/calibration`.
5. Click `Use connected cameras`.
6. Click `Start`.
7. Wait for readiness to become fully green:
   - connected = true
   - in sync = true
   - board = true
8. Move checkerboard through diverse poses:
   - center, edges, corners
   - near and far distance
   - tilted orientations
9. Click `Capture` repeatedly until required capture count is reached.
10. Click `Solve`.
11. Check calibration score, rating, and tips.
12. If score is low, recalibrate with better lighting and broader pose diversity.

## Common Mistakes
- `not_all_cameras_ready`:
  - wait for sync, reduce FPS, stabilize network.
- `checkerboard_not_found:<camera_id>`:
  - increase board visibility and lighting, avoid motion blur.
- `not_enough_captures`:
  - continue capture sequence until target is met.
- Inconsistent score between runs:
  - keep camera positions fixed and avoid changing lens zoom/focus mid-session.
