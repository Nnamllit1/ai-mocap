# Blender Live Sync Guide

Use this guide to connect the runtime to Blender and preview a live skeleton.

## Install Add-on
1. Open Blender 3.6 or newer.
2. Go to `Edit > Preferences > Add-ons`.
3. Click `Install...` and choose:
   - `integrations/blender/mocap_live_sync_addon.py`
4. Enable the add-on entry `Mocap Live Sync (OSC)`.

## Configure Runtime OSC Target
Update `configs/default.yaml` so OSC packets are sent to Blender:

- `osc.host`: Blender machine IP, or `127.0.0.1` if server and Blender run on the same PC.
- `osc.port`: add-on listen port (default `9000`).
- `osc.address_prefix`: add-on prefix (default `/mocap`).

## Start Live Sync
1. Start server and camera streaming as usual.
2. In Blender, open `View3D > Sidebar > Mocap Live`.
3. Set:
   - Listen port
   - OSC prefix
   - Axis mapping preset
   - Confidence threshold
4. Click `Create Skeleton`.
5. Click `Start Sync`.
6. Start a tracking session in the web app. The skeleton should update live.

## Notes
- If confidence is below the threshold, the add-on holds the last good joint position.
- Use axis mapping presets to correct orientation without restarting either app.
- `Reset Live State` clears live cache only; it does not delete created armature/empties.

## Troubleshooting
- No movement in Blender:
  - Check `osc.host`, `osc.port`, and `osc.address_prefix`.
  - Confirm the add-on panel shows `Listening: yes`.
  - Check Windows firewall rules for UDP on the selected port.
- Wrong orientation:
  - Change the axis mapping preset in the panel.
- Packets arrive but rig does not exist:
  - Click `Create Skeleton` again to rebuild the armature and constraints.
- Intermittent updates:
  - Lower `Packets/Tick` only if UI stalls.
  - Keep polling interval near `0.02s` for normal live preview.
