# Multi-Agent (Collaborative) Visualization

This branch adds the ability to visualize **multiple agents in one shared world at the same
time** — vehicles, roadside units, and aerial vehicles (drones) — instead of one ego per scene.
It is the foundation for visualizing cooperative-perception datasets such as V2V, V2I, and
Griffin.

## Why this was an extension, not a rewrite

The "one vehicle at a time" limitation was never in the data model. `py123d` already stores ego
poses in a **global frame** (`EgoStateSE3.imu_se3`), exposes a **generic timestamp-based access
API** (`get_modality_at_timestamp(..., criteria="nearest")`), and has a working Viser viewer with
playback. The single-vehicle behavior lived in three places in the **viewer**:

1. The viewer rendered one `SceneAPI` at a time ("Next Scene" cycled, never overlaid).
2. Every Viser node name was hardcoded and global (`"ego_mesh"`, `"lidar_points"`, …) — they
   would collide if two agents drew at once.
3. Everything recentered on a single scene's initial ego.

So the multi-vehicle feature is a viewer-layer change plus a thin "collaborative scene" concept.

## What was added

- **`ElementContext`** (`base_element.py`) gained `node_prefix`, `agent_color`, `agent_id`, a
  `node()` namespacing helper, and a `for_agent(...)` constructor. Defaults preserve the exact
  single-agent behavior (empty prefix → original node names, no color override).
- **Every viewer element** now namespaces its Viser nodes via `self._context.node(...)` so agents
  never collide. Ego and lidar tint to the per-agent color; perceived box detections keep their
  semantic label colors.
- **`MultiAgentScene`** (`multi_agent_scene.py`) — pure, renderer-free logic that wraps a list of
  per-agent scenes, picks one **shared world origin**, and builds a single **unified timeline**
  with per-agent local-iteration mappings. Supports `"timestamp"` alignment (nearest-in-time,
  robust to differing rates/offsets) and `"iteration"` alignment (frame-paired), with automatic
  fallback. Fully unit-tested.
- **`MultiAgentViserViewer`** (`multi_agent_viser_viewer.py`) — renders all agents at once on one
  timeline, one folder of controls per agent, with the HD map rendered once and shared.
- **Example**: `examples/02_multi_agent_overlay.py`.
- **Tests**: `tests/unit/visualization/viser/test_multi_agent_scene.py` (17 cases, no viser
  dependency).

## Usage

```bash
export PY123D_DATA_ROOT=/path/to/py123d_data
python examples/02_multi_agent_overlay.py --log-names log_a log_b log_c
# or
python examples/02_multi_agent_overlay.py --dataset av2 --num-agents 3
```

```python
from py123d.api import SceneFilter, get_filtered_scenes
from py123d.visualization.viser.multi_agent_viser_viewer import launch_multi_agent_viewer

scenes = get_filtered_scenes(SceneFilter(log_names=["log_a", "log_b"]))
launch_multi_agent_viewer(scenes, agent_ids=["ego", "rsu"])
```

A meaningful overlay requires the agents to share a world frame. That holds for a
cooperative-perception dataset (see Phase 2) or for georeferenced logs in the same global frame.

## Verified

- `tests/unit/visualization/viser/test_multi_agent_scene.py` — **17 passed** (timeline alignment,
  world origin, palette, error handling).
- All changed/added modules byte-compile; all element node names confirmed namespaced.
- Live browser rendering was **not** exercised in this environment (no display / no converted
  dataset). Run the example against real data to confirm the visual result.

## Phase 2 — roadmap (cooperative datasets end-to-end)

1. **A cooperative-dataset parser** (e.g. V2V4Real / OPV2V / DAIR-V2X / Griffin) that emits one
   `py123d` log per agent sharing a global frame + map, tagged with a session id and an agent role
   (`vehicle` / `rsu` / `drone`).
2. **Cross-agent alignment** for non-georeferenced agents (lidar/GPS registration → a per-agent
   world transform stored in metadata).
3. **A lightweight data-model concept** (session id / agent id / role) so the loader can group
   agents into a `MultiAgentScene` automatically.
4. **Camera + render controllers** wired through per-agent (currently the 3D overlay covers
   ego/lidar/boxes/cameras/map; the 2D camera panel and offline video export are single-agent).
