"""Visualize several agents in one shared world at the same time (collaborative perception).

The stock viewer (``py123d-viser``) shows one ego vehicle per scene and cycles between scenes
with "Next Scene". This example instead overlays multiple per-agent scenes in a single 3D world
and plays them back on one synchronized timeline — the building block for visualizing V2V,
vehicle-to-infrastructure, and aerial-vehicle (e.g. Griffin) datasets where several platforms
observe the same scene at once.

Each scene is treated as one agent. All agents are recentered onto a single shared world origin,
each is drawn in its own color under its own ``agent_<i>/`` namespace, and the HD map is rendered
once and shared.

IMPORTANT: a meaningful overlay requires the agents to actually share a world frame. That holds
for a cooperative-perception dataset (a dedicated parser is the Phase-2 step) or for georeferenced
logs whose poses are in the same global frame. Pointing this at unrelated logs will place every
agent at its true world coordinate — which may be kilometers apart.

Usage::

    export PY123D_DATA_ROOT=/path/to/py123d_data
    # Overlay specific logs as agents:
    python examples/02_multi_agent_overlay.py --log-names log_a log_b log_c
    # Or overlay the first N scenes of a dataset:
    python examples/02_multi_agent_overlay.py --dataset av2 --num-agents 3
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

from py123d.api import SceneAPI, SceneFilter, get_filtered_scenes
from py123d.visualization.viser.multi_agent_viser_viewer import launch_multi_agent_viewer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def select_agent_scenes(
    datasets: Optional[List[str]],
    log_names: Optional[List[str]],
    num_agents: Optional[int],
    data_root: Optional[str],
) -> List[SceneAPI]:
    """Resolve the per-agent scenes to overlay from a dataset/log filter.

    :param datasets: Optional dataset names to filter by (e.g. ``["av2"]``).
    :param log_names: Optional explicit log names; each becomes one agent.
    :param num_agents: If set, cap the number of agents (scenes) used.
    :param data_root: Optional py123d data root (else ``PY123D_DATA_ROOT``).
    :return: The selected scenes, one per agent.
    """
    scene_filter = SceneFilter(
        datasets=datasets,
        log_names=log_names,
        required_scene_modalities=["ego_state_se3"],
    )
    scenes: List[SceneAPI] = get_filtered_scenes(scene_filter, data_root=data_root)
    if num_agents is not None:
        scenes = scenes[:num_agents]
    return scenes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", dest="datasets", help="Dataset name to filter by (repeatable).")
    parser.add_argument("--log-names", nargs="+", default=None, help="Explicit log names; each becomes one agent.")
    parser.add_argument("--num-agents", type=int, default=None, help="Cap the number of agents (scenes) overlaid.")
    parser.add_argument("--data-root", default=None, help="py123d data root (defaults to PY123D_DATA_ROOT).")
    parser.add_argument(
        "--time-alignment",
        choices=["timestamp", "iteration"],
        default="timestamp",
        help="Sync agents by nearest absolute time (default) or by shared iteration index.",
    )
    parser.add_argument("--no-map", action="store_true", help="Disable the shared HD map.")
    args = parser.parse_args()

    scenes = select_agent_scenes(args.datasets, args.log_names, args.num_agents, args.data_root)
    if len(scenes) == 0:
        raise SystemExit("No scenes matched the filter. Check PY123D_DATA_ROOT and your --dataset/--log-names.")

    logger.info("Overlaying %d agent(s).", len(scenes))
    if len(scenes) == 1:
        logger.warning("Only one agent selected — this renders like the single-agent viewer.")

    launch_multi_agent_viewer(
        scenes,
        agent_ids=[scene.log_name for scene in scenes],
        time_alignment=args.time_alignment,
        show_map=not args.no_map,
    )
    # Open http://localhost:8080 to view. Ctrl-C to exit.


if __name__ == "__main__":
    main()
