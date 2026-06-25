"""Collaborative (multi-agent) scene model for the viser viewer.

``py123d`` represents one drive of one vehicle as a :class:`~py123d.api.scene.scene_api.SceneAPI`
with a single ego (``ModalityType.EGO_STATE_SE3``). Cooperative-perception datasets (V2V, V2I,
and aerial-vehicle sets such as Griffin) instead capture *several* sensing agents — vehicles,
roadside units, drones — observing the same scene at the same time.

A :class:`MultiAgentScene` is a thin wrapper over a list of per-agent ``SceneAPI`` objects that
already live in a common (georeferenced) world frame. It does two things, both pure data logic
with no rendering dependency:

1. Picks a single shared **world origin** so every agent is recentered into one frame instead of
   each onto its own ego (which would stack all agents at the viewer origin).
2. Builds a single **unified timeline** and, for each agent, the mapping from a global timeline
   index to that agent's local iteration — so agents recorded at different start times or rates
   stay synchronized during playback.

This module deliberately avoids importing viser (or any heavy renderer) so the alignment logic
can be imported and unit-tested on its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # avoid importing the (heavy) scene/ego stack at runtime
    from py123d.api.scene.scene_api import SceneAPI
    from py123d.datatypes.time.timestamp import Timestamp
    from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3

logger = logging.getLogger(__name__)

TimeAlignment = str  # one of: "timestamp", "iteration"

# A perceptually distinct palette used to color agents (vehicles / RSUs / drones).
# RGB, 0-255. Extends by cycling if there are more agents than colors.
AGENT_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (222, 112, 97),   # red
    (105, 156, 219),  # blue
    (176, 230, 133),  # green
    (227, 140, 71),   # orange
    (154, 138, 165),  # purple
    (74, 196, 189),   # cyan
    (255, 196, 0),    # amber
    (214, 122, 177),  # pink
)


def agent_color(index: int) -> Tuple[int, int, int]:
    """Return a stable RGB color (0-255) for the agent at ``index`` (cycles the palette)."""
    return AGENT_PALETTE[index % len(AGENT_PALETTE)]


def compute_iteration_alignment(num_iterations_per_agent: Sequence[int]) -> Tuple[int, List[List[int]]]:
    """Align agents purely by iteration index (frame i of every agent shown together).

    The unified timeline length is the maximum agent length; shorter agents clamp to their last
    iteration so they simply hold their final frame once exhausted. This is the right default for
    cooperative datasets whose agents are sampled synchronously (frame-paired).

    :param num_iterations_per_agent: Iteration count for each agent.
    :return: ``(num_iterations, per_agent_local_iterations)`` where ``per_agent_local_iterations[a][g]``
        is the local iteration of agent ``a`` at global timeline index ``g``.
    """
    if len(num_iterations_per_agent) == 0:
        return 0, []
    counts = [max(0, int(n)) for n in num_iterations_per_agent]
    num_iterations = max(counts) if counts else 0
    per_agent: List[List[int]] = []
    for n in counts:
        last = max(0, n - 1)
        per_agent.append([min(g, last) for g in range(num_iterations)])
    return num_iterations, per_agent


def compute_timestamp_alignment(
    master_timestamps_us: Sequence[int],
    agent_timestamps_us: Sequence[Sequence[int]],
) -> List[List[int]]:
    """Align agents by absolute time: for each master timestamp, pick each agent's nearest frame.

    This handles agents with different capture rates or start offsets — the common case once
    multiple independently-clocked platforms are merged into one world.

    :param master_timestamps_us: The unified timeline's timestamps, in microseconds, ordered.
    :param agent_timestamps_us: Per-agent ordered timestamps (microseconds); entry ``a`` lists
        the timestamp of every local iteration of agent ``a``.
    :return: ``per_agent_local_iterations[a][g]`` — the nearest local iteration of agent ``a`` for
        the master timestamp at global index ``g``. Agents with no timestamps map everything to 0.
    """
    master = np.asarray(list(master_timestamps_us), dtype=np.int64)
    per_agent: List[List[int]] = []
    for agent_ts in agent_timestamps_us:
        agent = np.asarray(list(agent_ts), dtype=np.int64)
        if agent.size == 0 or master.size == 0:
            per_agent.append([0] * int(master.size))
            continue
        # Nearest-neighbor index for each master timestamp via sorted search.
        order = np.argsort(agent, kind="stable")
        agent_sorted = agent[order]
        pos = np.searchsorted(agent_sorted, master)
        pos = np.clip(pos, 1, len(agent_sorted) - 1)
        left = agent_sorted[pos - 1]
        right = agent_sorted[pos]
        choose_left = (master - left) <= (right - master)
        nearest_sorted = np.where(choose_left, pos - 1, pos)
        # Map back to original (unsorted) local iteration indices.
        nearest_local = order[nearest_sorted]
        per_agent.append([int(i) for i in nearest_local])
    return per_agent


@dataclass(frozen=True)
class AgentView:
    """One sensing agent (vehicle, RSU, or drone) in a :class:`MultiAgentScene`."""

    scene: "SceneAPI"
    agent_id: str
    color: Tuple[int, int, int]
    node_prefix: str
    local_iterations: Tuple[int, ...]
    """Mapping from global timeline index to this agent's local iteration."""

    def local_iteration(self, global_iteration: int) -> int:
        """Return this agent's local iteration for a global timeline index (clamped to range)."""
        if len(self.local_iterations) == 0:
            return 0
        gi = max(0, min(int(global_iteration), len(self.local_iterations) - 1))
        return self.local_iterations[gi]


class MultiAgentScene:
    """A collaborative scene: several per-agent scenes sharing one world frame and timeline.

    :param scenes: One :class:`~py123d.api.scene.scene_api.SceneAPI` per agent. They are assumed
        to already share a common (e.g. georeferenced) world frame; alignment of agents that are
        not georeferenced is a separate, upstream registration step.
    :param agent_ids: Optional per-agent identifiers (defaults to ``agent_0``, ``agent_1``, ...).
    :param colors: Optional per-agent RGB colors (0-255); defaults to :data:`AGENT_PALETTE`.
    :param time_alignment: ``"timestamp"`` (nearest-time, robust to differing rates/offsets) or
        ``"iteration"`` (frame-paired). Defaults to ``"timestamp"``, falling back to ``"iteration"``
        if ego timestamps are unavailable.
    :param origin_agent_index: Which agent's initial ego defines the shared world origin.
    """

    def __init__(
        self,
        scenes: Sequence["SceneAPI"],
        agent_ids: Optional[Sequence[str]] = None,
        colors: Optional[Sequence[Tuple[int, int, int]]] = None,
        time_alignment: TimeAlignment = "timestamp",
        origin_agent_index: int = 0,
    ) -> None:
        scenes = list(scenes)
        if len(scenes) == 0:
            raise ValueError("MultiAgentScene requires at least one agent scene.")
        if not (0 <= origin_agent_index < len(scenes)):
            raise ValueError(
                f"origin_agent_index {origin_agent_index} out of range for {len(scenes)} agents."
            )
        if agent_ids is not None and len(agent_ids) != len(scenes):
            raise ValueError("agent_ids must have the same length as scenes.")
        if colors is not None and len(colors) != len(scenes):
            raise ValueError("colors must have the same length as scenes.")

        self._origin_agent_index = origin_agent_index

        world_origin = scenes[origin_agent_index].get_ego_state_se3_at_iteration(0)
        if world_origin is None:
            raise ValueError(
                f"Origin agent (index {origin_agent_index}) must have an ego state at iteration 0."
            )
        self._world_origin: "EgoStateSE3" = world_origin

        ids = list(agent_ids) if agent_ids is not None else [f"agent_{i}" for i in range(len(scenes))]
        cols = list(colors) if colors is not None else [agent_color(i) for i in range(len(scenes))]

        num_iterations, per_agent_local = self._build_alignment(scenes, time_alignment)
        self._time_alignment = self._resolve_alignment_mode(scenes, time_alignment)
        self._num_iterations = num_iterations

        self._agents: List[AgentView] = [
            AgentView(
                scene=scene,
                agent_id=ids[i],
                color=cols[i],
                node_prefix=f"agent_{i}/",
                local_iterations=tuple(per_agent_local[i]),
            )
            for i, scene in enumerate(scenes)
        ]

    # ------------------------------------------------------------------------------------------
    # Timeline construction
    # ------------------------------------------------------------------------------------------

    @staticmethod
    def _resolve_alignment_mode(scenes: Sequence["SceneAPI"], requested: TimeAlignment) -> TimeAlignment:
        if requested == "iteration":
            return "iteration"
        if requested != "timestamp":
            raise ValueError(f"Unknown time_alignment '{requested}'. Use 'timestamp' or 'iteration'.")
        # timestamp requested -> only valid if every agent exposes ego timestamps.
        for scene in scenes:
            if len(scene.get_all_ego_state_se3_timestamps()) == 0:
                logger.warning(
                    "time_alignment='timestamp' requested but an agent has no ego timestamps; "
                    "falling back to iteration alignment."
                )
                return "iteration"
        return "timestamp"

    def _build_alignment(
        self, scenes: Sequence["SceneAPI"], requested: TimeAlignment
    ) -> Tuple[int, List[List[int]]]:
        mode = self._resolve_alignment_mode(scenes, requested)
        if mode == "iteration":
            return compute_iteration_alignment([s.number_of_iterations for s in scenes])

        master = [int(ts) for ts in scenes[self._origin_agent_index].get_all_ego_state_se3_timestamps()]
        agent_ts = [[int(ts) for ts in s.get_all_ego_state_se3_timestamps()] for s in scenes]
        per_agent = compute_timestamp_alignment(master, agent_ts)
        return len(master), per_agent

    # ------------------------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------------------------

    @property
    def agents(self) -> List[AgentView]:
        """The participating agents, in input order."""
        return self._agents

    @property
    def num_agents(self) -> int:
        """Number of agents in the collaborative scene."""
        return len(self._agents)

    @property
    def world_origin(self) -> "EgoStateSE3":
        """The shared world-origin ego state used to recenter all agents."""
        return self._world_origin

    @property
    def origin_agent_index(self) -> int:
        """Index of the agent whose initial ego defines the world origin."""
        return self._origin_agent_index

    @property
    def origin_scene(self) -> "SceneAPI":
        """The scene of the origin agent (also the source of the shared map, if any)."""
        return self._agents[self._origin_agent_index].scene

    @property
    def num_iterations(self) -> int:
        """Length of the unified playback timeline."""
        return self._num_iterations

    @property
    def time_alignment(self) -> TimeAlignment:
        """The alignment mode actually used (``"timestamp"`` or ``"iteration"``)."""
        return self._time_alignment

    def timestamp_at_iteration(self, global_iteration: int) -> "Timestamp":
        """Return the origin agent's timestamp at a global timeline index."""
        origin_agent = self._agents[self._origin_agent_index]
        local = origin_agent.local_iteration(global_iteration)
        return origin_agent.scene.get_timestamp_at_iteration(local)
