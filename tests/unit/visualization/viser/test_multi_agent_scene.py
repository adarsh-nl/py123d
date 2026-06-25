"""Unit tests for the collaborative (multi-agent) scene model.

These cover the pure timeline/alignment logic and the :class:`MultiAgentScene` wrapper, using a
lightweight fake scene so the tests have no viser (or sensor-stack) dependency.
"""

from typing import List, Optional, Sequence

import pytest

from py123d.datatypes.time.timestamp import Timestamp
from py123d.visualization.viser.multi_agent_scene import (
    AGENT_PALETTE,
    MultiAgentScene,
    agent_color,
    compute_iteration_alignment,
    compute_timestamp_alignment,
)


class FakeScene:
    """Minimal stand-in implementing only what MultiAgentScene reads from a SceneAPI."""

    def __init__(self, timestamps_us: Sequence[int], has_ego_at_zero: bool = True):
        self._timestamps = [Timestamp.from_us(int(t)) for t in timestamps_us]
        self._has_ego_at_zero = has_ego_at_zero
        # A sentinel object that doubles as the "ego state at iteration 0".
        self._ego0 = object() if has_ego_at_zero else None

    def get_ego_state_se3_at_iteration(self, iteration: int):
        if iteration == 0:
            return self._ego0
        return object()

    @property
    def number_of_iterations(self) -> int:
        return len(self._timestamps)

    def get_all_ego_state_se3_timestamps(self, include_history: bool = False) -> List[Timestamp]:
        return list(self._timestamps)

    def get_timestamp_at_iteration(self, iteration: int) -> Timestamp:
        return self._timestamps[iteration]


# ----------------------------------------------------------------------------------------------
# Pure alignment helpers
# ----------------------------------------------------------------------------------------------


class TestComputeIterationAlignment:
    def test_equal_lengths_is_identity(self):
        num_iters, per_agent = compute_iteration_alignment([3, 3])
        assert num_iters == 3
        assert per_agent == [[0, 1, 2], [0, 1, 2]]

    def test_shorter_agent_clamps_to_last_frame(self):
        num_iters, per_agent = compute_iteration_alignment([4, 2])
        assert num_iters == 4
        assert per_agent[0] == [0, 1, 2, 3]
        assert per_agent[1] == [0, 1, 1, 1]  # held at last frame once exhausted

    def test_empty(self):
        assert compute_iteration_alignment([]) == (0, [])


class TestComputeTimestampAlignment:
    def test_offset_and_different_rate(self):
        # Agent 0 at t=0,100,200,300; agent 1 offset by 50us: t=50,150,250.
        master = [0, 100, 200, 300]
        per_agent = compute_timestamp_alignment(master, [master, [50, 150, 250]])
        assert per_agent[0] == [0, 1, 2, 3]  # origin maps to itself
        assert per_agent[1] == [0, 0, 1, 2]  # nearest-in-time, ties resolve to the earlier frame

    def test_unsorted_agent_timestamps(self):
        master = [0, 100, 200]
        # Same instants, but provided out of order -> indices must map back to originals.
        per_agent = compute_timestamp_alignment(master, [[200, 0, 100]])
        assert per_agent[0] == [1, 2, 0]

    def test_agent_without_timestamps_maps_to_zero(self):
        per_agent = compute_timestamp_alignment([0, 100], [[]])
        assert per_agent[0] == [0, 0]


class TestAgentColor:
    def test_cycles_palette(self):
        assert agent_color(0) == AGENT_PALETTE[0]
        assert agent_color(len(AGENT_PALETTE)) == AGENT_PALETTE[0]
        assert agent_color(1) == AGENT_PALETTE[1]


# ----------------------------------------------------------------------------------------------
# MultiAgentScene
# ----------------------------------------------------------------------------------------------


class TestMultiAgentScene:
    def test_basic_timestamp_alignment(self):
        scenes = [FakeScene([0, 100, 200, 300]), FakeScene([50, 150, 250])]
        mas = MultiAgentScene(scenes)
        assert mas.num_agents == 2
        assert mas.time_alignment == "timestamp"
        assert mas.num_iterations == 4
        assert mas.agents[0].node_prefix == "agent_0/"
        assert mas.agents[1].node_prefix == "agent_1/"
        assert mas.agents[0].color == AGENT_PALETTE[0]
        assert mas.agents[1].color == AGENT_PALETTE[1]
        assert mas.agents[1].local_iteration(0) == 0
        assert mas.agents[1].local_iteration(3) == 2

    def test_world_origin_is_origin_agent_ego_zero(self):
        scenes = [FakeScene([0, 100]), FakeScene([0, 100])]
        mas = MultiAgentScene(scenes, origin_agent_index=1)
        assert mas.origin_agent_index == 1
        assert mas.world_origin is scenes[1]._ego0
        assert mas.origin_scene is scenes[1]

    def test_iteration_alignment_mode(self):
        scenes = [FakeScene([0, 100, 200, 300]), FakeScene([0, 100])]
        mas = MultiAgentScene(scenes, time_alignment="iteration")
        assert mas.time_alignment == "iteration"
        assert mas.num_iterations == 4
        assert mas.agents[1].local_iteration(3) == 1  # clamped to last frame

    def test_timestamp_falls_back_to_iteration_when_no_ego_timestamps(self):
        scenes = [FakeScene([0, 100]), FakeScene([])]
        mas = MultiAgentScene(scenes, time_alignment="timestamp")
        assert mas.time_alignment == "iteration"

    def test_custom_agent_ids_and_colors(self):
        scenes = [FakeScene([0, 100]), FakeScene([0, 100])]
        mas = MultiAgentScene(scenes, agent_ids=["ego", "rsu"], colors=[(1, 2, 3), (4, 5, 6)])
        assert [a.agent_id for a in mas.agents] == ["ego", "rsu"]
        assert mas.agents[0].color == (1, 2, 3)
        assert mas.agents[1].color == (4, 5, 6)

    def test_timestamp_at_iteration(self):
        scenes = [FakeScene([0, 100, 200, 300]), FakeScene([50, 150, 250])]
        mas = MultiAgentScene(scenes)
        assert mas.timestamp_at_iteration(2).time_us == 200

    def test_empty_scenes_raises(self):
        with pytest.raises(ValueError):
            MultiAgentScene([])

    def test_origin_without_ego_raises(self):
        with pytest.raises(ValueError):
            MultiAgentScene([FakeScene([0, 100], has_ego_at_zero=False)])

    def test_bad_origin_index_raises(self):
        with pytest.raises(ValueError):
            MultiAgentScene([FakeScene([0, 100])], origin_agent_index=5)

    def test_mismatched_agent_ids_raises(self):
        with pytest.raises(ValueError):
            MultiAgentScene([FakeScene([0, 100]), FakeScene([0, 100])], agent_ids=["only_one"])
