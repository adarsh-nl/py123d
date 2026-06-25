import abc
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import viser

from py123d.api.scene.scene_api import SceneAPI
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ElementContext:
    """Immutable per-scene context shared with all viewer elements.

    A single-agent viewer uses one context (``node_prefix=""``, ``agent_color=None``),
    so every element writes to root-level viser nodes exactly as before. A multi-agent
    viewer builds one context per agent that all share the same world origin
    (``initial_ego_state`` / ``scene_center_array``) but carry a distinct ``node_prefix``
    and ``agent_color``, so agents are recentered into one common world frame and never
    collide on viser scene-tree node names.
    """

    scene: SceneAPI
    initial_ego_state: EgoStateSE3
    num_frames: int
    scene_center_array: npt.NDArray[np.float64]
    dark_mode: bool = False
    node_prefix: str = ""
    """Prefix prepended to every viser node name this element creates (e.g. ``"agent_0/"``)."""
    agent_color: Optional[Tuple[int, int, int]] = None
    """Optional per-agent RGB color (0-255). When set, ego/lidar tint to this color."""
    agent_id: Optional[str] = None
    """Optional human-readable identifier for the agent owning this context."""

    def node(self, name: str) -> str:
        """Namespace a viser node name with this context's ``node_prefix``.

        With the default empty prefix the name is returned unchanged, preserving the
        original single-agent node paths. Any leading slash on ``name`` is dropped before
        prefixing so absolute names (e.g. ``"/map/lane"``) compose correctly.

        :param name: The base (single-agent) viser node name.
        :return: The namespaced node name.
        """
        if not self.node_prefix:
            return name
        return f"{self.node_prefix}{name.lstrip('/')}"

    @staticmethod
    def from_scene(scene: SceneAPI, dark_mode: bool = False) -> "ElementContext":
        """Create an ElementContext from a SceneAPI instance."""
        initial_ego_state = scene.get_ego_state_se3_at_iteration(0)
        if initial_ego_state is None:
            raise ValueError("Scene must have an ego state at iteration 0.")
        scene_center_array = initial_ego_state.center_se3.point_3d.array
        return ElementContext(
            scene=scene,
            initial_ego_state=initial_ego_state,
            num_frames=scene.number_of_iterations,
            scene_center_array=scene_center_array,
            dark_mode=dark_mode,
        )

    @classmethod
    def for_agent(
        cls,
        scene: SceneAPI,
        world_origin: EgoStateSE3,
        num_frames: int,
        node_prefix: str,
        agent_color: Optional[Tuple[int, int, int]] = None,
        agent_id: Optional[str] = None,
        dark_mode: bool = False,
    ) -> "ElementContext":
        """Create an ElementContext for one agent in a multi-agent (collaborative) scene.

        All agents in a collaborative scene must share a single ``world_origin`` so their
        geometry is recentered into the same frame instead of each agent recentering onto
        its own ego (which would stack every agent at the viewer origin).

        :param scene: The agent's own scene.
        :param world_origin: The shared world-origin ego state used to recenter all agents.
        :param num_frames: The number of frames on the unified (collaborative) timeline.
        :param node_prefix: The per-agent viser node-name prefix (e.g. ``"agent_0/"``).
        :param agent_color: Optional per-agent RGB color (0-255).
        :param agent_id: Optional human-readable agent identifier.
        :param dark_mode: Whether the viewer is in dark mode.
        :return: An :class:`ElementContext` bound to ``scene`` but recentered on ``world_origin``.
        """
        return cls(
            scene=scene,
            initial_ego_state=world_origin,
            num_frames=num_frames,
            scene_center_array=world_origin.center_se3.point_3d.array,
            dark_mode=dark_mode,
            node_prefix=node_prefix,
            agent_color=agent_color,
            agent_id=agent_id,
        )


class ViewerElement(abc.ABC):
    """Self-contained visualization element that owns its handles, GUI controls, and data fetching."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name used for the GUI folder."""

    @abc.abstractmethod
    def create_gui(self, server: viser.ViserServer) -> None:
        """Create GUI controls inside the element's own folder. Called once per scene load."""

    @abc.abstractmethod
    def update(self, iteration: int) -> None:
        """Fetch data from SceneAPI and update viser scene handles. Called on every timestep change."""

    @abc.abstractmethod
    def remove(self) -> None:
        """Remove all scene handles and clean up. Called before scene switch."""

    def on_dark_mode_changed(self, dark_mode: bool) -> None:
        """Called when the viewer's dark mode setting changes. Override to adapt colors."""
