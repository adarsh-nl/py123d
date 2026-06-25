"""Multi-agent (collaborative) viser viewer.

Renders several agents — vehicles, roadside units, drones — in one shared 3D world at the same
time, instead of one scene at a time. It reuses the existing per-modality viewer elements; the
only additions over the single-agent :class:`~py123d.visualization.viser.viser_viewer.ViserViewer`
are:

* every agent is recentered onto one shared world origin (so they don't stack at the origin);
* each agent's viser nodes are namespaced under ``agent_<i>/`` (so handles don't collide) and
  tinted with a distinct color; and
* one playback timeline drives all agents, mapping each global frame to the per-agent local
  iteration computed by :class:`~py123d.visualization.viser.multi_agent_scene.MultiAgentScene`.

The HD map (if present) is rendered once, shared by all agents.
"""

from __future__ import annotations

import copy
import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from py123d.api.scene.scene_api import SceneAPI
from py123d.visualization.viser.elements.base_element import ElementContext
from py123d.visualization.viser.elements.box_detections_se3_element import BoxDetectionsSE3Element
from py123d.visualization.viser.elements.camera_frustum_element import CameraFrustumElement
from py123d.visualization.viser.elements.ego_state_se3_element import EgoElement
from py123d.visualization.viser.elements.lidar_element import LidarElement
from py123d.visualization.viser.elements.map_element import MapElement
from py123d.visualization.viser.element_manager import ElementManager
from py123d.visualization.viser.multi_agent_scene import MultiAgentScene
from py123d.visualization.viser.playback_controller import PlaybackController
from py123d.visualization.viser.viser_config import ViserConfig
from py123d.visualization.viser.viser_viewer import HDRI, _build_viser_server

logger = logging.getLogger(__name__)


class MultiAgentViserViewer:
    """Orchestrates a viser viewer that shows multiple agents in one shared world at once."""

    def __init__(
        self,
        multi_agent_scene: MultiAgentScene,
        viser_config: ViserConfig = ViserConfig(),
        show_map: bool = True,
    ) -> None:
        self._mas = multi_agent_scene
        self._config = viser_config
        self._show_map = show_map
        self._server, self._titlebar = _build_viser_server(self._config)
        self._dark_mode = self._config.theme.dark_mode
        self._environment_intensity = 0.25

        self._agent_managers: List[Tuple[object, ElementManager]] = []
        self._map_element: Optional[MapElement] = None

        self._run()

    def _run(self) -> None:
        """Build the collaborative scene and run the (blocking) playback loop, then restart."""
        mas = self._mas
        origin = mas.world_origin
        num_frames = mas.num_iterations
        scene_center_array = np.asarray(origin.center_se3.point_3d.array, dtype=np.float64)

        # One element manager per agent, all recentered on the shared world origin.
        self._agent_managers = []
        for index, agent in enumerate(mas.agents):
            context = ElementContext.for_agent(
                scene=agent.scene,
                world_origin=origin,
                num_frames=num_frames,
                node_prefix=agent.node_prefix,
                agent_color=agent.color,
                agent_id=agent.agent_id,
                dark_mode=self._dark_mode,
            )
            manager = self._build_agent_elements(context)
            self._agent_managers.append((agent, manager))

        # Shared HD map (rendered once, follows the origin agent), if available.
        self._map_element = None
        if self._show_map and mas.origin_scene.get_map_api() is not None:
            map_context = ElementContext(
                scene=mas.origin_scene,
                initial_ego_state=origin,
                num_frames=num_frames,
                scene_center_array=scene_center_array,
                dark_mode=self._dark_mode,
            )
            self._map_element = MapElement(map_context, copy.deepcopy(self._config.map))

        # Single playback timeline shared by all agents.
        playback_context = ElementContext(
            scene=mas.origin_scene,
            initial_ego_state=origin,
            num_frames=num_frames,
            scene_center_array=scene_center_array,
            dark_mode=self._dark_mode,
        )
        playback = PlaybackController(
            self._server,
            self._config.playback,
            playback_context,
            on_dark_mode_changed=self._on_dark_mode_changed,
        )

        # GUI: playback first, then one folder per agent, then the shared map.
        playback.create_gui(mas.origin_scene)
        for index, (agent, manager) in enumerate(self._agent_managers):
            with self._server.gui.add_folder(f"Agent: {agent.agent_id}", expand_by_default=(index == 0)):
                manager.create_all_gui(self._server)
        if self._map_element is not None:
            with self._server.gui.add_folder("Map (shared)", expand_by_default=False):
                self._map_element.create_gui(self._server)

        # Re-apply persisted environment intensity (scene.reset() clears it).
        self._server.scene.configure_environment_map(hdri=HDRI, environment_intensity=self._environment_intensity)

        playback.set_on_iteration_changed(self._on_iteration_changed)
        self._on_iteration_changed(0)

        # Blocking playback loop -- returns on "Next Scene" (treated here as "restart").
        playback.run_loop()

        self._teardown()
        self._server.flush()
        self._server.gui.reset()
        self._server.scene.reset()
        self._run()

    def _build_agent_elements(self, context: ElementContext) -> ElementManager:
        """Register the per-agent elements available for ``context.scene`` (map is shared, not here)."""
        manager = ElementManager()
        scene = context.scene
        if len(scene.get_lidar_metadatas()) > 0:
            manager.register(LidarElement(context, copy.deepcopy(self._config.lidar)))
        if scene.get_box_detections_se3_metadata() is not None:
            manager.register(BoxDetectionsSE3Element(context, copy.deepcopy(self._config.detection)))
        if len(scene.get_camera_metadatas()) > 0:
            manager.register(CameraFrustumElement(context, copy.deepcopy(self._config.camera_frustum)))
        if scene.get_ego_state_se3_metadata() is not None:
            manager.register(EgoElement(context, copy.deepcopy(self._config.ego)))
        return manager

    def _on_iteration_changed(self, global_iteration: int) -> None:
        """Advance every agent (and the shared map) to a global timeline index."""
        for agent, manager in self._agent_managers:
            manager.update_all(agent.local_iteration(global_iteration))
        if self._map_element is not None:
            origin_agent = self._mas.agents[self._mas.origin_agent_index]
            self._map_element.update(origin_agent.local_iteration(global_iteration))

    def _on_dark_mode_changed(self, dark_mode: bool) -> None:
        """Handle the dark-mode toggle from the playback controller."""
        theme = self._config.theme
        self._dark_mode = dark_mode
        self._server.gui.configure_theme(
            titlebar_content=self._titlebar,
            control_layout=theme.control_layout,
            control_width=theme.control_width,
            dark_mode=dark_mode,
            show_logo=theme.show_logo,
            show_share_button=theme.show_share_button,
            brand_color=theme.brand_color,
        )
        for _agent, manager in self._agent_managers:
            manager.notify_dark_mode_changed(dark_mode)
        if self._map_element is not None:
            self._map_element.on_dark_mode_changed(dark_mode)

    def _teardown(self) -> None:
        """Remove all scene handles before a restart."""
        for _agent, manager in self._agent_managers:
            manager.remove_all()
        self._agent_managers = []
        if self._map_element is not None:
            self._map_element.remove()
            self._map_element = None


def launch_multi_agent_viewer(
    scenes: Sequence[SceneAPI],
    viser_config: ViserConfig = ViserConfig(),
    agent_ids: Optional[Sequence[str]] = None,
    time_alignment: str = "timestamp",
    origin_agent_index: int = 0,
    show_map: bool = True,
) -> MultiAgentViserViewer:
    """Convenience entry point: wrap per-agent scenes and launch the collaborative viewer.

    :param scenes: One :class:`~py123d.api.scene.scene_api.SceneAPI` per agent, sharing a world frame.
    :param viser_config: Viewer configuration.
    :param agent_ids: Optional per-agent identifiers.
    :param time_alignment: ``"timestamp"`` (default) or ``"iteration"``.
    :param origin_agent_index: Which agent defines the shared world origin (and the shared map).
    :param show_map: Whether to render the (shared) HD map.
    :return: The running :class:`MultiAgentViserViewer`.
    """
    multi_agent_scene = MultiAgentScene(
        scenes=scenes,
        agent_ids=agent_ids,
        time_alignment=time_alignment,
        origin_agent_index=origin_agent_index,
    )
    logger.info(
        "Launching multi-agent viewer: %d agents, %d frames, alignment=%s.",
        multi_agent_scene.num_agents,
        multi_agent_scene.num_iterations,
        multi_agent_scene.time_alignment,
    )
    return MultiAgentViserViewer(multi_agent_scene, viser_config=viser_config, show_map=show_map)
