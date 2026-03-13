import logging
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np

from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.utils import vizard

if TYPE_CHECKING:
    from bsk_rl.sats import Satellite
    from bsk_rl.scene.vessel_targets import TargetWithVessels

logger = logging.getLogger(__name__)


class UniqueVesselPriorityData(Data):
    def __init__(
        self,
        detected: Optional[list[str]] = None,
        imaged: Optional[list[str]] = None,
        duplicates: int = 0,
        known: Optional[list["TargetWithVessels"]] = None,
    ):
        """Track detected vessel IDs and known target objects (with vessels)."""
        if detected is None:
            detected = []
        if imaged is None:
            imaged = []
        self.detected = list(set(detected))
        self.imaged = list(set(imaged))
        self.duplicates = duplicates + len(detected) - len(self.detected)
        self.known = known or []

    def __add__(self, other: "UniqueVesselPriorityData") -> "UniqueVesselPriorityData":
        detected = list(set(self.detected + other.detected))
        imaged = list(set(self.imaged + other.imaged))
        duplicates = (
            self.duplicates
            + other.duplicates
            + len(self.detected)
            + len(other.detected)
            - len(detected)
        )
        known = list(set(self.known + other.known))
        return UniqueVesselPriorityData(detected=detected, imaged=imaged, duplicates=duplicates, known=known)


class UniqueVesselPriorityStore(DataStore):
    data_type = UniqueVesselPriorityData

    def __init__(self, *args, **kwargs) -> None:
        """DataStore for unique vessels.
        """
        super().__init__(*args, **kwargs)

    def get_log_state(self) -> np.ndarray:
        """Read stored data from the satellite's onboard storage."""
        return np.array(
            self.satellite.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData
        )

    def compare_log_states(
        self, old_state: np.ndarray, new_state: np.ndarray
    ) -> UniqueVesselPriorityData:
        """Detect newly added vessel IDs based on changes in onboard data."""
        update_idx = np.where(new_state - old_state > 0)[0]
        detected_vessels = []
        imaged = []

        for idx in update_idx:
            message = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg
            target_id = message.read().storedDataName[int(idx)]
            # Match imaged target with known target
            target = next(t for t in self.data.known if t.id == target_id)
            # All vessels in that target
            detected_vessels.extend(target.vessels)
            imaged.append(target)

        self.update_target_colors(imaged)
        return UniqueVesselPriorityData(detected=detected_vessels, imaged=imaged)

    @vizard.visualize
    def update_target_colors(self, targets, vizInstance=None, vizSupport=None):
        """Update target colors in Vizard."""
        for location in vizInstance.locations:
            if location.stationName in [target.name for target in targets]:
                location.color = vizSupport.toRGBA255(
                    self.satellite.vizard_color)


class UniqueVesselPriorityReward(GlobalReward):
    datastore_type = UniqueVesselPriorityStore  # Handles data and detection

    def __init__(self, reward_fn: Callable[[float], float] = lambda p: p):
        """
        Reward function over vessels. By default, 1 point per unique vessel.
        You can pass a function that weights vessel importance (e.g., risk-based).
        """
        super().__init__()
        self.reward_fn = reward_fn

    def initial_data(self, satellite: "Satellite") -> UniqueVesselPriorityData:
        """Provide full list of known targets (with vessels) to each satellite."""
        return UniqueVesselPriorityData(known=self.scenario.targets)

    def create_data_store(self, satellite: "Satellite") -> None:
        """Attach filter to block imaging of targets with all vessels already detected."""
        super().create_data_store(satellite)

        def vessel_filter(opportunity):
            if opportunity["type"] == "target":
                target = opportunity["object"]
                return any(
                    v not in satellite.data_store.data.detected for v in target.vessels
                )
            return True

        satellite.add_access_filter(vessel_filter)

    def calculate_reward(
        self, new_data_dict: dict[str, UniqueVesselPriorityData]
    ) -> dict[str, float]:
        """Assign reward for each newly detected vessel, once only."""
        reward = {}
        all_new_detected = sum(
            [data.detected for data in new_data_dict.values()], []
        )

        for sat_id, data in new_data_dict.items():
            reward[sat_id] = 0.0
            for vessel_id, target in zip(data.detected, data.imaged):
                if vessel_id not in self.data.detected:
                    reward[sat_id] += 1.0 / \
                        all_new_detected.count(vessel_id) * target.priority
                # else:
                #     print("find one repeat")
        return reward


__doc_title__ = "Unique Vessel with Priority"
__all__ = ["UniqueVesselPriorityReward",
           "UniqueVesselPriorityStore", "UniqueVesselPriorityData"]
