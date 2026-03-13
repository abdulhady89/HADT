import logging
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np

from bsk_rl.data.base import Data, DataStore, GlobalReward, GlobalImgSARReward
from bsk_rl.utils import vizard
from bsk_rl.data.unique_image_data import UniqueImageStore, UniqueImageData

if TYPE_CHECKING:
    from bsk_rl.sats import Satellite
    from bsk_rl.scene.vessel_targets import TargetWithVessels

logger = logging.getLogger(__name__)


class UniqueImageSARReward(GlobalReward):
    """GlobalReward for rewarding unique images."""

    datastore_type = UniqueImageStore

    def __init__(
        self,
        reward_fn: Callable = lambda p: p,
    ) -> None:
        """GlobalReward for rewarding unique images.

        This data system should be used with the :class:`~bsk_rl.sats.ImagingSatellite` and
        a scenario that generates targets, such as :class:`~bsk_rl.scene.UniformTargets` or
        :class:`~bsk_rl.scene.CityTargets`.

        The satellites all start with complete knowledge of the targets in the scenario.
        Each target can only give one satellite a reward once; if any satellite has imaged
        a target, reward will never again be given for that target. The satellites filter
        known imaged targets from consideration for imaging to prevent duplicates.
        Communication can transmit information about what targets have been imaged in order
        to prevent reimaging.


        Args:
            scenario: GlobalReward.scenario
            reward_fn: Reward as function of priority.
        """
        super().__init__()
        self.reward_fn = reward_fn

    def initial_data(self, satellite: "Satellite") -> "UniqueImageData":
        """Furnish data to the scenario.

        Currently, it is assumed that all targets are known a priori, so the initial data
        given to the data store is the list of all targets.
        """
        return self.data_type(known=self.scenario.targets)

    def create_data_store(self, satellite: "Satellite") -> None:
        """Override the access filter in addition to creating the data store."""
        super().create_data_store(satellite)

        def unique_target_filter(opportunity):
            if opportunity["type"] == "target":
                return opportunity["object"] not in satellite.data_store.data.imaged
            return True

        satellite.add_access_filter(unique_target_filter)

    def calculate_reward(
        self, new_data_dict: dict[str, UniqueImageData]
    ) -> dict[str, float]:
        """Reward each new unique image once.

        Reward is evaluated based on ``self.reward_fn(target.priority)``.

        Args:
            new_data_dict: Record of new images for each satellite

        Returns:
            reward: Cumulative reward across satellites for one step
        """
        reward = {}
        imaged_targets = sum(
            [new_data.imaged for new_data in new_data_dict.values()], []
        )
        for sat_id, new_data in new_data_dict.items():
            reward[sat_id] = 0.0
            for target in new_data.imaged:
                if target not in self.data.imaged:
                    if "SAR" in sat_id:
                        if target.cloud_cover_forecast < 0.5:
                            reward[sat_id] += (-1 + target.cloud_cover_forecast) + target.priority / \
                                imaged_targets.count(target)
                        else:
                            reward[sat_id] += (target.cloud_cover_forecast) + target.priority / \
                                imaged_targets.count(target)
                    else:
                        if target.cloud_cover_forecast > 0.5:
                            reward[sat_id] += -1 * target.cloud_cover_forecast + target.priority / \
                                imaged_targets.count(target)
                        else:
                            reward[sat_id] += (1-target.cloud_cover_forecast) + target.priority / \
                                imaged_targets.count(target)

        return reward


__doc_title__ = "Unique Image and SAR with Priority"
__all__ = ["UniqueImageSARReward"]
