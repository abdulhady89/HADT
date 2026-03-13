import gymnasium as gym
from gymnasium.envs.registration import register

from bsk_rl.check_bsk_version import check_bsk_version
from bsk_rl.gym import (
    NO_ACTION,
    ConstellationTasking,
    GeneralSatelliteTasking,
    SatelliteTasking,
    GeneralSatelliteTaskingCloud,
)

__all__ = [
    "GeneralSatelliteTasking",
    "SatelliteTasking",
    "ConstellationTasking",
    "GeneralSatelliteTaskingCloud",
]

register(
    id="GeneralSatelliteTasking-v1",
    entry_point="bsk_rl.gym:GeneralSatelliteTasking",
)

register(
    id="SatelliteTasking-v1",
    entry_point="bsk_rl.gym:SatelliteTasking",
)

register(
    id="ConstellationTasking-v1",
    entry_point="bsk_rl.gym:ConstellationTasking",
)

register(
    id="GeneralSatelliteTaskingCloud-v1",
    entry_point="bsk_rl.gym:GeneralSatelliteTaskingCloud",
)

check_bsk_version()
