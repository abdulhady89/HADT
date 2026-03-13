"""``bsk_rl.scene`` provides scenarios, or the underlying environment in which the satellite can collect data.

Scenarios typically correspond to certain type(s) of :ref:`bsk_rl.data` systems. The
following scenarios have been implemented:

* :class:`UniformTargets`: Uniformly distributed targets to be imaged by an :class:`~bsk_rl.sats.ImagingSatellite`.
* :class:`CityTargets`: Targets distributed near population centers.
* :class:`UniformNadirScanning`: Uniformly desireable data over the surface of the Earth.
"""

from bsk_rl.scene.scenario import Scenario, UniformNadirScanning
from bsk_rl.scene.targets import CityTargets, UniformTargets
from bsk_rl.scene.user_targets import UserDefinedTargets, OceanTargets, UserDefOceanTargetswithCloud, RandomOceanTargetswithCloud, RandomOrbitalTargetswithCloud

# vessel
from bsk_rl.scene.vessel_targets import UniformTargetsWithVessels
from bsk_rl.scene.vessel_targets import UniformTargetsWithRandomVessels
from bsk_rl.scene.vessel_targets import OceanRandomVesselsTargets
from bsk_rl.scene.vessel_targets import OceanTargetsWithFixedVessels
from bsk_rl.scene.vessel_targets import UniformRandomOceanVesselsTargets

__doc_title__ = "Scenario"
__all__ = [
    "Scenario",
    "UniformTargets",
    "CityTargets",
    "UniformNadirScanning",
    "UserDefinedTargets",
    "OceanTargets",
    "UniformTargetsWithVessels",
    "UniformTargetsWithRandomVessels",
    "OceanTargetsWithFixedVessels",
    "OceanRandomVesselsTargets",
    "UniformRandomOceanVesselsTargets",
    "UserDefOceanTargetswithCloud",
    "RandomOceanTargetswithCloud",
    "RandomOrbitalTargetswithCloud"
]
