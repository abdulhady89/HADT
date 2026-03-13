"""Target scenarios distribute ground targets with some distribution.

Currently, targets are all known to the satellites a priori and are available based on
the imaging requirements given by the dynamics and flight software models.
"""

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd
import random
from Basilisk.utilities import orbitalMotion

from bsk_rl.scene import Scenario
from bsk_rl.utils import vizard
from bsk_rl.utils.orbital import lla2ecef

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.data.base import Data
    from bsk_rl.sats import Satellite

logger = logging.getLogger(__name__)


class Target:
    """Ground target with associated value."""

    def __init__(self, name: str, r_LP_P: Iterable[float], priority: float) -> None:
        """Ground target with associated priority and location.

        Args:
            name: Identifier; does not need to be unique
            r_LP_P: Planet-fixed, planet relative location [m]
            priority: Value metric.
        """
        self.name = name
        self.r_LP_P = np.array(r_LP_P)
        self.priority = priority

    @property
    def id(self) -> str:
        """Get unique, human-readable identifier."""
        try:
            return self._id
        except AttributeError:
            self._id = f"{self.name}_{id(self)}"
            return self._id

    def __hash__(self) -> int:
        """Hash target by unique id."""
        return hash((self.id))

    def __repr__(self) -> str:
        """Get string representation of target.

        Use ``target.id`` for a unique string identifier.

        Returns:
            Target string
        """
        return f"Target({self.name})"


class UniformTargets(Scenario):
    """Environment with targets distributed uniformly."""

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
    ) -> None:
        """An environment with evenly-distributed static targets.

        Can be used with :class:`~bsk_rl.data.UniqueImageReward`.

        Args:
            n_targets: Number of targets to generate. Can also be specified as a range
                ``(low, high)`` where the number of targets generated is uniformly selected
                ``low ≤ n_targets ≤ high``.
            priority_distribution: Function for generating target priority. Defaults
                to ``lambda: uniform(0, 1)`` if not specified.
            radius: [m] Radius to place targets from body center. Defaults to Earth's
                equatorial radius.
        """
        self._n_targets = n_targets
        if priority_distribution is None:
            def priority_distribution(): return np.random.rand()  # noqa: E731
        self.priority_distribution = priority_distribution
        self.radius = radius

    def reset_overwrite_previous(self) -> None:
        """Overwrite target list from previous episode."""
        self.targets = []

    def reset_pre_sim_init(self) -> None:
        """Regenerate target set for new episode."""
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(
                self._n_targets[0], self._n_targets[1])
        logger.info(f"Generating {self.n_targets} targets")
        self.regenerate_targets()
        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for target in self.targets:
                    satellite.add_location_for_access_checking(
                        object=target,
                        r_LP_P=target.r_LP_P,
                        min_elev=satellite.sat_args_generator[
                            "imageTargetMinimumElevation"
                        ],  # Assume not randomized
                        type="target",
                    )

    def reset_during_sim_init(self) -> None:
        """Visualize targets in Vizard on reset."""
        for target in self.targets:
            self.visualize_target(target)
        for groundStation in self.satellites[0].locations_for_access_checking:
            if groundStation['type'] == 'ground_station':
                self.visualize_groundStation(groundStation)

    @vizard.visualize
    def visualize_target(self, target, vizSupport=None, vizInstance=None):
        """Visualize target in Vizard."""
        vizSupport.addLocation(
            vizInstance,
            stationName=target.name,
            parentBodyName="earth",
            r_GP_P=list(target.r_LP_P),
            fieldOfView=np.arctan(500 / 800),
            color=vizSupport.toRGBA255("white"),
            range=1000.0 * 1000,  # meters
        )
        if vizInstance.settings.showLocationCones == 0:
            vizInstance.settings.showLocationCones = -1
        if vizInstance.settings.showLocationCommLines == 0:
            vizInstance.settings.showLocationCommLines = -1
        if vizInstance.settings.showLocationLabels == 0:
            vizInstance.settings.showLocationLabels = -1

    @vizard.visualize
    def visualize_groundStation(self, target, vizSupport=None, vizInstance=None):
        """Visualize ground station in Vizard."""
        vizSupport.addLocation(
            vizInstance,
            stationName=f"{target['ground_station']}".replace(
                "GroundStation", ""),
            parentBodyName="earth",
            r_GP_P=list(target['r_LP_P']),
            fieldOfView=np.arctan(500 / 800),
            color=vizSupport.toRGBA255("green"),
            range=1000.0 * 1000,  # meters
        )
        if vizInstance.settings.showLocationCones == 0:
            vizInstance.settings.showLocationCones = -1
        if vizInstance.settings.showLocationCommLines == 0:
            vizInstance.settings.showLocationCommLines = -1
        if vizInstance.settings.showLocationLabels == 0:
            vizInstance.settings.showLocationLabels = -1

    def regenerate_targets(self) -> None:
        """Regenerate targets uniformly.

        Override this method (as demonstrated in :class:`CityTargets`) to generate
        other distributions.
        """
        self.targets = []
        for i in range(self.n_targets):
            x = np.random.normal(size=3)
            x *= self.radius / np.linalg.norm(x)
            self.targets.append(
                Target(name=f"tgt-{i}", r_LP_P=x,
                       priority=self.priority_distribution())
            )


class UserDefinedTargets(UniformTargets):
    """Environment with user defined targets."""

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        n_select_from: Optional[int] = None,
        location_offset: float = 200000,
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
    ) -> None:
        """Construct environment with static targets around population centers.

        Uses the `simplemaps Word Cities Database <https://simplemaps.com/data/world-cities>`_
        for population center locations. This data is installed by ``finish_install``.

        Args:
            n_targets: Number of targets to generate, as a fixed number or a range.
            n_select_from: Generate targets from the top `n_select_from` most populous
                cities. Will use all cities in the database if not specified.
            location_offset: [m] Offset targets randomly from the city center by up to
                this amount.
            priority_distribution: Function for generating target priority.
            radius: Radius to place targets from body center.
        """
        super().__init__(n_targets, priority_distribution, radius)
        if n_select_from == "all" or n_select_from is None:
            n_select_from = sys.maxsize
        self.n_select_from = n_select_from
        self.location_offset = location_offset

    def regenerate_targets(self) -> None:
        """Regenerate targets based on cities.

        :meta private:
        """
        self.targets = []
        cities = pd.read_csv(
            Path(os.path.realpath(__file__)).parent.parent
            / "_dat"
            / "simplemaps_worldcities"
            / "worldcities.csv",
        )

        # select only 5 cities in AU closest to bushfire location
        cityList = ['Emerald', 'Narrabri', 'Wallan', 'Gawler', 'Northam']
        tgtPriority = [1, 2, 4, 3, 4]
        maxPriority = max(tgtPriority)

        for cityName, priority_level in zip(cityList, tgtPriority):
            for i in range(5):
                city = cities.loc[cities['city'] == cityName]
                location = lla2ecef(city["lat"], city["lng"], self.radius)
                offset = np.random.normal(size=3)
                offset /= np.linalg.norm(offset)
                offset *= self.location_offset
                location = location.reshape(-1)
                location += offset
                location /= np.linalg.norm(location)
                location *= self.radius
                self.targets.append(
                    Target(
                        name=f"{city['city'].values.item()}-{priority_level}".replace("'", ""),
                        r_LP_P=location,
                        priority=priority_level/maxPriority,
                    )
                )


class OceanTargets(UniformTargets):
    """Environment with user defined ocean targets."""

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        n_select_from: Optional[int] = None,
        location_offset: float = 50000,
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
    ) -> None:
        """Construct environment with static targets around population centers.

        Uses the `simplemaps Word Cities Database <https://simplemaps.com/data/world-cities>`_
        for population center locations. This data is installed by ``finish_install``.

        Args:
            n_targets: Number of targets to generate, as a fixed number or a range.
            n_select_from: Generate targets from the top `n_select_from` most populous
                cities. Will use all cities in the database if not specified.
            location_offset: [m] Offset targets randomly from the city center by up to
                this amount.
            priority_distribution: Function for generating target priority.
            radius: Radius to place targets from body center.
        """
        super().__init__(n_targets, priority_distribution, radius)
        if n_select_from == "all" or n_select_from is None:
            n_select_from = sys.maxsize
        self.n_select_from = n_select_from
        self.location_offset = location_offset

    def regenerate_targets(self) -> None:
        """Regenerate targets based on cities.

        :meta private:
        """
        self.targets = []
        oceans = pd.read_csv(
            Path(os.path.realpath(__file__)).parent.parent
            / "_dat"
            / "ocean"
            / "ocean.csv",
        )

        # select only 5 cities in AU closest to bushfire location
        ocean_list = oceans['ocean']
        tgtPriority = oceans['priority']
        maxPriority = max(tgtPriority)

        for oceanName, priority_level in zip(ocean_list, tgtPriority):
            for i in range(3):
                ocean = oceans.loc[oceans['ocean'] == oceanName]
                location = lla2ecef(ocean["lat"], ocean["lng"], self.radius)
                offset = np.random.normal(size=3)
                offset /= np.linalg.norm(offset)
                offset *= self.location_offset
                location = location.reshape(-1)
                location += offset
                location /= np.linalg.norm(location)
                location *= self.radius
                self.targets.append(
                    Target(
                        name=f"{ocean['ocean'].values.item()}-{priority_level}".replace("'", ""),
                        r_LP_P=location,
                        priority=priority_level/maxPriority,
                    )
                )


# A new target with cloud definition
class TargetwithCloud:
    """Ground target with associated value."""

    def __init__(self,
                 name: str,
                 r_LP_P: Iterable[float],
                 priority: float,
                 cloud_cover_true: float,
                 cloud_cover_sigma: float,
                 cloud_cover_forecast: float
                 ) -> None:
        """Ground target with associated priority and location.

        Args:
            name: Identifier; does not need to be unique
            r_LP_P: Planet-fixed, planet relative location [m]
            priority: Value metric.
        """
        self.name = name
        self.r_LP_P = np.array(r_LP_P)
        self.priority = priority
        self.cloud_cover_true = cloud_cover_true
        self.cloud_cover_sigma = cloud_cover_sigma
        self.cloud_cover_forecast = cloud_cover_forecast

    @property
    def id(self) -> str:
        """Get unique, human-readable identifier."""
        try:
            return self._id
        except AttributeError:
            self._id = f"{self.name}_{id(self)}"
            return self._id

    def __hash__(self) -> int:
        """Hash target by unique id."""
        return hash((self.id))

    def __repr__(self) -> str:
        """Get string representation of target.

        Use ``target.id`` for a unique string identifier.

        Returns:
            Target string
        """
        return f"Target({self.name})"

# A new target with cloud scenario
class UserDefOceanTargetswithCloud(Scenario):
    """Environment with user defined ocean targets with cloud coverage."""
    # mu_data = 0.6740208166434426  # Average global cloud coverage

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        location_offset: float = 30000, #30000,
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
        sigma_levels: tuple[float, float] = (0.01, 0.05),
        reward_thresholds: Union[float, tuple[float, float]] = 0.95,
    ) -> None:
        """Construct environment with static targets around sea area with cloud
        coverage predefined with refer to a .csv file in _dat folder.
        """
        self._n_targets = n_targets
        self.location_offset = location_offset
        self.reward_thresholds = reward_thresholds
        self.sigma_levels = sigma_levels
        self.radius = radius

    def reset_overwrite_previous(self) -> None:
        self.targets = []

    def reset_pre_sim_init(self) -> None:
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(
                self._n_targets[0], self._n_targets[1])

        logger.info(f"Generating {self.n_targets} targets")
        self.regenerate_targets()

        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for target in self.targets:
                    satellite.add_location_for_access_checking(
                        object=target,
                        r_LP_P=target.r_LP_P,
                        min_elev=np.radians(83), # was satellite.sat_args_generator["imageTargetMinimumElevation"]
                        type="target",
                    )

    def reset_during_sim_init(self) -> None:
        for target in self.targets:
            self.visualize_target(target)

        for groundStation in self.satellites[0].locations_for_access_checking:
            if groundStation['type'] == 'ground_station':
                self.visualize_groundStation(groundStation)

    def regenerate_targets(self) -> None:
        self.targets = []
        oceans = pd.read_csv(
            Path(os.path.realpath(__file__)).parent.parent
            / "_dat"
            / "ocean"
            / "CloudSARAoIs_diffPriority.csv",
            # / "CloudSARAoIs_diffPriority_mip.csv",
            # / "northwestAUoceanCloud_SAR_new.csv",
        )

        # select ocean column
        ocean_list = oceans['ocean']
        tgtPriority = oceans['priority']
        maxPriority = max(tgtPriority)
        tgtCloud_cover = oceans['cloud']
        bearing_deg = oceans['bearing']

        n_ocean = 0
        for oceanName, priority_level, cloud_cover, bearing in zip(ocean_list, tgtPriority, tgtCloud_cover, bearing_deg):
            ocean = oceans.loc[oceans['ocean'] == oceanName]
            ocean_grids = self.generate_grid_coordinates(
                ocean['lat'], ocean['lng'], spacing=self.location_offset, rows=3, cols=15, bearing_deg=bearing)

            i = 0
            for aoi_coords in ocean_grids:
                location = lla2ecef(
                    aoi_coords["lat"], aoi_coords["lng"], self.radius)
                location = location.reshape(-1)
                location /= np.linalg.norm(location)
                location *= self.radius
                cloud_cover_sigma = np.random.uniform(
                    self.sigma_levels[0], self.sigma_levels[1])
                cloud_cover_forecast = np.clip(np.random.normal(
                    cloud_cover, cloud_cover_sigma), 0.0, 1.0)

                self.targets.append(
                    TargetwithCloud(
                        name=f"AoI-{n_ocean}.{i}-[{cloud_cover_forecast}]-[{priority_level}]",
                        r_LP_P=location,
                        priority=priority_level/maxPriority,
                        cloud_cover_true=cloud_cover,
                        cloud_cover_sigma=cloud_cover_sigma,
                        cloud_cover_forecast=cloud_cover_forecast
                    )
                )
                i += 1
            n_ocean += 1

    @vizard.visualize
    def visualize_target(self, target, vizSupport=None, vizInstance=None):
        vizSupport.addLocation(
            vizInstance,
            stationName=target.name,
            parentBodyName="earth",
            r_GP_P=list(target.r_LP_P),
            fieldOfView=np.arctan(25 / 500),  # np.arctan(500 / 800),
            color=vizSupport.toRGBA255("white"),
            range=1000.0 * 1000,
        )
        if vizInstance.settings.showLocationCones == 0:
            vizInstance.settings.showLocationCones = -1
        if vizInstance.settings.showLocationCommLines == 0:
            vizInstance.settings.showLocationCommLines = -1
        if vizInstance.settings.showLocationLabels == 0:
            vizInstance.settings.showLocationLabels = -1

    @vizard.visualize
    def visualize_groundStation(self, target, vizSupport=None, vizInstance=None):
        """Visualize ground station in Vizard."""
        vizSupport.addLocation(
            vizInstance,
            stationName=f"{target['ground_station']}".replace(
                "GroundStation", ""),
            parentBodyName="earth",
            r_GP_P=list(target['r_LP_P']),
            fieldOfView=np.arctan(target['min_elev']),
            color=vizSupport.toRGBA255("green"),
            range=1000.0 * 1000,  # meters
        )

    def generate_grid_coordinates(self, center_lat, center_lon,
                                          spacing, rows, cols,
                                          bearing_deg):
        R = self.radius  # Earth radius

        # Convert center to radians
        lat0 = np.radians(center_lat)
        lon0 = np.radians(center_lon)

        # Use orbit inclination as the rotation angle
        theta = np.radians(bearing_deg)

        # Compute coordinate grid offsets in km from the center
        row_offsets = np.arange(-(rows // 2), (rows // 2) + 1)
        col_offsets = np.arange(-(cols // 2), (cols // 2) + 1)

        grid_coords = []

        for i in row_offsets:
            for j in col_offsets:
                # Offset in local East (x) and North (y) directions
                dx = j * spacing
                dy = -i * spacing  # minus to match geographic north-up convention

                # Rotate by orbit inclination
                x_rot = dx * np.cos(theta) - dy * np.sin(theta)
                y_rot = dx * np.sin(theta) + dy * np.cos(theta)

                # Convert km shifts to angular shifts
                dlat = y_rot / R
                dlon = x_rot / (R * np.cos(lat0))

                lat = lat0 + dlat
                lon = lon0 + dlon

                coords = {}
                coords['lat'] = np.degrees(lat)
                coords['lng'] = np.degrees(lon)

                grid_coords.append(coords)

        return grid_coords


class RandomOceanTargetswithCloud(Scenario):
    """Environment with randomised ocean targets with cloud coverage."""
    # mu_data = 0.6740208166434426  # Average global cloud coverage

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        location_offset: float = 30000,
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
        sigma_levels: tuple[float, float] = (0.01, 0.05),
        reward_thresholds: Union[float, tuple[float, float]] = 0.95,
    ) -> None:
        """Construct environment with static targets around sea area with cloud
        coverage predefined with refer to a .csv file in _dat folder.
        """
        self._n_targets = n_targets
        self.location_offset = location_offset
        self.reward_thresholds = reward_thresholds
        self.sigma_levels = sigma_levels
        self.radius = radius
        if priority_distribution is None:
            def priority_distribution(): return np.random.rand()  # noqa: E731
        self.priority_distribution = priority_distribution

    def reset_overwrite_previous(self) -> None:
        self.targets = []

    def reset_pre_sim_init(self) -> None:
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(
                self._n_targets[0], self._n_targets[1])

        logger.info(f"Generating {self.n_targets} targets")
        self.regenerate_targets()

        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for target in self.targets:
                    satellite.add_location_for_access_checking(
                        object=target,
                        r_LP_P=target.r_LP_P,
                        min_elev=satellite.sat_args_generator["imageTargetMinimumElevation"], # was  np.radians(83)
                        type="target",
                    )

    def reset_during_sim_init(self) -> None:
        for target in self.targets:
            self.visualize_target(target)

        for groundStation in self.satellites[0].locations_for_access_checking:
            if groundStation['type'] == 'ground_station':
                self.visualize_groundStation(groundStation)

    def regenerate_targets(self) -> None:
        self.targets = []
        oceans = pd.read_csv(
            Path(os.path.realpath(__file__)).parent.parent
            / "_dat"
            / "ocean"
            # / "northwestAUoceanCloud_SAR_new.csv",
            / "CloudSARAoIs_diffPriority.csv"
        )

        # select ocean column
        ocean_list = oceans['ocean']
        tgtPriority = oceans['priority']
        maxPriority = max(tgtPriority)
        tgtCloud_cover = oceans['cloud']
        bearing_deg = oceans['bearing']

        n_ocean = 0
        for oceanName, priority_level, cloud_cover, bearing in zip(ocean_list, tgtPriority, tgtCloud_cover,bearing_deg):
            ocean = oceans.loc[oceans['ocean'] == oceanName]
            ocean_grids = self.generate_grid_coordinates(
                ocean['lat'], ocean['lng'], spacing=self.location_offset, rows=3, cols=10, bearing_deg=bearing) # bearing_deg=-6

            i = 0
            for _ in range(int(len(ocean_grids)/2)):
                aoi_coords = random.choice(ocean_grids)
                location = lla2ecef(
                    aoi_coords["lat"], aoi_coords["lng"], self.radius)
                location = location.reshape(-1)
                location /= np.linalg.norm(location)
                location *= self.radius
                cloud_cover_sigma = np.random.uniform(
                    self.sigma_levels[0], self.sigma_levels[1])
                cloud_cover_forecast = np.clip(np.random.normal(
                    cloud_cover, cloud_cover_sigma), 0.0, 1.0)

                self.targets.append(
                    TargetwithCloud(
                        name=f"AoI-{n_ocean}.{i}-[{cloud_cover_forecast}]-[{priority_level}]",
                        r_LP_P=location,
                        priority=self.priority_distribution(),
                        cloud_cover_true=cloud_cover,
                        cloud_cover_sigma=cloud_cover_sigma,
                        cloud_cover_forecast=cloud_cover_forecast
                    )
                )
                i += 1
            n_ocean += 1

    @vizard.visualize
    def visualize_target(self, target, vizSupport=None, vizInstance=None):
        vizSupport.addLocation(
            vizInstance,
            stationName=target.name,
            parentBodyName="earth",
            r_GP_P=list(target.r_LP_P),
            fieldOfView=np.arctan(25 / 500),  # np.arctan(500 / 800),
            color=vizSupport.toRGBA255("white"),
            range=1000.0 * 1000,
        )
        if vizInstance.settings.showLocationCones == 0:
            vizInstance.settings.showLocationCones = -1
        if vizInstance.settings.showLocationCommLines == 0:
            vizInstance.settings.showLocationCommLines = -1
        if vizInstance.settings.showLocationLabels == 0:
            vizInstance.settings.showLocationLabels = -1

    @vizard.visualize
    def visualize_groundStation(self, target, vizSupport=None, vizInstance=None):
        """Visualize ground station in Vizard."""
        vizSupport.addLocation(
            vizInstance,
            stationName=f"{target['ground_station']}".replace(
                "GroundStation", ""),
            parentBodyName="earth",
            r_GP_P=list(target['r_LP_P']),
            fieldOfView=np.arctan(target['min_elev']),
            color=vizSupport.toRGBA255("green"),
            range=1000.0 * 1000,  # meters
        )

    def generate_grid_coordinates(self, center_lat, center_lon,
                                          spacing, rows, cols,
                                          bearing_deg):
        R = self.radius  # Earth radius

        # Convert center to radians
        lat0 = np.radians(center_lat)
        lon0 = np.radians(center_lon)

        # Use orbit inclination as the rotation angle
        theta = np.radians(bearing_deg)

        # Compute coordinate grid offsets in km from the center
        row_offsets = np.arange(-(rows // 2), (rows // 2) + 1)
        col_offsets = np.arange(-(cols // 2), (cols // 2) + 1)

        grid_coords = []

        for i in row_offsets:
            for j in col_offsets:
                # Offset in local East (x) and North (y) directions
                dx = j * spacing
                dy = -i * spacing  # minus to match geographic north-up convention

                # Rotate by orbit inclination
                x_rot = dx * np.cos(theta) - dy * np.sin(theta)
                y_rot = dx * np.sin(theta) + dy * np.cos(theta)

                # Convert km shifts to angular shifts
                dlat = y_rot / R
                dlon = x_rot / (R * np.cos(lat0))

                lat = lat0 + dlat
                lon = lon0 + dlon

                coords = {}
                coords['lat'] = np.degrees(lat)
                coords['lng'] = np.degrees(lon)

                grid_coords.append(coords)

        return grid_coords


class RandomOrbitalTargetswithCloud(Scenario):
    """Environment with randomised targets under the orbital groundtrack plus cloud coverage."""
    # mu_data = 0.6740208166434426  # Average global cloud coverage

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        location_offset: float = 40000,
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
        sigma_levels: tuple[float, float] = (0.01, 0.05),
        reward_thresholds: Union[float, tuple[float, float]] = 0.95,
        n_orbits: float = 1.0,


    ) -> None:
        """Construct environment with static targets around sea area with cloud
        coverage predefined with refer to a .csv file in _dat folder.
        """
        self._n_targets = n_targets
        self.n_orbits = n_orbits
        self.location_offset = location_offset
        self.reward_thresholds = reward_thresholds
        self.sigma_levels = sigma_levels
        self.radius = radius
        if priority_distribution is None:
            def priority_distribution(): return np.random.rand()  # noqa: E731
        self.priority_distribution = priority_distribution

    def reset_overwrite_previous(self) -> None:
        self.targets = []

    def reset_pre_sim_init(self) -> None:
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(
                self._n_targets[0], self._n_targets[1])

        logger.info(f"Generating {self.n_targets} targets")
        self.regenerate_targets()

        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for target in self.targets:
                    satellite.add_location_for_access_checking(
                        object=target,
                        r_LP_P=target.r_LP_P,
                        min_elev=np.radians(83), # was satellite.sat_args_generator["imageTargetMinimumElevation"]
                        type="target",
                    )

    def reset_during_sim_init(self) -> None:
        for target in self.targets:
            self.visualize_target(target)

        for groundStation in self.satellites[0].locations_for_access_checking:
            if groundStation['type'] == 'ground_station':
                self.visualize_groundStation(groundStation)

    def regenerate_targets(self) -> None:
        self.targets = []
        tgt_list = []
        for sat in self.satellites:
            # Generate track
            latitudes, longitudes = self.generate_LEO_ground_track(
                inclination_deg=np.rad2deg(sat.sat_args['oe'].i),
                altitude_km=sat.sat_args['oe'].a/1000,
                num_points_per_orbit=50,
                num_orbits=self.n_orbits,
                phase_offset_deg=np.rad2deg(sat.sat_args['oe'].f)
            )
            # Generate Region Coords
            tgt_sat=self.generate_regions_under_track(latitudes,longitudes,num_targets=self._n_targets)
            tgt_list.append(tgt_sat)
        
        tgt_list=list(np.concatenate(tgt_list))
        n_regions=0
        for tgt in tgt_list:
            tgt_grids = self.generate_grid_coordinates(
                tgt[0], tgt[1], spacing=self.location_offset, rows=3, cols=10, bearing_deg=np.rad2deg(sat.sat_args['oe'].i))
            i = 0
            cloud_cover = np.random.rand()
            for _ in range(int(len(tgt_grids)/2)):
                aoi_coords = random.choice(tgt_grids)
                location = lla2ecef(
                    aoi_coords["lat"], aoi_coords["lng"], self.radius)
                location = location.reshape(-1)
                location /= np.linalg.norm(location)
                location *= self.radius
                cloud_cover_sigma = np.random.uniform(
                    self.sigma_levels[0], self.sigma_levels[1])
                cloud_cover_forecast = np.clip(np.random.normal(
                    cloud_cover, cloud_cover_sigma), 0.0, 1.0)
                priority_level = self.priority_distribution()
                self.targets.append(
                    TargetwithCloud(
                        name=f"AoI-{n_regions}.{i}-[{cloud_cover_forecast}]-[{priority_level}]",
                        r_LP_P=location,
                        priority=priority_level,
                        cloud_cover_true=cloud_cover,
                        cloud_cover_sigma=cloud_cover_sigma,
                        cloud_cover_forecast=cloud_cover_forecast
                    )
                )
                i += 1
            n_regions += 1

    @vizard.visualize
    def visualize_target(self, target, vizSupport=None, vizInstance=None):
        vizSupport.addLocation(
            vizInstance,
            stationName=target.name,
            parentBodyName="earth",
            r_GP_P=list(target.r_LP_P),
            fieldOfView=np.arctan(25 / 500),  # np.arctan(500 / 800),
            color=vizSupport.toRGBA255("white"),
            range=1000.0 * 1000,
        )
        if vizInstance.settings.showLocationCones == 0:
            vizInstance.settings.showLocationCones = -1
        if vizInstance.settings.showLocationCommLines == 0:
            vizInstance.settings.showLocationCommLines = -1
        if vizInstance.settings.showLocationLabels == 0:
            vizInstance.settings.showLocationLabels = -1

    @vizard.visualize
    def visualize_groundStation(self, target, vizSupport=None, vizInstance=None):
        """Visualize ground station in Vizard."""
        vizSupport.addLocation(
            vizInstance,
            stationName=f"{target['ground_station']}".replace(
                "GroundStation", ""),
            parentBodyName="earth",
            r_GP_P=list(target['r_LP_P']),
            fieldOfView=np.arctan(target['min_elev']),
            color=vizSupport.toRGBA255("green"),
            range=1000.0 * 1000,  # meters
        )

    def generate_grid_coordinates(self, center_lat, center_lon,
                                          spacing, rows, cols,
                                          bearing_deg):
        R = self.radius  # Earth radius

        # Convert center to radians
        lat0 = np.radians(center_lat)
        lon0 = np.radians(center_lon)

        # Use orbit inclination as the rotation angle
        theta = np.radians(bearing_deg)

        # Compute coordinate grid offsets in km from the center
        row_offsets = np.arange(-(rows // 2), (rows // 2) + 1)
        col_offsets = np.arange(-(cols // 2), (cols // 2) + 1)

        grid_coords = []

        for i in row_offsets:
            for j in col_offsets:
                # Offset in local East (x) and North (y) directions
                dx = j * spacing
                dy = -i * spacing  # minus to match geographic north-up convention

                # Rotate by orbit inclination
                x_rot = dx * np.cos(theta) - dy * np.sin(theta)
                y_rot = dx * np.sin(theta) + dy * np.cos(theta)

                # Convert km shifts to angular shifts
                dlat = y_rot / R
                dlon = x_rot / (R * np.cos(lat0))

                lat = lat0 + dlat
                lon = lon0 + dlon

                coords = {}
                coords['lat'] = np.degrees(lat)
                coords['lng'] = np.degrees(lon)

                grid_coords.append(coords)

        return grid_coords

    def generate_LEO_ground_track(self, inclination_deg=50.0, altitude_km=500,
                                num_points_per_orbit=20, num_orbits=0.5, phase_offset_deg=0.0):
        EARTH_RADIUS = self.radius/1000 # km
        MU = 398600.4418       # km^3/s^2
        ROTATION_RATE = 360 / 86164.0  # deg/s (Earth rotation rate)

        semi_major_axis = EARTH_RADIUS + altitude_km
        period = 2 * np.pi * np.sqrt(semi_major_axis**3 / MU)  # seconds
        total_time = num_orbits * period
        time_steps = np.linspace(0, total_time, int(num_points_per_orbit * num_orbits))

        latitudes, longitudes = [], []
        prev_long = None

        for t in time_steps:
            mean_motion = 2 * np.pi / period
            true_anomaly = (mean_motion * t + np.radians(phase_offset_deg)) % (2 * np.pi)

            # Latitude from inclination
            lat = np.degrees(np.arcsin(np.sin(np.radians(inclination_deg)) * np.sin(true_anomaly)))

            # Include Earth's rotation shift
            lon = (np.degrees(true_anomaly+1) - ROTATION_RATE * t) % 360
            lon = lon - 360 if lon > 180 else lon

            # Prevent discontinuity when orbit crosses ±180°
            if prev_long is not None:
                delta = lon - prev_long
                if delta > 180:
                    lon -= 360
                elif delta < -180:
                    lon += 360
            prev_long = lon

            latitudes.append(lat)
            longitudes.append(lon)

        return np.array(latitudes), np.array(longitudes)

    # --------------------------
    # Generate targets directly under ground track
    # --------------------------
    def generate_regions_under_track(self, latitudes, longitudes, num_targets=5, lat_offset_deg=0.05, lon_offset_deg=0.1):
        step = max(1, len(latitudes) // num_targets)
        targets = []
        for i in range(0, len(latitudes), step):
            lat = latitudes[i] + random.uniform(-lat_offset_deg, lat_offset_deg)
            lon = longitudes[i] + random.uniform(-lon_offset_deg, lon_offset_deg)
            elev = 0.0  # elevation in meters
            targets.append((lat, lon, elev))
            if len(targets) >= num_targets:
                break
        return targets


__doc_title__ = "Target Scenarios"
__all__ = ["Target", "TargetwithCloud", "UniformTargets",
           "CityTargets", "UserDefinedTargets", "UserDefOceanTargetswithCloud","RandomOceanTargetswithCloud"]
