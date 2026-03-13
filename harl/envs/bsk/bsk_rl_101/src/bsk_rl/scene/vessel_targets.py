# target_with_vessels.py
from bsk_rl.utils import vizard
from bsk_rl.scene import Scenario
from bsk_rl.utils.orbital import lla2ecef
from Basilisk.utilities import orbitalMotion
from typing import Callable, Optional, Union, Tuple
import logging
import numpy as np
import pandas as pd
from typing import Iterable
from pathlib import Path
import os
import random


class TargetWithVessels:
    """Ground target with a configurable number of embedded vessels."""

    def __init__(self, name: str, r_LP_P: Iterable[float], priority: float, n_vessels: int = 3) -> None:
        self.name = name
        self.r_LP_P = np.array(r_LP_P)
        self.priority = priority
        self.vessels = [f"{self.name}_v{i}" for i in range(n_vessels)]

    @property
    def id(self) -> str:
        try:
            return self._id
        except AttributeError:
            self._id = f"{self.name}_{id(self)}"
            return self._id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"TargetWithVessels({self.name}, vessels={len(self.vessels)})"


# uniform_targets_with_vessels.py

logger = logging.getLogger(__name__)


class UniformTargetsWithVessels(Scenario):
    """Uniformly distributed targets with embedded vessels."""

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        n_vessels_per_target: Optional[Union[int, Tuple[int, int]]] = None,
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
    ) -> None:
        super().__init__()
        self._n_targets = n_targets
        self.n_vessels_per_target = n_vessels_per_target
        self.radius = radius
        self.priority_distribution = priority_distribution or (
            lambda: np.random.rand())

    def reset_overwrite_previous(self) -> None:
        self.targets = []

    def reset_pre_sim_init(self) -> None:
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(
                self._n_targets[0], self._n_targets[1])

        logger.info(f"Generating {self.n_targets} vessel-based targets")
        self.regenerate_targets()

        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for target in self.targets:
                    satellite.add_location_for_access_checking(
                        object=target,
                        r_LP_P=target.r_LP_P,
                        min_elev=satellite.sat_args_generator["imageTargetMinimumElevation"],
                        type="target",
                    )

    def reset_during_sim_init(self) -> None:
        for target in self.targets:
            self.visualize_target(target)

    def regenerate_targets(self) -> None:
        self.targets = []
        for i in range(self.n_targets):
            x = np.random.normal(size=3)
            x *= self.radius / np.linalg.norm(x)
            target = TargetWithVessels(
                name=f"tgt-vessel-{i}",
                r_LP_P=x,
                priority=self.priority_distribution(),
                n_vessels=self.n_vessels_per_target
            )
            self.targets.append(target)

    @vizard.visualize
    def visualize_target(self, target, vizSupport=None, vizInstance=None):
        vizSupport.addLocation(
            vizInstance,
            stationName=target.name,
            parentBodyName="earth",
            r_GP_P=list(target.r_LP_P),
            fieldOfView=np.arctan(500 / 800),
            color=vizSupport.toRGBA255("white"),
            range=1000.0 * 1000,
        )
        if vizInstance.settings.showLocationCones == 0:
            vizInstance.settings.showLocationCones = -1
        if vizInstance.settings.showLocationCommLines == 0:
            vizInstance.settings.showLocationCommLines = -1
        if vizInstance.settings.showLocationLabels == 0:
            vizInstance.settings.showLocationLabels = -1


class UniformTargetsWithRandomVessels(UniformTargetsWithVessels):
    """Extension of UniformTargetsWithVessels with variable vessel count per target."""

    def __init__(self, *args, vessel_distribution_fn=None, **kwargs):
        """
        Args:
            vessel_distribution_fn: Callable that returns an int for vessels per target.
        """
        super().__init__(*args, **kwargs)
        if vessel_distribution_fn is None:
            raise ValueError("vessel_distribution_fn must be provided.")
        self.vessel_distribution_fn = vessel_distribution_fn

    def regenerate_targets(self) -> None:
        self.targets = []
        for i in range(self.n_targets):
            x = np.random.normal(size=3)
            x *= self.radius / np.linalg.norm(x)
            n_vessels = self.vessel_distribution_fn()
            target = TargetWithVessels(
                name=f"tgt-vessel-{i}",
                r_LP_P=x,
                priority=self.priority_distribution(),
                n_vessels=n_vessels
            )
            self.targets.append(target)

# Module for locating the vessels around certain sea areas Stage-1


class OceanTargetsWithFixedVessels(Scenario):
    """Ocean targets with embedded fixed number of vessels in each AoI."""

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        n_vessels_per_target: Optional[Union[int, Tuple[int, int]]] = None,
        priority_distribution: Optional[Callable] = None,
        location_offset: float = 50000,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
    ) -> None:
        super().__init__()
        self._n_targets = n_targets
        self.n_vessels_per_target = n_vessels_per_target
        self.radius = radius
        self.location_offset = location_offset
        self.priority_distribution = priority_distribution or (
            lambda: np.random.rand())

    def reset_overwrite_previous(self) -> None:
        self.targets = []

    def reset_pre_sim_init(self) -> None:
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(
                self._n_targets[0], self._n_targets[1])

        logger.info(f"Generating {self.n_targets} vessel-based targets")
        self.regenerate_targets()

        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for target in self.targets:
                    satellite.add_location_for_access_checking(
                        object=target,
                        r_LP_P=target.r_LP_P,
                        min_elev=satellite.sat_args_generator["imageTargetMinimumElevation"],
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
            / "northwestAUocean.csv",
        )

        # select ocean column
        ocean_list = oceans['ocean']
        tgtPriority = oceans['priority']
        maxPriority = max(tgtPriority)

        n_ocean = 0
        n_vessels = self.n_vessels_per_target

        for oceanName, priority_level in zip(ocean_list, tgtPriority):
            ocean = oceans.loc[oceans['ocean'] == oceanName]
            ocean_grids = self.generate_grid_coordinates(
                ocean['lat'], ocean['lng'], spacing=self.location_offset, rows=2, cols=7, bearing_deg=20)
            i = 0
            for aoi_coords in ocean_grids:
                location = lla2ecef(
                    aoi_coords["lat"], aoi_coords["lng"], self.radius)
                location = location.reshape(-1)
                location /= np.linalg.norm(location)
                location *= self.radius
                self.targets.append(
                    TargetWithVessels(
                        name=f"AoI-{n_ocean}.{i}-[{priority_level}]",
                        r_LP_P=location,
                        priority=priority_level/maxPriority,
                        n_vessels=n_vessels
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
        if vizInstance.settings.showLocationCones == 0:
            vizInstance.settings.showLocationCones = -1
        if vizInstance.settings.showLocationCommLines == 0:
            vizInstance.settings.showLocationCommLines = -1
        if vizInstance.settings.showLocationLabels == 0:
            vizInstance.settings.showLocationLabels = -1

    def generate_grid_coordinates(self, center_lat, center_lon, spacing, rows, cols, bearing_deg):
        R = self.radius  # Earth radius

        # Convert center to radians
        lat0 = np.radians(center_lat)
        lon0 = np.radians(center_lon)

        # Rotation angle in radians (clockwise from north)
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

                # Rotate by bearing
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


class OceanRandomVesselsTargets(UniformTargetsWithVessels):
    """Environment with user defined Vessels located in specific sea targets defined in src/bsk_rl/_dat/ocean/ocean.csv."""

    def __init__(self, *args, vessel_distribution_fn=None, **kwargs):
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
        super().__init__(*args, **kwargs)
        self.location_offset = 50000
        self.radius = orbitalMotion.REQ_EARTH * 1e3,
        if vessel_distribution_fn is None:
            raise ValueError("vessel_distribution_fn must be provided.")
        self.vessel_distribution_fn = vessel_distribution_fn

    def reset_overwrite_previous(self) -> None:
        self.targets = []

    def reset_pre_sim_init(self) -> None:
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(
                self._n_targets[0], self._n_targets[1])

        logger.info(f"Generating {self.n_targets} vessel-based targets")
        self.regenerate_targets()

        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for target in self.targets:
                    satellite.add_location_for_access_checking(
                        object=target,
                        r_LP_P=target.r_LP_P,
                        min_elev=satellite.sat_args_generator["imageTargetMinimumElevation"],
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
            / "northwestAUocean.csv",
        )

        # select ocean column
        ocean_list = oceans['ocean']
        tgtPriority = oceans['priority']
        maxPriority = max(tgtPriority)

        n_ocean = 0
        for oceanName, priority_level in zip(ocean_list, tgtPriority):
            ocean = oceans.loc[oceans['ocean'] == oceanName]
            ocean_grids = self.generate_grid_coordinates(
                ocean['lat'], ocean['lng'], spacing=self.location_offset, rows=2, cols=7, bearing_deg=20)
            i = 0
            for aoi_coords in ocean_grids:
                location = lla2ecef(
                    aoi_coords["lat"], aoi_coords["lng"], self.radius)
                location = location.reshape(-1)
                location /= np.linalg.norm(location)
                location *= self.radius
                n_vessels = self.vessel_distribution_fn()
                self.targets.append(
                    TargetWithVessels(
                        name=f"AoI-{n_ocean}.{i}-[{priority_level}]",
                        r_LP_P=location,
                        priority=priority_level/maxPriority,
                        n_vessels=n_vessels
                    )
                )
                i += 1
            n_ocean += 1

    def generate_grid_coordinates(self, center_lat, center_lon, spacing, rows, cols, bearing_deg):
        R = self.radius  # Earth radius

        # Convert center to radians
        lat0 = np.radians(center_lat)
        lon0 = np.radians(center_lon)

        # Rotation angle in radians (clockwise from north)
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

                # Rotate by bearing
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
        if vizInstance.settings.showLocationCones == 0:
            vizInstance.settings.showLocationCones = -1
        if vizInstance.settings.showLocationCommLines == 0:
            vizInstance.settings.showLocationCommLines = -1
        if vizInstance.settings.showLocationLabels == 0:
            vizInstance.settings.showLocationLabels = -1


# Module for locating the vessels around the sea all over the world Stage-2
class UniformRandomOceanVesselsTargets(UniformTargetsWithVessels):
    """Environment with user defined Vessels located in specific sea targets defined in src/bsk_rl/_dat/ocean/ocean.csv."""

    def __init__(self, *args, vessel_distribution_fn=None, **kwargs):
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
        super().__init__(*args, **kwargs)
        self.location_offset = 50000
        self.radius = orbitalMotion.REQ_EARTH * 1e3,
        if vessel_distribution_fn is None:
            raise ValueError("vessel_distribution_fn must be provided.")
        self.vessel_distribution_fn = vessel_distribution_fn

    def reset_overwrite_previous(self) -> None:
        self.targets = []

    def reset_pre_sim_init(self) -> None:
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(
                self._n_targets[0], self._n_targets[1])

        logger.info(f"Generating {self.n_targets} vessel-based targets")
        self.regenerate_targets()

        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for target in self.targets:
                    satellite.add_location_for_access_checking(
                        object=target,
                        r_LP_P=target.r_LP_P,
                        min_elev=satellite.sat_args_generator["imageTargetMinimumElevation"],
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
        target_coords = self.generate_sea_coordinates()
        n_ocean = 0
        for x in target_coords:
            bearing = self.priority_distribution()
            bearing_deg = bearing*90
            ocean_grids = self.generate_grid_coordinates(
                x[0], x[1], spacing=self.location_offset, rows=2, cols=7, bearing_deg=bearing_deg)
            i = 0
            for aoi_coords in ocean_grids:
                location = lla2ecef(
                    aoi_coords["lat"], aoi_coords["lng"], self.radius)
                location = location.reshape(-1)
                location /= np.linalg.norm(location)
                location *= self.radius
                n_vessels = self.vessel_distribution_fn()
                self.targets.append(
                    TargetWithVessels(
                        name=f"AoI-{n_ocean}.{i}",
                        r_LP_P=location,
                        priority=self.priority_distribution(),
                        n_vessels=n_vessels
                    )
                )
                i += 1
            n_ocean += 1

    def generate_grid_coordinates(self, center_lat, center_lon, spacing, rows, cols, bearing_deg):
        R = self.radius[0]  # Earth radius

        # Convert center to radians
        lat0 = np.radians(center_lat)
        lon0 = np.radians(center_lon)

        # Rotation angle in radians (clockwise from north)
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

                # Rotate by bearing
                x_rot = dx * np.cos(theta) - dy * np.sin(theta)
                y_rot = dx * np.sin(theta) + dy * np.cos(theta)

                # Convert km shifts to angular shifts
                dlat = y_rot / R
                dlon = x_rot / (R * np.array(np.cos(lat0)))

                lat = lat0 + dlat
                lon = lon0 + dlon

                coords = {}
                coords['lat'] = np.degrees(lat)
                coords['lng'] = np.degrees(lon)

                grid_coords.append(coords)

        return grid_coords

    def generate_sea_coordinates(self):
        sea_coords = []

        # Regions with higher ocean coverage
        # Avoid land areas or continents like Europe, Asia interiors, Africa, etc.
        while len(sea_coords) < self.n_targets:
            lat = random.uniform(-60, 60)  # Avoid polar regions
            lon = random.uniform(-180, 180)

            # Filter to reduce land points (heuristics only)
            if (
                # Avoid Africa/Europe
                not (-60 < lat < 70 and -20 < lon < 50) and
                not (0 < lat < 70 and 60 < lon < 160) and        # Avoid Asia
                not (-50 < lat < 60 and -100 < lon < -30) and    # Avoid Americas
                not (-40 < lat < -10 and 110 < lon <
                     155)        # Avoid Australia
            ):
                sea_coords.append([round(lat, 4), round(lon, 4)])

        return sea_coords

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
        if vizInstance.settings.showLocationCones == 0:
            vizInstance.settings.showLocationCones = -1
        if vizInstance.settings.showLocationCommLines == 0:
            vizInstance.settings.showLocationCommLines = -1
        if vizInstance.settings.showLocationLabels == 0:
            vizInstance.settings.showLocationLabels = -1


__doc_title__ = "Target Scenarios"
__all__ = ["TargetWithVessels", "UniformTargetsWithVessels",
           "UniformTargetsWithRandomVessels", "OceanTargetsWithFixedVessels", "OceanRandomVesselsTargets", "UniformRandomOceanVesselsTargets"]
