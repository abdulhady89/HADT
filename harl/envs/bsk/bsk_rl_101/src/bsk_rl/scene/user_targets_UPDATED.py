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


class TargetwithCloud:
    """Ground target with associated value and cloud coverage."""

    def __init__(self,
                 name: str,
                 r_LP_P: Iterable[float],
                 priority: float,
                 cloud_cover_true: float,
                 cloud_cover_sigma: float,
                 cloud_cover_forecast: float,
                 lat: float,
                 lon: float,
                 ) -> None:
        """Ground target with associated priority, location, and cloud coverage.

        Args:
            name: Identifier; does not need to be unique
            r_LP_P: Planet-fixed, planet relative location [m]
            priority: Value metric.
            cloud_cover_true: True cloud coverage (0-1)
            cloud_cover_sigma: Uncertainty in cloud forecast
            cloud_cover_forecast: Forecasted cloud coverage (0-1)
            lat: Latitude in degrees
            lon: Longitude in degrees
        """
        self.name = name
        self.r_LP_P = np.array(r_LP_P)
        self.priority = priority
        self.cloud_cover_true = cloud_cover_true
        self.cloud_cover_sigma = cloud_cover_sigma
        self.cloud_cover_forecast = cloud_cover_forecast
        self.lat = lat
        self.lon = lon

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


class UserDefOceanTargetswithCloud(Scenario):
    """Environment with user defined ocean targets with cloud coverage."""

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        location_offset: float = 50000,
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
        sigma_levels: tuple[float, float] = (0.01, 0.05),
        reward_thresholds: Union[float, tuple[float, float]] = 0.95,
    ) -> None:
        """Construct environment with static targets around sea area with cloud
        coverage predefined with refer to a .csv file in _dat folder.

        Automatically reads from _dat/ocean/WP4_input.csv if available, 
        otherwise uses northwestAUoceanCloud.csv as fallback.

        Args:
            n_targets: Number of targets to generate
            location_offset: Offset for grid generation (meters)
            priority_distribution: Function for generating priority (unused if CSV provided)
            radius: Earth radius (meters)
            sigma_levels: Range for cloud forecast uncertainty
            reward_thresholds: Reward threshold parameters
        """
        self._n_targets = n_targets
        self.location_offset = location_offset
        self.reward_thresholds = reward_thresholds
        self.sigma_levels = sigma_levels
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
                        min_elev=satellite.sat_args_generator["imageTargetMinimumElevation"],
                        type="target",
                    )

    def reset_during_sim_init(self) -> None:
        """Visualize targets and ground stations in Vizard on reset."""
        for target in self.targets:
            self.visualize_target(target)

        for groundStation in self.satellites[0].locations_for_access_checking:
            if groundStation['type'] == 'ground_station':
                self.visualize_groundStation(groundStation)

    def regenerate_targets(self) -> None:
        """Modified to read from WP4_input.csv in _dat/ocean/ directory,
        otherwise fall back to default northwestAUoceanCloud.csv
        """
        self.targets = []

        input_csv_path = Path(os.path.realpath(
            __file__)).parent.parent / "_dat" / "ocean" / "WP4_input.csv"

        print(f"\n{'='*60}")
        print(f"CHECKING FOR INPUT FILE")
        print(f"{'='*60}")
        print(f"Looking for: {input_csv_path}")
        print(f"File exists: {os.path.exists(input_csv_path)}")
        print(f"{'='*60}\n")

        try:
            if os.path.exists(input_csv_path):
                print(f"SUCCESS: Reading targets from {input_csv_path}\n")
                logger.info(f"Loading targets from {input_csv_path}")
                input_data = pd.read_csv(input_csv_path)
                print(f"Loaded {len(input_data)} rows from WP4_input.csv")
                print(f"Columns found: {list(input_data.columns)}")
                print(f"First 3 rows:\n{input_data.head(3)}\n")

                required_cols = ['lat', 'lon']
                if not all(col in input_data.columns for col in required_cols):
                    raise ValueError(
                        f"Input CSV must contain columns: {required_cols}")

                if 'priority' not in input_data.columns:
                    input_data['priority'] = 1
                if 'cloud' not in input_data.columns:
                    input_data['cloud'] = 0.5
                if 'point_id' not in input_data.columns:
                    input_data['point_id'] = range(len(input_data))

                if input_data['priority'].max() > 1:
                    maxPriority = input_data['priority'].max()
                else:
                    maxPriority = 1

                for idx, row in input_data.iterrows():
                    lat = row['lat']
                    lon = row['lon']
                    priority = row.get('priority', 1)
                    cloud_cover = row.get('cloud', 0.5)
                    point_id = row.get('point_id', idx)

                    location = lla2ecef(lat, lon, self.radius)
                    location = location.reshape(-1)
                    location /= np.linalg.norm(location)
                    location *= self.radius

                    cloud_cover_sigma = np.random.uniform(
                        self.sigma_levels[0], self.sigma_levels[1])
                    cloud_cover_forecast = np.clip(np.random.normal(
                        cloud_cover, cloud_cover_sigma), 0.0, 1.0)

                    self.targets.append(
                        TargetwithCloud(
                            name=f"Point-{point_id}-[{cloud_cover_forecast:.2f}]-[{priority}]",
                            r_LP_P=location,
                            priority=priority/maxPriority,
                            cloud_cover_true=cloud_cover,
                            cloud_cover_sigma=cloud_cover_sigma,
                            cloud_cover_forecast=cloud_cover_forecast,
                            lat=lat,
                            lon=lon,
                        )
                    )

                logger.info(
                    f"Successfully loaded {len(self.targets)} targets from WP4_input.csv")
                return

        except Exception as e:
            logger.warning(
                f"Could not load WP4_input.csv: {e}. Using default targets from northwestAUoceanCloud.csv")

        oceans = pd.read_csv(
            Path(os.path.realpath(__file__)).parent.parent
            / "_dat"
            / "ocean"
            / "northwestAUoceanCloud.csv",
        )

        ocean_list = oceans['ocean']
        tgtPriority = oceans['priority']
        maxPriority = max(tgtPriority)
        tgtCloud_cover = oceans['cloud']

        n_ocean = 0
        for oceanName, priority_level, cloud_cover in zip(ocean_list, tgtPriority, tgtCloud_cover):
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
                        cloud_cover_forecast=cloud_cover_forecast,
                        lat=aoi_coords["lat"],
                        lon=aoi_coords["lng"],
                    )
                )
                i += 1
            n_ocean += 1

    @vizard.visualize
    def visualize_target(self, target, vizSupport=None, vizInstance=None):
        """Visualize target in Vizard."""
        vizSupport.addLocation(
            vizInstance,
            stationName=target.name,
            parentBodyName="earth",
            r_GP_P=list(target.r_LP_P),
            fieldOfView=np.arctan(25 / 500),
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
            range=1000.0 * 1000,
        )
        if vizInstance.settings.showLocationCones == 0:
            vizInstance.settings.showLocationCones = -1
        if vizInstance.settings.showLocationCommLines == 0:
            vizInstance.settings.showLocationCommLines = -1
        if vizInstance.settings.showLocationLabels == 0:
            vizInstance.settings.showLocationLabels = -1

    def generate_grid_coordinates(self, center_lat, center_lon, spacing, rows, cols, bearing_deg):
        """Generate a grid of coordinates around a center point."""
        R = self.radius

        lat0 = np.radians(center_lat)
        lon0 = np.radians(center_lon)

        theta = np.radians(bearing_deg)

        row_offsets = np.arange(-(rows // 2), (rows // 2) + 1)
        col_offsets = np.arange(-(cols // 2), (cols // 2) + 1)

        grid_coords = []

        for i in row_offsets:
            for j in col_offsets:
                dx = j * spacing
                dy = -i * spacing

                x_rot = dx * np.cos(theta) - dy * np.sin(theta)
                y_rot = dx * np.sin(theta) + dy * np.cos(theta)

                dlat = y_rot / R
                dlon = x_rot / (R * np.cos(lat0))

                lat = lat0 + dlat
                lon = lon0 + dlon

                coords = {}
                coords['lat'] = np.degrees(lat)
                coords['lng'] = np.degrees(lon)

                grid_coords.append(coords)

        return grid_coords


__doc_title__ = "Target Scenarios"
__all__ = ["Target", "TargetwithCloud", "UniformTargets",
           "CityTargets", "UserDefinedTargets", "UserDefOceanTargetswithCloud"]
