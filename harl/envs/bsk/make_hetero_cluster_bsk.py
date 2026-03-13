from collections.abc import Iterable
import warnings
warnings.filterwarnings("ignore")
import gymnasium as gym
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit
import numpy as np


try:
    from bsk_rl import GeneralSatelliteTasking
except ImportError:
    warnings.warn(
        "BSK-RL is not installed, so these environments will not be available! Please double-check the installation of Basilisk and BSK-RL library first (`pip show basilisk` and `pip show bsk-rl`)"
    )
from bsk_rl import sats, act, obs, scene, data, comm
from bsk_rl.sim import dyn, fsw

from bsk_rl.utils.orbital import random_orbit
from bsk_rl.utils.orbital import walker_delta_args


def make_BSK_Cluster_env(env_args, optim_challenge, randomness=None):
    # Specify the Satelllite Name:
    satellite_names = []
    satellite_names = ["OPT-"+f'{i}' for i in range(env_args.n_satellites)]
    
    # Common orbital parameters for all satellites
    inclination = [39.0, 41.0, 40.0]
    altitude = 500  # km, fixed for all satellites
    eccentricity = 0  # Circular orbit
    LAN = 0  # Longitude of Ascending Node (Omega), fixed for all
    arg_periapsis = 0  # Argument of Periapsis (omega), fixed for all
    true_anomaly_offsets = [-74, -74, -75]

    orbit_ls = []
    for offset in true_anomaly_offsets:
        orbit = random_orbit(
            i=inclination, alt=altitude, e=eccentricity, Omega=LAN, omega=arg_periapsis, f=offset
        )
        orbit_ls.append(orbit)

    if optim_challenge == "easy":
        battery_size = env_args.battery_capacity
        init_battery_level = env_args.init_battery_level
        memory_size = env_args.memory_size
        init_memory_percent = env_args.init_memory_percent
        baud_rate = env_args.baud_rate
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False

    elif optim_challenge == "hard":
        battery_size = env_args.battery_capacity
        init_battery_level = 85
        memory_size = env_args.memory_size
        init_memory_percent = 60
        baud_rate = int(0.3 * env_args.baud_rate)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
    else:
        print("Optimisation challenge name is not available")
        NotImplementedError
    
    if randomness is not None:
        if randomness == "random_all":
            random_init_memory = True
            random_init_battery = True
            random_disturbance = True
            random_RW_speed = True

        elif randomness == "random_batt":
            random_init_memory = False
            random_init_battery = True
            random_disturbance = False
            random_RW_speed = False
            
        elif randomness == "random_mem":
            random_init_memory = True
            random_init_battery = False
            random_disturbance = False
            random_RW_speed = False

        elif randomness == "random_dist":
            random_init_memory = False
            random_init_battery = False
            random_disturbance = True
            random_RW_speed = False

        elif randomness == "random_rw":
            random_init_memory = False
            random_init_battery = False
            random_disturbance = False
            random_RW_speed = True

        else:
            print("Randomization name not available")
            NotImplementedError

    # Define four satellites in a "train" Cluster formation along the same orbit
    multiSat = []
    index = 0

    for orbit in orbit_ls:
        sat_args = dict(
            # Power
            batteryStorageCapacity=battery_size * 3600,
            storedCharge_Init=int(battery_size * init_battery_level / 100 * 3600) if not random_init_battery else np.random.uniform(
                battery_size * 3600 * 0.4, battery_size * 3600 * 0.5),
            panelArea=1.0,
            panelEfficiency=20.0,
            basePowerDraw=-10.0,
            instrumentPowerDraw=-30,
            transmitterPowerDraw=-25,
            thrusterPowerDraw=-80,
            # Data Storage
            dataStorageCapacity=memory_size * 8e6,  # MB to bits,
            storageInit=int(memory_size * init_memory_percent/100) * 8e6 if not random_init_memory \
                else np.random.uniform(memory_size * 8e6 * 0.2, memory_size * 8e6 * 0.8),
            instrumentBaudRate=env_args.instr_baud_rate * 1e6,
            transmitterBaudRate=-1*baud_rate * 1e6,
            # Attitude
            imageAttErrorRequirement=0.1,
            imageRateErrorRequirement=0.1,
            disturbance_vector=lambda: np.random.normal(
                scale=0.0001, size=3) if random_disturbance else np.array([0.0, 0.0, 0.0]),
            maxWheelSpeed=6000.0,  # RPM
            wheelSpeeds=lambda: np.random.uniform(
                -3000, 3000, 3) if random_RW_speed else np.array([0.0, 0.0, 0.0]),
            desatAttitude="nadir",
            u_max=0.4,
            K1=0.25,
            K3=3.0,
            omega_max=0.1,
            servo_Ki=5.0,
            servo_P=150,
            # Orbital elements
            oe=orbit
        )

         # Define Imaging Satellite Object:
        class ImagingSatellite(sats.ImagingSatellite):
            observation_spec = [
                obs.SatProperties(
                    dict(prop="storage_level_fraction"),                        # 1
                    dict(prop="battery_charge_fraction"),                       # 1
                    dict(prop="wheel_speeds_fraction"),                         # 3
                    # dict(prop="omega_BP_P", norm=0.03),
                    # dict(prop="c_hat_P"),
                    # dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
                    # dict(prop="v_BN_P", norm=7616.5),
                ),

                obs.OpportunityProperties(
                    dict(prop="priority"),
                    # Cloud coverage forecast (percentage of the area covered by clouds)
                    dict(fn=lambda sat, opp: opp["object"].cloud_cover_forecast),
                    # Confidence on the cloud coverage forecast
                    dict(fn=lambda sat, opp: opp["object"].cloud_cover_sigma),
                    dict(prop="opportunity_open", norm=5700.0),
                    type="target",
                    n_ahead_observe=env_args.n_obs_image,
                ),
                obs.OpportunityProperties(
                    dict(prop="opportunity_open", norm=5700),
                    dict(prop="opportunity_close", norm=5700),
                    type="ground_station",
                    n_ahead_observe=1,
                ),
                obs.Eclipse(norm=5700),
                obs.Time(),
            ]
            action_spec = [
                act.Charge(duration=20.0),
                act.Downlink(duration=20.0),
                act.Desat(duration=20.0),
                # act.Drift(duration=1.0)
                act.Image(n_ahead_image=env_args.n_act_image),
            ]
            fsw_type = fsw.SteeringImagerFSWModel
            dyn_type = dyn.ManyGroundStationFullFeaturedDynModel

        sat = ImagingSatellite(satellite_names[index], sat_args)
        multiSat.append(sat)
        index += 1

    duration = env_args.orbit_num * 5700.0  # About 2 orbits

    env = GeneralSatelliteTasking(
        satellites=multiSat,
        scenario=scene.UserDefOceanTargetswithCloud(env_args.uniform_targets),
        rewarder=data.UniqueImageSARReward(),
        time_limit=duration,
        communicator=comm.LOSCommunication(),
        log_level="ERROR",
        terminate_on_time_limit=True,
        failure_penalty=env_args.failure_penalty,
        vizard_dir="./tmp/vizard" if env_args.use_render == True else None,
        vizard_settings=dict(showLocationLabels=-1, showLocationCommLines=1, showLocationCones=-1) if env_args.use_render == True else None,
    )
    return env



def make_BSK_SAR_OPT_env(env_args, optim_challenge, randomness=None):
    satellite_names = []
    satellite_names.append(f"OPT-1-Sat")
    satellite_names.append(f"OPT-2-Sat")
    satellite_names.append(f"SAR-Sat")

    inclination = [39.0, 41.0, 40.0]
    altitude = 500  # km, fixed for all satellites
    eccentricity = 0  # Circular orbit
    LAN = 0  # Longitude of Ascending Node (Omega), fixed for all
    arg_periapsis = 0  # Argument of Periapsis (omega), fixed for all
    true_anomaly_offsets = [-74, -74, -75]

    orbit_ls = []
    for offset, inc in zip(true_anomaly_offsets,inclination):
        orbit = random_orbit(
            i=inc, alt=altitude, e=eccentricity, Omega=LAN, omega=arg_periapsis, f=offset
        )
        orbit_ls.append(orbit)

    if optim_challenge == "easy":
        battery_size = env_args.battery_capacity
        init_battery_level = env_args.init_battery_level
        memory_size = env_args.memory_size
        init_memory_percent = env_args.init_memory_percent
        baud_rate = env_args.baud_rate
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False

    elif optim_challenge == "hard":
        battery_size = env_args.battery_capacity
        init_battery_level = 85
        memory_size = env_args.memory_size
        init_memory_percent = 60
        baud_rate = int(0.3 * env_args.baud_rate)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
    else:
        print("Optimisation challenge name is not available")
        NotImplementedError
    
    if randomness is not None:
        if randomness == "random_all":
            random_init_memory = True
            random_init_battery = True
            random_disturbance = True
            random_RW_speed = True

        elif randomness == "random_batt":
            random_init_memory = False
            random_init_battery = True
            random_disturbance = False
            random_RW_speed = False
            
        elif randomness == "random_mem":
            random_init_memory = True
            random_init_battery = False
            random_disturbance = False
            random_RW_speed = False

        elif randomness == "random_dist":
            random_init_memory = False
            random_init_battery = False
            random_disturbance = True
            random_RW_speed = False

        elif randomness == "random_rw":
            random_init_memory = False
            random_init_battery = False
            random_disturbance = False
            random_RW_speed = True

        else:
            print("Randomization name not available")
            NotImplementedError


    # Define the Optical-1 satellite arguments
    opt1_sat_args = dict(
        u_max=0.4,
        omega_max=0.1,
        servo_Ki=5.0,
        servo_P=150/5,
        K1=0.25,
        K3=3.0,
        imageTargetMinimumElevation=np.radians(83), #np.arctan(800 / 500),
        dataStorageCapacity=memory_size * 8e6,
        storageInit=int(memory_size * init_memory_percent/100) * 8e6 if not random_init_memory else np.random.uniform(memory_size * 8e6 * 0.6, memory_size * 8e6 * 0.8),
        instrumentBaudRate=env_args.instr_baud_rate * 1e6,  # 1Mbps
        transmitterBaudRate=-1*baud_rate * 1e6,
        batteryStorageCapacity=battery_size * 3600,
        storedCharge_Init=int(battery_size * 3600 * init_battery_level / 100) if not random_init_battery else np.random.uniform(battery_size * 3600 * 0.8, battery_size * 3600 * 0.95),
        panelArea=1.0,
        panelEfficiency=20.0,
        basePowerDraw=-10.0,
        instrumentPowerDraw=-30.0,
        transmitterPowerDraw=-25.0,
        thrusterPowerDraw=-80.0,
        imageAttErrorRequirement=0.1,
        imageRateErrorRequirement=0.1,
        disturbance_vector=np.array(
            [0.0, 0.0, 0.0]) if not random_disturbance else lambda: np.random.normal(scale=0.0001, size=3),
        maxWheelSpeed=6000.0,
        wheelSpeeds=np.array(
            [0.0, 0.0, 0.0]) if not random_RW_speed else lambda: np.random.uniform(-3000, 3000, 3),
        desatAttitude="nadir",
        oe=orbit_ls[0]
    )

    # Define the Optical-2 satellite arguments
    opt2_sat_args = dict(
        u_max=0.4,
        omega_max=0.1,
        servo_Ki=5.0,
        servo_P=150/5,
        K1=0.25,
        K3=3.0,
        imageTargetMinimumElevation=np.radians(83), #np.arctan(800 / 500),
        dataStorageCapacity=memory_size * 8e6,
        storageInit=int(memory_size * init_memory_percent/100) * 8e6 if not random_init_memory else np.random.uniform(memory_size * 8e6 * 0.6, memory_size * 8e6 * 0.8),
        instrumentBaudRate=env_args.instr_baud_rate * 1e6,  # 1Mbps
        transmitterBaudRate=-1*baud_rate * 1e6,
        batteryStorageCapacity=battery_size * 3600,
        storedCharge_Init=int(battery_size * 3600 * init_battery_level / 100) if not random_init_battery else np.random.uniform(battery_size * 3600 * 0.8, battery_size * 3600 * 0.95),
        panelArea=1.0,
        panelEfficiency=20.0,
        basePowerDraw=-10.0,
        instrumentPowerDraw=-30.0,
        transmitterPowerDraw=-25.0,
        thrusterPowerDraw=-80.0,
        imageAttErrorRequirement=0.1,
        imageRateErrorRequirement=0.1,
        disturbance_vector=np.array(
            [0.0, 0.0, 0.0]) if not random_disturbance else lambda: np.random.normal(scale=0.0001, size=3),
        maxWheelSpeed=6000.0,
        wheelSpeeds=np.array(
            [0.0, 0.0, 0.0]) if not random_RW_speed else lambda: np.random.uniform(-3000, 3000, 3),
        desatAttitude="nadir",
        oe=orbit_ls[1]
    )

    # Define the SAR satellite arguments
    sar_sat_args = dict(
        u_max=0.4,
        omega_max=0.1,
        servo_Ki=5.0,
        servo_P=150/5,
        K1=0.25,
        K3=3.0,
        imageTargetMinimumElevation=np.radians(60),
        dataStorageCapacity=2*memory_size * 8e6,
        storageInit=int(2*memory_size *init_memory_percent/100) * 8e6 if not random_init_memory else np.random.uniform(2*memory_size * 8e6 * 0.6, 2*memory_size * 8e6 * 0.8),
        instrumentBaudRate=2 * env_args.instr_baud_rate * 1e6,  # 1Mbps
        transmitterBaudRate=-1*baud_rate * 1e6,
        batteryStorageCapacity=2*battery_size * 3600,
        storedCharge_Init=int(2*battery_size * 3600 *init_battery_level / 100) if not random_init_battery else np.random.uniform(2*battery_size * 3600 * 0.8, 2*battery_size * 3600 * 0.95),
        panelArea=2.0,
        panelEfficiency=20.0,
        basePowerDraw=-10.0,
        instrumentPowerDraw=-120.0,
        transmitterPowerDraw=-25.0,
        thrusterPowerDraw=-100.0,
        imageAttErrorRequirement=0.1,
        imageRateErrorRequirement=0.1,
        disturbance_vector=np.array(
            [0.0, 0.0, 0.0]) if not random_disturbance else lambda: np.random.normal(scale=0.0001, size=3),
        maxWheelSpeed=6000.0,
        wheelSpeeds=np.array(
            [0.0, 0.0, 0.0]) if not random_RW_speed else lambda: np.random.uniform(-3000, 3000, 3),
        desatAttitude="nadir",
        oe=orbit_ls[2]
    )


    class ImagingSatellite(sats.ImagingSatellite):
        observation_spec = [
            obs.SatProperties(
                dict(prop="storage_level_fraction"),                        # 1
                dict(prop="battery_charge_fraction"),                       # 1
                dict(prop="wheel_speeds_fraction"),                         # 3
                # dict(prop="omega_BP_P", norm=0.03),
                # dict(prop="c_hat_P"),
                # dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
                # dict(prop="v_BN_P", norm=7616.5),
            ),
            # total 19 data

            obs.OpportunityProperties(
                dict(prop="priority"),
                # Cloud coverage forecast (percentage of the area covered by clouds)
                dict(fn=lambda sat, opp: opp["object"].cloud_cover_forecast),
                # Confidence on the cloud coverage forecast
                dict(fn=lambda sat, opp: opp["object"].cloud_cover_sigma),
                dict(prop="opportunity_open", norm=5700.0),
                type="target",
                n_ahead_observe=env_args.n_obs_image,
            ),
            obs.OpportunityProperties(
                dict(prop="opportunity_open", norm=5700),
                dict(prop="opportunity_close", norm=5700),
                type="ground_station",
                n_ahead_observe=1,
            ),
            obs.Eclipse(norm=5700),
            obs.Time(),
        ]
        action_spec = [
            act.Charge(duration=20.0),
            act.Downlink(duration=20.0),
            act.Desat(duration=20.0),
            # act.Drift(duration=1.0)
            act.Image(n_ahead_image=env_args.n_act_image),
        ]
        fsw_type = fsw.SteeringImagerFSWModel
        dyn_type = dyn.ManyGroundStationFullFeaturedDynModel


    class SARImagingSatellite(sats.SARImagingSatellite):
        observation_spec = [
            obs.SatProperties(
                dict(prop="storage_level_fraction"),                        # 1
                dict(prop="battery_charge_fraction"),                       # 1
                dict(prop="wheel_speeds_fraction"),                         # 3
                # dict(prop="omega_BP_P", norm=0.03),
                # dict(prop="c_hat_P"),
                # dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
                # dict(prop="v_BN_P", norm=7616.5),
            ),
            # total 19 data

            obs.OpportunityProperties(
                dict(prop="priority"),
                # Cloud coverage forecast (percentage of the area covered by clouds)
                dict(fn=lambda sat, opp: opp["object"].cloud_cover_forecast),
                # Confidence on the cloud coverage forecast
                dict(fn=lambda sat, opp: opp["object"].cloud_cover_sigma),
                dict(prop="opportunity_open", norm=5700.0),
                type="target",
                n_ahead_observe=2*env_args.n_obs_image, # 4 per obs image
            ),
            obs.OpportunityProperties(
                dict(prop="opportunity_open", norm=5700),       # dim = 1
                dict(prop="opportunity_close", norm=5700),      # dim = 1
                type="ground_station",
                n_ahead_observe=1,
            ),
            # obs.Eclipse(norm=5700), # dim = 2
            # obs.Time(),           # dim = 1
        ]
        action_spec = [
            act.Charge(duration=20.0),
            act.Downlink(duration=20.0),
            act.Desat(duration=20.0),
            # act.Drift(duration=1.0),
            act.Image(n_ahead_image=2*env_args.n_act_image),
        ]
        fsw_type = fsw.SteeringImagerFSWModel
        dyn_type = dyn.ManyGroundStationFullFeaturedDynModel


    opt1_sat = ImagingSatellite(satellite_names[0], opt1_sat_args)
    opt2_sat = ImagingSatellite(satellite_names[1], opt2_sat_args)
    sar_sat = SARImagingSatellite(satellite_names[2], sar_sat_args)

    multiSat = [opt1_sat, opt2_sat, sar_sat]
    duration = np.round(env_args.orbit_num * 5700.0)
    target_total = env_args.uniform_targets

    env = GeneralSatelliteTasking(
        satellites=multiSat,
        scenario=scene.UserDefOceanTargetswithCloud(target_total),
        # scenario=scene.RandomOceanTargetswithCloud(target_total) if args.randomize_enabled else scene.UserDefOceanTargetswithCloud(target_total),
        rewarder=data.UniqueImageSARReward(),
        time_limit=duration,
        communicator=comm.LOSCommunication(),
        log_level="WARNING",
        terminate_on_time_limit=True,
        failure_penalty=-100.0,
        # activate these lines to record visualization
        vizard_dir="./tmp/vizard" if env_args.use_render == True else None,
        vizard_settings=dict(showLocationLabels=-1, showLocationCommLines=1, showLocationCones=-1) if env_args.use_render == True else None,
    )
    
    return env