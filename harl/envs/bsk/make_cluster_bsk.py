from bsk_rl.utils.orbital import walker_delta_args
from bsk_rl.utils.orbital import random_orbit
from bsk_rl.sim import dyn, fsw
from bsk_rl import sats, act, obs, scene, data, comm
import numpy as np
from collections.abc import Iterable
import warnings
warnings.filterwarnings("ignore")


try:
    from bsk_rl import GeneralSatelliteTasking
except ImportError:
    warnings.warn(
        "BSK-RL is not installed, so these environments will not be available! Please double-check the installation of Basilisk and BSK-RL library first (`pip show basilisk` and `pip show bsk-rl`)"
    )


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
    for inc, offset in zip(inclination, true_anomaly_offsets):
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
        random_disturbance = False
        random_RW_speed = False

    elif optim_challenge == "hard":
        battery_size = env_args.battery_capacity
        init_battery_level = 80
        memory_size = env_args.memory_size
        init_memory_percent = 90
        baud_rate = int(0.3 * env_args.baud_rate)
        random_disturbance = False
        random_RW_speed = False
    else:
        print("Optimisation challenge name is not available")
        NotImplementedError

    randomize_target = env_args.randomize_enabled

    if randomness is not None:
        if randomness == "random_res":
            init_battery_level -= np.random.uniform(0,5) # drop the init randomly between 0-5%
            init_memory_percent += np.random.uniform(0,1) # increase init memory storage between 0-10%
            random_disturbance = True
            random_RW_speed = True
            randomize_target = False

        elif randomness == "random_batt":
            init_battery_level -= np.random.uniform(0,5) # drop the init randomly between 0-5%
            random_disturbance = False
            random_RW_speed = False
            randomize_target = False

        elif randomness == "random_mem":
            init_memory_percent += np.random.uniform(0,1) # increase init memory storage between 0-10%
            random_disturbance = False
            random_RW_speed = False
            randomize_target = False

        elif randomness == "random_dist":
            random_disturbance = True
            random_RW_speed = False
            randomize_target = False

        elif randomness == "random_rw":
            random_disturbance = False
            random_RW_speed = True
            randomize_target = False

        elif randomness == "random_res_n_target":
            init_battery_level -= np.random.uniform(0,5) # drop the init randomly between 0-5%
            init_memory_percent += np.random.uniform(0,1) # increase init memory storage between 0-10%
            random_disturbance = True
            random_RW_speed = True
            randomize_target = True
        else:
            print("Randomization name not available")
            NotImplementedError

    # Define four satellites in a "train" Cluster formation along the same orbit
    multiSat = []
    index = 0

    for orbit in orbit_ls:
        sat_args = dict(
            # Attitude Control:
            u_max=0.4,
            omega_max=0.1,
            servo_Ki=5.0,
            servo_P=150/5,
            K1=0.25,
            K3=3.0,
            imageAttErrorRequirement=0.1,
            imageRateErrorRequirement=0.1,
            imageTargetMinimumElevation=np.radians(
                83),  # np.arctan(800 / 500),
            # Memory:
            dataStorageCapacity=memory_size * 8e6,
            storageInit=int(memory_size * init_memory_percent/100),
            # Instruments:
            instrumentBaudRate=env_args.instr_baud_rate * 1e6,  # 1Mbps
            transmitterBaudRate=-1*baud_rate * 1e6,
            # Power:
            batteryStorageCapacity=battery_size * 3600,
            storedCharge_Init=int(battery_size * 3600 * init_battery_level / 100),
            panelArea=1.0,
            panelEfficiency=20.0,
            basePowerDraw=-10.0,
            instrumentPowerDraw=-30.0,
            transmitterPowerDraw=-25.0,
            thrusterPowerDraw=-80.0,
            # Reaction Wheels:
            disturbance_vector=np.array(
                [0.0, 0.0, 0.0]) if not random_disturbance else lambda: np.random.normal(scale=0.0001, size=3),
            maxWheelSpeed=6000.0,
            wheelSpeeds=np.array(
                [0.0, 0.0, 0.0]) if not random_RW_speed else lambda: np.random.uniform(-3000, 3000, 3),
            desatAttitude="nadir",
            oe=orbit
        )

        # Define Imaging Satellite Object:
        class ImagingSatellite(sats.ImagingSatellite):
            observation_spec = [
                obs.SatProperties(
                    # 1
                    dict(prop="storage_level_fraction"),
                    # 1
                    dict(prop="battery_charge_fraction"),
                    # 3
                    dict(prop="wheel_speeds_fraction"),
                    # dict(prop="omega_BP_P", norm=0.03),
                    # dict(prop="c_hat_P"),
                    # dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
                    # dict(prop="v_BN_P", norm=7616.5),
                ),

                obs.OpportunityProperties(
                    dict(prop="priority"),
                    # Cloud coverage forecast (percentage of the area covered by clouds)
                    dict(fn=lambda sat,
                         opp: opp["object"].cloud_cover_forecast),
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
    target_total = env_args.uniform_targets

    env = GeneralSatelliteTasking(
        satellites=multiSat,
        scenario=scene.RandomOceanTargetswithCloud(
            target_total) if randomize_target else scene.UserDefOceanTargetswithCloud(target_total),
        rewarder=data.UniqueImageSARReward(),
        time_limit=duration,
        communicator=comm.LOSCommunication(),
        log_level="ERROR",
        terminate_on_time_limit=True,
        failure_penalty=env_args.failure_penalty,
        vizard_dir="./tmp/vizard" if env_args.use_render == True else None,
        vizard_settings=dict(showLocationLabels=-1, showLocationCommLines=1,
                             showLocationCones=-1) if env_args.use_render == True else None,
    )
    return env


def make_BSK_Walker_env(env_args, satellite_names, scenario):
    # Define four satellites in walker delta orbits
    sat_arg_randomizer = walker_delta_args(
        altitude=500.0, inc=50.0, n_planes=env_args.n_satellites, randomize_lan=False, randomize_true_anomaly=False)

    if scenario == "ideal":
        battery_sizes = [1e6]*len(satellite_names)
        memory_size = 1e6
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = env_args.baud_rate
        instr_baud_rate = 500

    elif scenario == "limited":
        battery_sizes = [int(env_args.battery_capacity/4)
                         ]*len(satellite_names)
        memory_size = int(env_args.memory_size/20)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = env_args.baud_rate
        instr_baud_rate = env_args.instr_baud_rate

    elif scenario == "default":
        battery_sizes = [int(env_args.battery_capacity)
                         ]*len(satellite_names)
        memory_size = int(env_args.memory_size)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = env_args.baud_rate
        instr_baud_rate = env_args.instr_baud_rate

    elif scenario == "random":
        battery_sizes = [env_args.battery_capacity]*len(satellite_names)
        memory_size = env_args.memory_size
        random_init_memory = True
        random_init_battery = True
        random_disturbance = True
        random_RW_speed = True
        baud_rate = env_args.baud_rate
        instr_baud_rate = env_args.instr_baud_rate

    elif scenario == "limited_all":
        battery_sizes = [50]*len(satellite_names)
        memory_sizes = [5000]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = 0.5
        instr_baud_rate = 125

    else:
        print("Scenario name not available")
        NotImplementedError

    # Define four satellites in a "train" Cluster formation along the same orbit
    multiSat = []
    index = 0
    for battery_size, memory_size in zip(battery_sizes, memory_sizes):
        sat_args = dict(
            # Power
            batteryStorageCapacity=battery_size * 3600,
            storedCharge_Init=int(battery_size * env_args.init_battery_level / 100 * 3600) if not random_init_battery else np.random.uniform(
                battery_size * 3600 * 0.4, battery_size * 3600 * 0.5),
            panelArea=1.0,
            panelEfficiency=20.0,
            basePowerDraw=-10.0,
            instrumentPowerDraw=-30.0,
            transmitterPowerDraw=-25.0,
            thrusterPowerDraw=-80.0,
            # Data Storage
            dataStorageCapacity=memory_size * 8e6,  # MB to bits,
            storageInit=int(memory_size *
                            env_args.init_memory_percent/100) * 8e6 if not random_init_memory else np.random.uniform(memory_size * 8e6 * 0.2, memory_size * 8e6 * 0.8),
            instrumentBaudRate=instr_baud_rate * 1e6,
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
        )

        class ImagingSatellite(sats.ImagingSatellite):
            observation_spec = [
                obs.SatProperties(
                    dict(prop="storage_level_fraction"),
                    dict(prop="battery_charge_fraction"),
                    dict(prop="wheel_speeds_fraction"),

                ),
                obs.Eclipse(),
                obs.OpportunityProperties(
                    dict(prop="priority"),
                    dict(prop="opportunity_open", norm=5700.0),
                    n_ahead_observe=env_args.n_obs_image,
                ),
                obs.OpportunityProperties(
                    dict(prop="opportunity_open", norm=5700),
                    dict(prop="opportunity_close", norm=5700),
                    type="ground_station",
                    n_ahead_observe=1,
                ),
                obs.Time(),
            ]
            action_spec = [act.Image(n_ahead_image=env_args.n_act_image),
                           act.Downlink(duration=20.0),
                           act.Desat(duration=20.0),
                           act.Charge(duration=20.0),
                           ]
            dyn_type = dyn.ManyGroundStationFullFeaturedDynModel
            fsw_type = fsw.SteeringImagerFSWModel

        sat = ImagingSatellite(f"EO-{index}", sat_args)
        multiSat.append(sat)
        index += 1

    duration = env_args.orbit_num * 5700.0  # About 2 orbits

    env = GeneralSatelliteTasking(
        satellites=multiSat,
        scenario=scene.UniformTargets(env_args.uniform_targets),
        rewarder=data.UniqueImageReward(),
        time_limit=duration,
        # Note that dyn must inherit from LOSCommunication
        communicator=comm.LOSCommunication(),
        sat_arg_randomizer=sat_arg_randomizer,
        log_level="WARNING",
        terminate_on_time_limit=True,
        failure_penalty=env_args.failure_penalty,
        vizard_dir="./tmp_cluster/vizard" if env_args.use_render else None,
        vizard_settings=dict(showLocationLabels=-
                             1) if env_args.use_render else None,
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
    for offset, inc in zip(true_anomaly_offsets, inclination):
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
        random_disturbance = False
        random_RW_speed = False

    elif optim_challenge == "hard":
        battery_size = env_args.battery_capacity
        init_battery_level = 85
        memory_size = env_args.memory_size
        init_memory_percent = 90
        baud_rate = int(0.3 * env_args.baud_rate)
        random_disturbance = False
        random_RW_speed = False
    else:
        print("Optimisation challenge name is not available")
        NotImplementedError


    if randomness is not None:
        if randomness == "random_res":
            init_battery_level -= np.random.uniform(0,5) # drop the init randomly between 0-5%
            init_memory_percent += np.random.uniform(0,10) # increase init memory storage between 0-10%
            random_disturbance = True
            random_RW_speed = True
            # randomize_target = False

        elif randomness == "random_batt":
            init_battery_level -= np.random.uniform(0,5) # drop the init randomly between 0-5%
            random_disturbance = False
            random_RW_speed = False
            # randomize_target = False

        elif randomness == "random_mem":
            init_memory_percent += np.random.uniform(0,10) # increase init memory storage between 0-10%
            random_disturbance = False
            random_RW_speed = False
            # randomize_target = False

        elif randomness == "random_dist":
            random_disturbance = True
            random_RW_speed = False
            # randomize_target = False

        elif randomness == "random_rw":
            random_disturbance = False
            random_RW_speed = True
            # randomize_target = False

        # elif randomness == "random_res_n_target":
        #     init_battery_level -= np.random.uniform(0,5) # drop the init randomly between 0-5%
        #     init_memory_percent += np.random.uniform(0,1) # increase init memory storage between 0-10%
        #     random_disturbance = True
        #     random_RW_speed = True
        #     randomize_target = True
        else:
            print("Randomization name not available")
            NotImplementedError

    randomize_target = env_args.tgt_randomize_enabled

    # Define the Optical-1 satellite arguments
    opt1_sat_args = dict(
        u_max=0.4,
        omega_max=0.1,
        servo_Ki=5.0,
        servo_P=150/5,
        K1=0.25,
        K3=3.0,
        imageTargetMinimumElevation=np.radians(83),  # np.arctan(800 / 500),
        dataStorageCapacity=memory_size * 8e6,
        storageInit=int(memory_size * init_memory_percent/100) * 8e6,
        instrumentBaudRate=env_args.instr_baud_rate * 1e6,  # 1Mbps
        transmitterBaudRate=-1*baud_rate * 1e6,
        batteryStorageCapacity=battery_size * 3600,
        storedCharge_Init=int(battery_size * 3600 * init_battery_level / 100),
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
        imageTargetMinimumElevation=np.radians(83),  # np.arctan(800 / 500),
        dataStorageCapacity=memory_size * 8e6,
        storageInit=int(memory_size * init_memory_percent/100) * 8e6,
        instrumentBaudRate=env_args.instr_baud_rate * 1e6,  # 1Mbps
        transmitterBaudRate=-1*baud_rate * 1e6,
        batteryStorageCapacity=battery_size * 3600,
        storedCharge_Init=int(battery_size * 3600 * init_battery_level / 100),
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
        storageInit=int(2*memory_size * init_memory_percent/100) * 8e6,
        instrumentBaudRate=2 * env_args.instr_baud_rate * 1e6,  # 1Mbps
        transmitterBaudRate=-1*baud_rate * 1e6,
        batteryStorageCapacity=2*battery_size * 3600,
        storedCharge_Init=int(2*battery_size * 3600 * init_battery_level / 100),
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
                n_ahead_observe=2*env_args.n_obs_image,  # 4 per obs image
            ),
            obs.OpportunityProperties(
                dict(prop="opportunity_open", norm=5700),       # dim = 1
                dict(prop="opportunity_close", norm=5700),      # dim = 1
                type="ground_station",
                n_ahead_observe=1,
            ),
            obs.Eclipse(norm=5700), # dim = 2
            obs.Time(),           # dim = 1
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
        world_args = dict(utc_init='2012 JUN 01 03:03:39.009 (UTC)'),
        scenario=scene.RandomOceanTargetswithCloud(
            target_total) if randomize_target else scene.UserDefOceanTargetswithCloud(target_total),
        rewarder=data.UniqueImageSARReward(),
        time_limit=duration,
        communicator=comm.LOSCommunication(),
        log_level="Error",
        terminate_on_time_limit=True,
        failure_penalty=env_args.failure_penalty,
        # activate these lines to record visualization
        vizard_dir="./tmp/vizard" if env_args.use_render == True else None,
        vizard_settings=dict(showLocationLabels=-1, showLocationCommLines=1,
                             showLocationCones=-1) if env_args.use_render == True else None,
    )

    return env


def make_BSK_FLOCK_env(env_args, optim_challenge, randomness=None):
    satellite_names = []
    satellite_names.append(f"OPT-FLOCK 4Q-34")
    satellite_names.append(f"OPT-FLOCK 4Q-35")
    satellite_names.append(f"OPT-FLOCK 4Q-36")

    # From WP-4:
    inc = [97.4036, 97.4058, 97.4051]
    # km, fixed for all satellites
    altitude = [6807.564685, 6815.740163, 6812.376574]
    eccentricity = [0.0008045, 0.0008793, 0.0007742]  # Circular orbit
    # Longitude of Ascending Node (Omega), fixed for all
    LAN = [48.8444, 48.5746, 48.5827]
    # Argument of Periapsis (omega), fixed for all
    arg_periapsis = [159.6904, 170.914, 175.8923]
    offset = [200.4666, 189.2268, 184.239]

    orbit_ls = []
    for i in range(len(satellite_names)):
        orbit = random_orbit(
            i=inc[i], alt=500, e=eccentricity[i], Omega=LAN[i], omega=arg_periapsis[i], f=offset[i]
        )
        orbit.a = altitude[i] * 1000
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

    randomize_target = env_args.randomize_enabled

    if randomness is not None:
        if randomness == "random_res":
            random_init_memory = True
            random_init_battery = True
            random_disturbance = True
            random_RW_speed = True
            randomize_target = False

        elif randomness == "random_batt":
            random_init_memory = False
            random_init_battery = True
            random_disturbance = False
            random_RW_speed = False
            randomize_target = False

        elif randomness == "random_mem":
            random_init_memory = True
            random_init_battery = False
            random_disturbance = False
            random_RW_speed = False
            randomize_target = False

        elif randomness == "random_dist":
            random_init_memory = False
            random_init_battery = False
            random_disturbance = True
            random_RW_speed = False
            randomize_target = False

        elif randomness == "random_rw":
            random_init_memory = False
            random_init_battery = False
            random_disturbance = False
            random_RW_speed = True
            randomize_target = False

        elif randomness == "random_res_n_target":
            random_init_memory = True
            random_init_battery = True
            random_disturbance = True
            random_RW_speed = True
            randomize_target = True
        else:
            print("Randomization name not available")
            NotImplementedError

    # Define four satellites in a "train" Cluster formation along the same orbit
    multiSat = []
    index = 0

    for orbit in orbit_ls:
        sat_args = dict(
            # Attitude Control:
            u_max=0.4,
            omega_max=0.1,
            servo_Ki=5.0,
            servo_P=150/5,
            K1=0.25,
            K3=3.0,
            imageAttErrorRequirement=0.01,
            imageRateErrorRequirement=0.01,
            imageTargetMinimumElevation=np.radians(
                83),  # np.arctan(800 / 500),
            # Memory:
            dataStorageCapacity=memory_size * 8e6,
            storageInit=int(memory_size * init_memory_percent/100) * 8e6 if not random_init_memory
            else np.random.uniform(memory_size * 8e6 * 0.6, memory_size * 8e6 * 0.8),
            # Instruments:
            instrumentBaudRate=env_args.instr_baud_rate * 1e6,  # 1Mbps
            transmitterBaudRate=-1*baud_rate * 1e6,
            # Power:
            batteryStorageCapacity=battery_size * 3600,
            storedCharge_Init=int(battery_size * 3600 * init_battery_level / 100) if not random_init_battery
            else np.random.uniform(battery_size * 3600 * 0.8, battery_size * 3600 * 0.95),
            panelArea=1.0,
            panelEfficiency=20.0,
            basePowerDraw=-10.0,
            instrumentPowerDraw=-30.0,
            transmitterPowerDraw=-25.0,
            thrusterPowerDraw=-80.0,
            # Reaction Wheels:
            disturbance_vector=np.array(
                [0.0, 0.0, 0.0]) if not random_disturbance else lambda: np.random.normal(scale=0.0001, size=3),
            maxWheelSpeed=6000.0,
            wheelSpeeds=np.array(
                [0.0, 0.0, 0.0]) if not random_RW_speed else lambda: np.random.uniform(-3000, 3000, 3),
            desatAttitude="nadir",
            oe=orbit
        )

        # Define Imaging Satellite Object:
        class ImagingSatellite(sats.ImagingSatellite):
            observation_spec = [
                obs.SatProperties(
                    # 1
                    dict(prop="storage_level_fraction"),
                    # 1
                    dict(prop="battery_charge_fraction"),
                    # 3
                    dict(prop="wheel_speeds_fraction"),
                    # dict(prop="omega_BP_P", norm=0.03),
                    # dict(prop="c_hat_P"),
                    # dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
                    # dict(prop="v_BN_P", norm=7616.5),
                ),

                obs.OpportunityProperties(
                    dict(prop="priority"),
                    # Cloud coverage forecast (percentage of the area covered by clouds)
                    dict(fn=lambda sat,
                         opp: opp["object"].cloud_cover_forecast),
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
                act.Charge(duration=5.0),
                act.Downlink(duration=5.0),
                act.Desat(duration=5.0),
                act.Drift(duration=5.0),
                act.Image(n_ahead_image=env_args.n_act_image),
            ]
            fsw_type = fsw.SteeringImagerFSWModel
            dyn_type = dyn.ManyGroundStationFullFeaturedDynModel

        sat = ImagingSatellite(satellite_names[index], sat_args)
        multiSat.append(sat)
        index += 1

    duration = env_args.orbit_num * 5700.0  # About 2 orbits
    target_total = env_args.uniform_targets

    env = GeneralSatelliteTasking(
        satellites=multiSat,
        scenario=scene.UniformTargetswithRandomCloud(
            target_total) if randomize_target else scene.UserDefOceanTargetswithCloud(target_total),
        rewarder=data.UniqueImageReward(),
        time_limit=duration,
        communicator=comm.LOSCommunication(),
        log_level="ERROR",
        terminate_on_time_limit=True,
        failure_penalty=env_args.failure_penalty,
        vizard_dir="./tmp/vizard" if env_args.use_render == True else None,
        vizard_settings=dict(showLocationLabels=-1, showLocationCommLines=1,
                             showLocationCones=-1) if env_args.use_render == True else None,
    )
    return env


def make_BSK_SAR_OPT_CLOUD_env(env_args, optim_challenge, randomness=None):
    satellite_names = []
    satellite_names.append(f"OPT-1-Sat")
    satellite_names.append(f"OPT-2-Sat")
    satellite_names.append(f"OPT-3-Sat")
    satellite_names.append(f"SAR-Sat")

    # inclination = -15.0  # degrees, fixed for all satellites
    # degrees, fixed for all satellites
    inclination = [-15, -14.0, -16.0, -15.0]
    # inclination = [39.0, 41.0, 40.0]
    altitude = 500  # km, fixed for all satellites
    eccentricity = 0  # Circular orbit
    LAN = 0  # Longitude of Ascending Node (Omega), fixed for all
    arg_periapsis = 0  # Argument of Periapsis (omega), fixed for all
    # offset = 225
    # true_anomaly_offsets = [
    #     30-i*1 for i in range(len(satellite_names))]  # degrees
    true_anomaly_offsets = [31, 30, 30, 29]
    # true_anomaly_offsets = [-74, -74, -75]

    orbit_ls = []
    for offset, inc in zip(true_anomaly_offsets, inclination):
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
        baud_rate = int(0.5 * env_args.baud_rate)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
    else:
        print("Optimisation challenge name is not available")
        NotImplementedError

    randomize_target = env_args.randomize_enabled

    if randomness is not None:
        if randomness == "random_res":
            random_init_memory = True
            random_init_battery = True
            random_disturbance = True
            random_RW_speed = True
            randomize_target = False

        elif randomness == "random_batt":
            random_init_memory = False
            random_init_battery = True
            random_disturbance = False
            random_RW_speed = False
            randomize_target = False

        elif randomness == "random_mem":
            random_init_memory = True
            random_init_battery = False
            random_disturbance = False
            random_RW_speed = False
            randomize_target = False

        elif randomness == "random_dist":
            random_init_memory = False
            random_init_battery = False
            random_disturbance = True
            random_RW_speed = False
            randomize_target = False

        elif randomness == "random_rw":
            random_init_memory = False
            random_init_battery = False
            random_disturbance = False
            random_RW_speed = True
            randomize_target = False

        elif randomness == "random_res_n_target":
            random_init_memory = True
            random_init_battery = True
            random_disturbance = True
            random_RW_speed = True
            randomize_target = True
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
        imageTargetMinimumElevation=np.arctan(800 / 500),
        dataStorageCapacity=memory_size * 8e6,
        storageInit=int(memory_size * init_memory_percent/100) *
        8e6 if not random_init_memory else np.random.uniform(
            memory_size * 8e6 * 0.6, memory_size * 8e6 * 0.8),
        instrumentBaudRate=env_args.instr_baud_rate * 1e6,  # 1Mbps
        transmitterBaudRate=-1*baud_rate * 1e6,
        batteryStorageCapacity=battery_size * 3600,
        storedCharge_Init=int(battery_size * 3600 * init_battery_level / 100) if not random_init_battery else np.random.uniform(
            battery_size * 3600 * 0.8, battery_size * 3600 * 0.95),
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
        imageTargetMinimumElevation=np.radians(83),  # np.arctan(800 / 500),
        dataStorageCapacity=memory_size * 8e6,
        storageInit=int(memory_size * init_memory_percent/100) *
        8e6 if not random_init_memory else np.random.uniform(
            memory_size * 8e6 * 0.6, memory_size * 8e6 * 0.8),
        instrumentBaudRate=env_args.instr_baud_rate * 1e6,  # 1Mbps
        transmitterBaudRate=-1*baud_rate * 1e6,
        batteryStorageCapacity=battery_size * 3600,
        storedCharge_Init=int(battery_size * 3600 * init_battery_level / 100) if not random_init_battery else np.random.uniform(
            battery_size * 3600 * 0.8, battery_size * 3600 * 0.95),
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

    # Define the Optical-3 satellite arguments
    opt3_sat_args = dict(
        u_max=0.4,
        omega_max=0.1,
        servo_Ki=5.0,
        servo_P=150/5,
        K1=0.25,
        K3=3.0,
        imageTargetMinimumElevation=np.radians(83),  # np.arctan(800 / 500),
        dataStorageCapacity=memory_size * 8e6,
        storageInit=int(memory_size * init_memory_percent/100) *
        8e6 if not random_init_memory else np.random.uniform(
            memory_size * 8e6 * 0.6, memory_size * 8e6 * 0.8),
        instrumentBaudRate=env_args.instr_baud_rate * 1e6,  # 1Mbps
        transmitterBaudRate=-1*baud_rate * 1e6,
        batteryStorageCapacity=battery_size * 3600,
        storedCharge_Init=int(battery_size * 3600 * init_battery_level / 100) if not random_init_battery else np.random.uniform(
            battery_size * 3600 * 0.8, battery_size * 3600 * 0.95),
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
        oe=orbit_ls[2]
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
        storageInit=int(2*memory_size * init_memory_percent/100) *
        8e6 if not random_init_memory else np.random.uniform(
            2*memory_size * 8e6 * 0.6, 2*memory_size * 8e6 * 0.8),
        instrumentBaudRate=2 * env_args.instr_baud_rate * 1e6,  # 1Mbps
        transmitterBaudRate=-1*baud_rate * 1e6,
        batteryStorageCapacity=2*battery_size * 3600,
        storedCharge_Init=int(2*battery_size * 3600 * init_battery_level / 100) if not random_init_battery else np.random.uniform(
            2*battery_size * 3600 * 0.8, 2*battery_size * 3600 * 0.95),
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
        oe=orbit_ls[3]
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
            act.Charge(duration=5.0),
            act.Downlink(duration=5.0),
            act.Desat(duration=5.0),
            # act.Drift(duration=5.0),
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
                n_ahead_observe=2*env_args.n_obs_image,  # 4 per obs image
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
            act.Charge(duration=5.0),
            act.Downlink(duration=5.0),
            act.Desat(duration=5.0),
            # act.Drift(duration=5.0),
            act.Image(n_ahead_image=2*env_args.n_act_image),
        ]
        fsw_type = fsw.SteeringImagerFSWModel
        dyn_type = dyn.ManyGroundStationFullFeaturedDynModel

    opt1_sat = ImagingSatellite(satellite_names[0], opt1_sat_args)
    opt2_sat = ImagingSatellite(satellite_names[1], opt2_sat_args)
    opt3_sat = ImagingSatellite(satellite_names[2], opt3_sat_args)
    sar_sat = SARImagingSatellite(satellite_names[3], sar_sat_args)

    multiSat = [opt1_sat, opt2_sat, opt3_sat, sar_sat]
    duration = np.round(env_args.orbit_num * 5700.0)
    target_total = env_args.uniform_targets

    env = GeneralSatelliteTasking(
        satellites=multiSat,
        scenario=scene.RandomOceanTargetswithCloud(
            target_total) if randomize_target else scene.UserDefAUOceanTargetswithCloud(target_total),
        rewarder=data.UniqueCloudDetectImageSARReward(),
        time_limit=duration,
        communicator=comm.LOSCommunication(),
        log_level="Error",
        terminate_on_time_limit=True,
        failure_penalty=env_args.failure_penalty,
        # activate these lines to record visualization
        vizard_dir="./tmp/vizard" if env_args.use_render == True else None,
        vizard_settings=dict(showLocationLabels=-1, showLocationCommLines=1,
                             showLocationCones=-1) if env_args.use_render == True else None,
    )

    return env




def make_BSK_MULTI_CLS_env(env_args, optim_challenge, randomness=None):
    
    n_clusters = env_args.n_cluster
    lead = 22.5 * n_clusters
    
    satellite_names = []
    cluster_orbit = []

    for i in range(n_clusters):
        satellite_names.append([f"CLS-{i}-OPT-1-Sat",f"CLS-{i}-OPT-2-Sat",f"CLS-{i}-SAR-Sat"])

        inclination = [39.0, 41.0, 40.0]
        altitude = 500  # km, fixed for all satellites
        eccentricity = 0  # Circular orbit
        LAN = 0  # Longitude of Ascending Node (Omega), fixed for all
        arg_periapsis = 0  # Argument of Periapsis (omega), fixed for all
        true_anomaly_offsets = [-74 + lead*i, -74 + lead*i, -75 + lead*i]

        orbit_ls = []
        for offset, inc in zip(true_anomaly_offsets, inclination):
            orbit = random_orbit(
                i=inc, alt=altitude, e=eccentricity, Omega=LAN, omega=arg_periapsis, f=offset
            )
            orbit_ls.append(orbit)
        cluster_orbit.append(orbit_ls)

    if optim_challenge == "easy":
        battery_size = env_args.battery_capacity
        init_battery_level = env_args.init_battery_level
        memory_size = env_args.memory_size
        init_memory_percent = env_args.init_memory_percent
        baud_rate = env_args.baud_rate
        random_disturbance = False
        random_RW_speed = False

    elif optim_challenge == "hard":
        battery_size = env_args.battery_capacity
        init_battery_level = 85
        memory_size = env_args.memory_size
        init_memory_percent = 90
        baud_rate = int(0.3 * env_args.baud_rate)
        random_disturbance = False
        random_RW_speed = False
    else:
        print("Optimisation challenge name is not available")
        NotImplementedError


    if randomness is not None:
        if randomness == "random_res":
            init_battery_level -= np.random.uniform(0,5) # drop the init randomly between 0-5%
            init_memory_percent += np.random.uniform(0,10) # increase init memory storage between 0-10%
            random_disturbance = True
            random_RW_speed = True
            # randomize_target = False

        elif randomness == "random_batt":
            init_battery_level -= np.random.uniform(0,5) # drop the init randomly between 0-5%
            random_disturbance = False
            random_RW_speed = False
            # randomize_target = False

        elif randomness == "random_mem":
            init_memory_percent += np.random.uniform(0,10) # increase init memory storage between 0-10%
            random_disturbance = False
            random_RW_speed = False
            # randomize_target = False

        elif randomness == "random_dist":
            random_disturbance = True
            random_RW_speed = False
            # randomize_target = False

        elif randomness == "random_rw":
            random_disturbance = False
            random_RW_speed = True
            # randomize_target = False

        # elif randomness == "random_res_n_target":
        #     init_battery_level -= np.random.uniform(0,5) # drop the init randomly between 0-5%
        #     init_memory_percent += np.random.uniform(0,1) # increase init memory storage between 0-10%
        #     random_disturbance = True
        #     random_RW_speed = True
        #     randomize_target = True
        else:
            print("Randomization name not available")
            NotImplementedError

    randomize_target = env_args.tgt_randomize_enabled

    

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
                n_ahead_observe=2*env_args.n_obs_image,  # 4 per obs image
            ),
            obs.OpportunityProperties(
                dict(prop="opportunity_open", norm=5700),       # dim = 1
                dict(prop="opportunity_close", norm=5700),      # dim = 1
                type="ground_station",
                n_ahead_observe=1,
            ),
            obs.Eclipse(norm=5700), # dim = 2
            obs.Time(),           # dim = 1
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

    multiSat = []
    for i in range(n_clusters):
        # Define the Optical-1 satellite arguments
        opt1_sat_args = dict(
            u_max=0.4,
            omega_max=0.1,
            servo_Ki=5.0,
            servo_P=150/5,
            K1=0.25,
            K3=3.0,
            imageTargetMinimumElevation=np.radians(83),  # np.arctan(800 / 500),
            dataStorageCapacity=memory_size * 8e6,
            storageInit=int(memory_size * init_memory_percent/100) * 8e6,
            instrumentBaudRate=env_args.instr_baud_rate * 1e6,  # 1Mbps
            transmitterBaudRate=-1*baud_rate * 1e6,
            batteryStorageCapacity=battery_size * 3600,
            storedCharge_Init=int(battery_size * 3600 * init_battery_level / 100),
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
            oe=cluster_orbit[i][0]
        )

        # Define the Optical-2 satellite arguments
        opt2_sat_args = dict(
            u_max=0.4,
            omega_max=0.1,
            servo_Ki=5.0,
            servo_P=150/5,
            K1=0.25,
            K3=3.0,
            imageTargetMinimumElevation=np.radians(83),  # np.arctan(800 / 500),
            dataStorageCapacity=memory_size * 8e6,
            storageInit=int(memory_size * init_memory_percent/100) * 8e6,
            instrumentBaudRate=env_args.instr_baud_rate * 1e6,  # 1Mbps
            transmitterBaudRate=-1*baud_rate * 1e6,
            batteryStorageCapacity=battery_size * 3600,
            storedCharge_Init=int(battery_size * 3600 * init_battery_level / 100),
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
            oe=cluster_orbit[i][1]
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
            storageInit=int(2*memory_size * init_memory_percent/100) * 8e6,
            instrumentBaudRate=2 * env_args.instr_baud_rate * 1e6,  # 1Mbps
            transmitterBaudRate=-1*baud_rate * 1e6,
            batteryStorageCapacity=2*battery_size * 3600,
            storedCharge_Init=int(2*battery_size * 3600 * init_battery_level / 100),
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
            oe=cluster_orbit[i][2]
        )

        opt1_sat = ImagingSatellite(satellite_names[i][0], opt1_sat_args)
        opt2_sat = ImagingSatellite(satellite_names[i][1], opt2_sat_args)
        sar_sat = SARImagingSatellite(satellite_names[i][2], sar_sat_args)

        multiSat.append(opt1_sat)
        multiSat.append(opt2_sat)
        multiSat.append(sar_sat)

    duration = np.round(env_args.orbit_num * 5700.0)
    target_total = env_args.uniform_targets

    env = GeneralSatelliteTasking(
        satellites=multiSat,
        world_args = dict(utc_init='2012 JUN 01 03:03:39.009 (UTC)'),
        scenario=scene.RandomOceanTargetswithCloud(
            target_total) if randomize_target else scene.UserDefOceanTargetswithCloud(target_total),
        rewarder=data.UniqueImageSARReward(),
        time_limit=duration,
        communicator=comm.LOSCommunication(),
        log_level="Error",
        terminate_on_time_limit=True,
        failure_penalty=env_args.failure_penalty,
        # activate these lines to record visualization
        vizard_dir="./tmp/vizard" if env_args.use_render == True else None,
        vizard_settings=dict(showLocationLabels=-1, showLocationCommLines=1,
                             showLocationCones=-1) if env_args.use_render == True else None,
    )

    return env
