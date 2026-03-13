from harl.envs.bsk.make_cluster_bsk import make_BSK_Cluster_env, make_BSK_Walker_env, make_BSK_SAR_OPT_env
from munch import Munch
import copy
import numpy as np
from gymnasium.spaces import flatdim
from gymnasium.spaces import Discrete, Box
from harl.envs.bsk.make_single_sat_bsk import make_BSK_SingleSat_env
from harl.envs.bsk.make_cluster_bsk import make_BSK_SAR_OPT_CLOUD_env
from harl.envs.bsk.make_cluster_bsk import make_BSK_FLOCK_env
from harl.envs.bsk.make_cluster_bsk import make_BSK_MULTI_CLS_env
import warnings
warnings.filterwarnings("ignore")


class ClusterbskEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        env_args = Munch.fromDict(self.args)
        bsk_scenario = env_args.key.split('-')[0]

        if len(env_args.key.split('-')) > 1:
            task_challenge = env_args.key.split('-')[1]
            randomness_key = None

            if len(env_args.key.split('-')) == 3:
                randomness_key = env_args.key.split('-')[2]
        else:
            randomness_key = None

        if bsk_scenario == "single_sat":
            self.env = make_BSK_SingleSat_env(
                env_args, task_challenge, randomness_key)
            print("Running BSK-ENV with single satellite scenario")

        elif bsk_scenario == "walker":
            self.env = make_BSK_Walker_env(
                env_args, self.satellite_names, bsk_scenario[1])
            print("Running BSK-ENV with walker-delta scenario")

        elif bsk_scenario == "hmg_cluster":
            self.env = make_BSK_Cluster_env(
                env_args, task_challenge, randomness_key)
            print("Running BSK-ENV with 3 Optical satellites cluster scenario")

        elif bsk_scenario == "het_cluster":
            self.env = make_BSK_SAR_OPT_env(
                env_args, task_challenge, randomness_key)
            print("Running BSK-ENV with 1 SAR and 2 OPTICAL satellites cluster scenario")

        elif bsk_scenario == "hmg_flock":
            self.env = make_BSK_FLOCK_env(
                env_args, task_challenge, randomness_key)
            print("Running BSK-ENV with FLOCK OPTICAL satellites cluster scenario")

        elif bsk_scenario == "het_cloud_cluster":
            self.env = make_BSK_SAR_OPT_CLOUD_env(
                env_args, task_challenge, randomness_key)
            print(
                "Running BSK-ENV with 1-Cloud detector, 2-OPTICAL, 1 SAR satellites cluster scenario")
        
        elif bsk_scenario == "multi_het_cluster":
            self.env = make_BSK_MULTI_CLS_env(
                env_args, task_challenge, randomness_key)
            print(
                "Running BSK-ENV with Multiple Cluster consists of 2-OPTICAL and 1 SAR satellites scenario")

        else:
            print("Scenario name not available")
            NotImplementedError
        
        if bsk_scenario == "het_cloud_cluster":
            self.satellite_names = []
            self.satellite_names.append(f"OPT-1-Sat")
            self.satellite_names.append(f"OPT-2-Sat")
            self.satellite_names.append(f"OPT-3-Sat")
            self.satellite_names.append(f"SAR-Sat")

        elif bsk_scenario == "multi_het_cluster":
            self.satellite_names = []
            for i in range(env_args.n_cluster):
                self.satellite_names.append(f"CLS-{i+1}-OPT-1-Sat")
                self.satellite_names.append(f"CLS-{i+1}-OPT-2-Sat")
                self.satellite_names.append(f"CLS-{i+1}-SAR-Sat")
        else:
            self.satellite_names = []
            for i in range(env_args.n_satellites):
                self.satellite_names.append(f"Sat-{i}")

        self.longest_action_space = max(
            self.env.action_space, key=lambda x: x.n)
        self.action_names = []
        self.action_names.append("Charge")
        self.action_names.append("Downlink")
        self.action_names.append("Desaturate")
        for i in range((self.longest_action_space.n)-3):
            self.action_names.append(f"Image_Target_{i}")

        self.n_agents = len(self.satellite_names)
        self.downlinked = {}
        self._info = {}
        self._obs = None
        # self._obs = [sat.get_obs() for sat in self.env.satellites]
        # self._obs = self._pad_observation(self._obs)

        for sat in self.satellite_names:
            for action_name in self.action_names:
                self._info[f'{sat}-{action_name}'] = 0.0
            self.downlinked[f'{sat}'] = 0.0
        self.img_cost = 0.0
        self.imaged = 0

        self.power_reward = env_args.power_reward
        self.battery_cost_scale = env_args.battery_cost_scale
        self.data_reward = env_args.data_reward
        self.data_cost_scale = env_args.data_cost_scale
        self.min_batt_level_penalty = env_args.min_batt_level_penalty

        self._past_obs = self._obs

        self.longest_observation_space = max(
            self.env.observation_space, key=lambda x: x.shape
        )
        self.batt_idx = 1
        self.mem_idx = 0
        if self.args['transformer_obs']:
            from gymnasium import spaces
            low         = self.longest_observation_space.low[0]
            high        = self.longest_observation_space.high[0]
            size        = self.longest_observation_space.shape[0] + 2
            data_type   = self.longest_observation_space.dtype
            self.longest_observation_space = spaces.Box(
                low=low, high=high, shape=(size,), dtype=data_type
            )
            self.batt_idx = 0
            self.mem_idx = 4

        self.observation_space = [self.longest_observation_space]*self.n_agents
        self.share_observation_space = [self.get_state_size()]
        self.action_space = [self.longest_action_space]*self.n_agents

        if self.env.action_space.__class__.__name__ == "Box":
            self.discrete = False
        else:
            self.discrete = True
        self.avail_actions = self.get_avail_actions()

    def _pad_observation(self, obs):
        if self.args['transformer_obs']:
            obs_sizes = [len(obs_sat) for obs_sat in obs]
            return [
            np.pad(
                o,
                (0, max(obs_sizes) - len(o)),
                "constant",
                constant_values=0,
            )
            for o in obs
            ]
        else:
            return [
                np.pad(
                    o,
                    (0, self.longest_observation_space.shape[0] - len(o)),
                    "constant",
                    constant_values=0,
                )
                for o in obs
            ]

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        obs, img_reward, done, trunc, info = self.env.step(actions.flatten())

        if self.args['transformer_obs']:
            obs = self._reshape_obs(obs)

        self._obs = self._pad_observation(obs)
        dones = done or trunc
        s_obs = self.repeat(self.get_state())
        env_time = self.env.simulator.sim_time
        power_usage_total = 0.0
        data_downlink_total = 0.0
        


        if self._past_obs == None:
            self._past_obs = self._obs

        for sat in self.satellite_names:
            for action_name in self.action_names:
                self._info[f'{sat}-{action_name}'] = 0

        for i, sat in enumerate(self.satellite_names):
            # Firstly, Check battery level penalty:
            # penalty = 0.0
            # if self.min_batt_level_penalty is not None:
            #     if obs[i][self.batt_idx].item() < self.min_batt_level_penalty:
            #         penalty += -5
                    # dones = True

            power_usage_gen = obs[i][self.batt_idx].item()-self._past_obs[i][self.batt_idx].item()
            downlinked = 0.0
            if self.power_reward:
                # Add power usage reward
                if power_usage_gen < 0:
                    battery_usage = -1*power_usage_gen*100
                    battery_cost = battery_usage/100 * (1 - self._past_obs[i][self.batt_idx].item())
                    power_usage_total += battery_cost

            if self.data_reward:
                # Add downlinked data reward
                if obs[i][self.mem_idx].item() < self._past_obs[i][self.mem_idx].item():
                    downlinked = (
                        self._past_obs[i][self.mem_idx].item() - obs[i][self.mem_idx].item()) * 100
                    downlinked_cost = downlinked/100
                    data_downlink_total += downlinked_cost
                    self.downlinked[f'{sat}'] = downlinked

            # Track battery for satellite `sat`
            self._info[f'{sat}-batt'] = obs[i][self.batt_idx].item()
            self._info[f'{sat}-power_usage_gen'] = power_usage_gen
            # Track memory for satellite `sat`
            self._info[f'{sat}-mem'] = obs[i][self.mem_idx].item()
            self._info[f'{sat}-downlinked'] = self.downlinked[f'{sat}']
            act = [int(a) for a in actions]
            self._info[f'{sat}-{self.action_names[act[i]]}'] = 1

        # self.img_cost.append(float(reward))
        # self._info[f'img_cost'] = np.mean(self.img_cost)
        self._info['img_cost'] = img_reward

        if img_reward != 0.0: 
            self.imaged += 1 
            self._info['imaged'] = self.imaged
        #     print(f'Current total AoI imaged: {self.imaged}')
        # print(f'Sim time: {env_time:.2f}')
        self._info['time'] = env_time
        reward = (1-(self.battery_cost_scale + self.data_cost_scale))* img_reward - self.battery_cost_scale * power_usage_total +  self.data_cost_scale * data_downlink_total #+ penalty
        self._past_obs = self._obs

        return (
            self._obs,
            s_obs,
            self.n_agents*[[reward]],
            self.n_agents*[dones],
            self.n_agents*[self._info],
            self.get_avail_actions(),
        )

    def reset(self):
        """Returns initial observations and states"""
        obs, info = self.env.reset()
        if self.args['transformer_obs']:
            obs = self._reshape_obs(obs)
        self._obs = self._pad_observation(obs)
        self._past_obs = self._obs
        s_obs = self.repeat(self.get_state())
        self._info = {}
        self.img_cost = 0.0
        self.imaged = 0
        self._info['imaged'] = self.imaged
        for sat in self.satellite_names:
            for action_name in self.action_names:
                self._info[f'{sat}-{action_name}'] = 0.0
            self.downlinked[f'{sat}'] = 0.0

        return self._obs, s_obs, self.get_avail_actions()
    
    def _reshape_obs(self, obs):
        dict_obs = []
        for j in range(self.n_agents):
            obs_names = [self.env.satellites[j].observation_description[k].item(
            ) for k in range(self.env.observation_space[j].shape[0])]
            detailed_obs = dict(zip(obs_names, obs[j]))
            dict_obs.append(self.create_nested_dict(detailed_obs))

        new_obs = []        
        for obs_sat in dict_obs:
            ch_obs = np.array([obs_sat['sat_props']['battery_charge_fraction'],
                      obs_sat['eclipse[0]'],
                      obs_sat['eclipse[1]'],
                      obs_sat['time']
                      ])
            dl_obs =  np.array([obs_sat['sat_props']['storage_level_fraction'],
                      obs_sat['ground_station']['ground_station_0']['opportunity_open_normd'],
                      obs_sat['ground_station']['ground_station_0']['opportunity_close_normd'],
                      obs_sat['time']
                      ])
            ds_obs =  np.array([dict_obs[0]['sat_props']['wheel_speeds_fraction[0]'],
                      dict_obs[0]['sat_props']['wheel_speeds_fraction[1]'],
                      dict_obs[0]['sat_props']['wheel_speeds_fraction[2]'],
                      obs_sat['time']
                      ])
            cap_obs = []
            for tgt_obs in obs_sat['target']:
                cap_obs.append(np.array([obs_sat['target'][tgt_obs]['priority'],
                                          obs_sat['target'][tgt_obs]['prop_1'],
                                          obs_sat['target'][tgt_obs]['prop_2'],
                                          obs_sat['target'][tgt_obs]['opportunity_open_normd']])
                                          )
            cap_obs = np.array(cap_obs).reshape(-1)
            other_obs = np.array([ch_obs, dl_obs, ds_obs]).reshape(-1)

            new_obs.append(np.concatenate((other_obs, cap_obs)))

        return new_obs
        
    def create_nested_dict(self,flat_dict):
        nested_dict = {}
        for key_string, value in flat_dict.items():
            # Split the key string into a list of keys
            keys = key_string.split('.')
            current_dict = nested_dict
            for i, key in enumerate(keys):
                if i == len(keys) - 1:  # If it's the last key, assign the value
                    current_dict[key] = value
                else:  # Otherwise, create a new dictionary or navigate to an existing one
                    current_dict = current_dict.setdefault(key, {})
        return nested_dict

    def get_avail_actions(self):
        if self.discrete:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        valid = flatdim(self.env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_state_size(self):
        """Returns the shape of the state"""
        if hasattr(self.env.unwrapped, "state_size"):
            return self.env.unwrapped.state_size
        # total_length = 0
        # for obs_space in self.observation_space:
        #     total_length+=flatdim(obs_space)
        total_length = self.longest_observation_space.shape[0]*self.n_agents
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_length,),
            dtype=self.env.observation_space[0].dtype,
        )

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def render(self):
        pass
        # self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self._seed = seed

    def wrap(self, l):
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = l[i]
        return d

    def unwrap(self, d):
        l = []
        for agent in range(self.n_agents):
            l.append(d[agent])
        return l

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
