"""HAPPO algorithm."""
import numpy as np
import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm



class RULEBASED:
    def __init__(self, args, env):
        """Initialize HAPPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        self.args = args
        self.envs = env


    def create_nested_dict(self, flat_dict):
        nested_dict = {}
        for key_string, value in flat_dict.items():
            keys = key_string.split('.')  # Split the key string into a list of keys
            current_dict = nested_dict
            for i, key in enumerate(keys):
                if i == len(keys) - 1:  # If it's the last key, assign the value
                    current_dict[key] = value
                else:  # Otherwise, create a new dictionary or navigate to an existing one
                    current_dict = current_dict.setdefault(key, {})
        return nested_dict

    def act(
        self, agent_id, obs, masks, available_actions=None, deterministic=False
    ):
        """Compute actions using the given inputs.
        Args:
            obs: (np.ndarray) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions= self.RuleBasedAct(agent_id,obs)
        
        return actions
    
    def RuleBasedAct(self, agent_id, sats_current_state):

        obs_names = [self.envs.env.satellites[agent_id].observation_description[k].item() for k in range(self.envs.env.observation_space[agent_id].shape[0])]
        sats_current_state_list = sats_current_state.tolist()[0]
        detailed_obs = dict(zip(obs_names, sats_current_state_list[:len(obs_names)]))
        current_state = self.create_nested_dict(detailed_obs)

        
        action_names = self.envs.action_names
        n_basic_act = 3         # Use this or 3 basic actions: Charge, Downlink, Desaturate
        # n_basic_act = 4       # Use this or 4 basic actions:Charge, Downlink, Desaturate, Drift

        sat_name=self.envs.satellite_names[agent_id]

        n_acts = len(action_names)

        if current_state['sat_props']['battery_charge_fraction'] > 0.8:               # Battery Level Check
            # Battery Ok, then Reaction Wheel Check
            if current_state['sat_props']['wheel_speeds_fraction[0]'] <= 0.8 and current_state['sat_props']['wheel_speeds_fraction[1]'] <= 0.8 and current_state['sat_props']['wheel_speeds_fraction[2]'] <= 0.8:
                # Battery OK, Reaction Wheel OK, then Memory Check
                if current_state['sat_props']['storage_level_fraction'] < 0.9:
                    # Battery OK, Reaction Wheel OK, Memory Available, then Check Ground Target Opportunity
                    # if current_state[8] > 0 or current_state[12] > 0 or current_state[16] > 0 or current_state[20] > 0:
                    if 'OPT' in sat_name:
                        priorities = [current_state['target'][f'target_{i}']['priority'] for i in range(int((n_acts-n_basic_act)/2))]
                    else:
                        priorities = [current_state['target'][f'target_{i}']['priority'] for i in range(n_acts-n_basic_act)]
                    max_priority = np.argmax(priorities)
                    priority = np.max(priorities)
                    cloud_coverage = current_state['target'][f'target_{max_priority}']['prop_1']

                    if sat_name=="OPT-1-Sat":
                        act = n_basic_act + max_priority
                        print(
                            f"{sat_name} Capturing tgt-{act}, Priority = {priority}, Cloud = {cloud_coverage}")
                    else:
                        if cloud_coverage > 0.5 and "SAR" in sat_name:
                            act = n_basic_act + max_priority
                            print(
                                f"{sat_name} Capturing tgt-{act}, Priority = {priority}, Cloud = {cloud_coverage}")
                            
                        elif cloud_coverage < 0.5 and "SAR" in sat_name:
                            act = 2
                            print(f"{sat_name}-Desaturating")

                        elif cloud_coverage < 0.5 and "OPT" in sat_name:
                            act = n_basic_act + max_priority
                            print(
                                f"{sat_name} Capturing tgt-{act}, Priority = {priority}, Cloud = {cloud_coverage}")
                            
                        elif cloud_coverage > 0.5 and "OPT" in sat_name:
                            act = 2
                            print(f"{sat_name}-Desaturating")
                else:
                    act = 1 # Downlinking
                    # Battery OK, Reaction Wheel OK, Memory Full, Ground Station always Available --> Downlinking!
                    print(f"{sat_name}-Downlinking")
            else:
                act = 2
                # Battery OK, Reaction Wheel Saturated --> Desaturating!
                print(f"{sat_name}-Desaturating")
        # Battery Low, Eclipse Check
        elif current_state['eclipse[0]'] >= 0.8 and current_state['eclipse[1]'] <= 0.2:
            act = 2
            # Battery Low, under Eclipse (Shaded Area) condition --> Desaturating! (No Action just desaturating)
            print(f"{sat_name}-Desaturating")
        else:
            act = 0 # Charging
            print(f"{sat_name} Charging at battery = {current_state['sat_props']['battery_charge_fraction']}")



        return [act]
