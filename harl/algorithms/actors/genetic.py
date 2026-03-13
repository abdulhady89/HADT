"""Genetic Algorithm."""
import numpy as np
import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
import random

class Individual(object):

    def __init__(self, numbers=None, mutate_prob=0.01, n_action=4, n_trajectory=400, env=None):
        self.envs = env
        self.n_agents = self.envs.n_agents
        self.n_steps = 0
        self.individual_fit = 0
        self.individual_nadir_reward = 0
        self.mean_batt = 0
        self.mean_mem = 0
        self.n_action = self.envs.action_space[0].n
        self.total_imaged = 0
        

        if numbers is None:
            # self.numbers = np.random.randint(
            #     self.n_action, size=[n_trajectory,self.n_agents]).tolist()
            self.numbers = []
            for i in range(n_trajectory):
                num=[]
                for agent_id in range(self.n_agents):
                    agent_action = np.sum(self.envs.envs[0].avail_actions[agent_id])
                    num.append(np.random.randint(agent_action))
                self.numbers.append(num)
        else:
            self.numbers = numbers
            # Mutate
            if mutate_prob > np.random.rand():
                mutate_index = np.random.randint(len(self.numbers) - 1)
                self.numbers[mutate_index] = np.random.randint(n_action,size=self.n_agents)

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
    
    def fitness(self):
        """
            Returns fitness of individual
            Fitness is the immidiate reward from action taken in environment
        """

        obs, state, available_actions = self.envs.reset()
        current_state = []
        for agent_id in range(self.n_agents):
            obs_names = [name.item() for name in self.envs.envs[0].env.satellites[agent_id].observation_description]
            sats_current_state_list = obs.tolist()[0]
            detailed_obs = dict(zip(obs_names, sats_current_state_list[:len(obs_names)]))
            current_state.append(self.create_nested_dict(detailed_obs))


        # current_state, info = self.env.reset(seed=123)
        trajectory_actions = self.numbers
        score = 0
        nadirScore = 0
        battery_total_charge_amount = 0
        batt_usage = []
        mem_usage = []

        for act in trajectory_actions:
            (obs, state, reward, dones, info, available_actions) = self.envs.step(np.array([act]))
            self.n_steps += 1
            next_state = []
            for agent_id in range(self.n_agents):
                obs_names = [name.item() for name in self.envs.envs[0].env.satellites[agent_id].observation_description]
                sats_next_state_list = obs.tolist()[0]
                detailed_obs = dict(zip(obs_names, sats_next_state_list[:len(obs_names)]))
                next_state.append(self.create_nested_dict(detailed_obs))

            current_state = next_state
            # if act == 1:
            #     battery_charged_fraction = next_state[1].item(
            #     ) - current_state[1].item()
            #     power_used = 40 * 20 / (3600 * battery_storage_capacity)
            #     battery_total_charge_amount += battery_charged_fraction + power_used

            # done = 1 if (terminated or truncated) else 0
            # if not done:
            #     pure_image_capture_reward = cdcp(reward)

            # # if soft reward:
            # if soft_reward:
            #     # here we have battery usage as small punishment for each step:
            #     if current_state[1].item() < battery_warning_level:
            #         battery_cost = current_state[1].item(
            #         ) - next_state[1].item()
            #         battery_cost *= battery_cost_scale * \
            #             (1 - current_state[1].item())
            #         reward -= battery_cost
            score += reward
            # nadirScore += pure_image_capture_reward
            # batt_usage.append(current_state[1].item())
            # mem_usage.append(current_state[0].item())
            if dones.any():
                break
            
        self.total_imaged = info[0][0]['imaged']#self.envs.envs[0].imaged 
        print(f'Individual fitness evaluation done, score: {score} and total_imaged: {self.total_imaged}')
        self.individual_fit = score
        #info['imaged']
        # self.individual_nadir_reward = nadirScore
        # self.mean_batt = 1 - np.mean(batt_usage)
        # self.mean_mem = np.mean(mem_usage)
        # print('Individual Mean Battery Usage:', self.mean_batt)
        # print('Individual Mean Memory Usage:', self.mean_mem)

        return score


class Population(object):

    def __init__(self, pop_size=10, mutate_prob=0.01, retain=0.2, random_retain=0.03, env=None):
        """
            Args
                pop_size: size of population
                fitness_goal: goal that population will be graded against
        """
        self.pop_size = pop_size
        self.mutate_prob = mutate_prob
        self.retain = retain
        self.random_retain = random_retain
        self.fitness_history = []
        self.nadir_score_history = []
        self.battery_usage = []
        self.memory_usage = []
        self.parents = []
        self.imaged = []
        self.done = False
        self.env = env

        # Create individuals
        self.individuals = []
        for x in range(pop_size):
            self.individuals.append(Individual(
                numbers=None, mutate_prob=self.mutate_prob, env=self.env))

    def grade(self, generation=None):
        """
            Grade the generation by getting the average fitness of its individuals
        """
        print('Evaluating individuals..')
        fitness_sum = 0
        nadir_score_sum = 0
        batt_use = 0
        mem_use = 0
        imged = 0
        for x in self.individuals:
            fitness_sum += x.fitness()
            nadir_score_sum += x.individual_nadir_reward
            batt_use += x.mean_batt
            mem_use += x.mean_mem
            imged += x.total_imaged

        pop_fitness = fitness_sum / self.pop_size
        pop_nadir_score = nadir_score_sum / self.pop_size
        pop_batt_use = batt_use / self.pop_size
        pop_mem_use = mem_use / self.pop_size
        pop_imaged = imged / self.pop_size

        self.fitness_history.append(pop_fitness)
        self.nadir_score_history.append(pop_nadir_score)
        self.memory_usage.append(pop_mem_use)
        self.battery_usage.append(pop_batt_use)
        self.imaged.append(pop_imaged)

        # Set Done flag if we hit target
        # if int(round(pop_fitness)) == 0:
        #     self.done = True

        if generation is not None:
            print("Episode", generation, "Population fitness:", pop_fitness)

    def select_parents(self):
        """
            Select the fittest individuals to be the parents of next generation (lower fitness it better in this case)
            Also select a some random non-fittest individuals to help get us out of local maximums
        """
        # Sort individuals by fitness (we use reversed because in case lower fitness is better)
        # self.individuals = list(reversed(sorted(self.individuals, key=lambda x: x.fitness(), reverse=True)))
        self.individuals = list(sorted(self.individuals, key=lambda x: x.individual_fit.mean(), reverse=True))
        # Keep the fittest as parents for next gen
        retain_length = int(self.retain * len(self.individuals))
        if retain_length == 0:
            self.parents = [self.individuals[0]]
        else:
            self.parents = self.individuals[:retain_length]
            # Randomly select some from unfittest and add to parents array
            unfittest = self.individuals[retain_length:]
            for unfit in unfittest:
                if self.random_retain > np.random.rand():
                    self.parents.append(unfit)

    def breed(self):
        """
            Crossover the parents to generate children and new generation of individuals
        """
        target_children_size = self.pop_size - len(self.parents)
        children = []
        if len(self.parents) > 1:
            while len(children) < target_children_size:
                parents = random.sample(self.parents, 2)
                father = parents[0]
                mother = parents[1]
                child_numbers = [random.choice(pixel_pair) for pixel_pair in zip(
                    father.numbers, mother.numbers)]
                child = Individual(numbers=child_numbers, env=self.env)
                children.append(child)
            self.individuals = self.parents + children

    def evolve(self):
        print('Evolving and breed new generation ..')
        # 1. Select fittest
        self.select_parents()
        # 2. Create children and new generation
        self.breed()
        # 3. Reset parents and children
        self.parents = []
        self.children = []


class GENETIC:
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
        # self.max_generation = args["max_generation"]
        self.pop_size = args["pop_size"]
        self.mutate_prob = args["mutate_prob"]
        self.retain =  args["retain"]
        self.random_retain =  args["random_retain"]
        self.pop = Population(pop_size=self.pop_size, mutate_prob=self.mutate_prob,
                     retain=self.retain, random_retain=self.random_retain, env=self.envs)