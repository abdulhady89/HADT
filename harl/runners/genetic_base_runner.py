"""Base runner for on-policy algorithms."""

import time
import numpy as np
import torch
import setproctitle
from collections import defaultdict
from harl.algorithms.actors import ALGO_REGISTRY
from harl.utils.trans_tools import _t2n
from harl.utils.envs_tools import (
    make_render_env,make_eval_env,make_train_env,
    get_num_agents,
)

from harl.utils.configs_tools import init_dir, save_config
from harl.envs import LOGGER_REGISTRY
import os


class GeneticRunner:
    """Base runner for on-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the Rulebase runner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        self.share_param = algo_args["algo"]["share_param"]
        self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
            args["env"],
            env_args,
            args["algo"],
            args["exp_name"],
            algo_args["seed"]["seed"],
            logger_path=algo_args["logger"]["log_dir"],
        )
        save_config(args, algo_args, env_args, self.run_dir)
        # set the title of the process
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )
        # set the config of env
        # if self.algo_args["render"]["use_render"]:  # make envs for rendering
        #     (
        #         self.envs,
        #         self.manual_render,
        #         self.manual_expand_dims,
        #         self.manual_delay,
        #         self.env_num,
        #     ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        # else:  # make envs for training and evaluation
        #     print("Rule based only available for rendering")
        #     NotImplementedError

        self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
        self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)
        
        # GA agent:
        self.agents = ALGO_REGISTRY[args["algo"]](
                {**algo_args["algo"]},self.envs
            )
        self.max_generations = algo_args["algo"]["max_generations"]

       

        self.num_agents = get_num_agents(args["env"], env_args, self.envs)
        # actor
        # if self.share_param:
        #     self.actor = []
        #     agent = ALGO_REGISTRY[args["algo"]](
        #         {**algo_args["model"], **algo_args["algo"]},
        #         self.envs.observation_space[0],
        #         self.envs.action_space[0],
        #         device=self.device,
        #     )
        #     self.actor.append(agent)
        #     for agent_id in range(1, self.num_agents):
        #         assert (
        #             self.envs.observation_space[agent_id]
        #             == self.envs.observation_space[0]
        #         ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
        #         assert (
        #             self.envs.action_space[agent_id] == self.envs.action_space[0]
        #         ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
        #         self.actor.append(self.actor[0])
        # else:
        #     self.actor = []
        #     for agent_id in range(self.num_agents):
        #         agent = ALGO_REGISTRY[args["algo"]](
        #             {**algo_args["algo"]},
        #             self.envs
        #         )
        #         self.actor.append(agent)


        self.logger = LOGGER_REGISTRY[args["env"]](
            args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
        )
        self.log_file = open(os.path.join(self.log_dir, "ga_progress.txt"), "w", encoding="utf-8")
   
    def run(self):
        gen=1
        for x in range(self.max_generations):
            print("Evaluating generation: ", x)
            self.agents.pop.grade(generation=x)
            print("Evolving generation: ", x)
            self.agents.pop.evolve()
            print("Finished at generation: ", x,
                ", Population fitness:", self.agents.pop.fitness_history[-1].mean())
            self.writter.add_scalar('Population Fitness',
                            self.agents.pop.fitness_history[-1].mean(), gen*self.agents.pop.pop_size)
            self.writter.add_scalar('Population Imaged',np.mean(self.agents.pop.imaged), gen*self.agents.pop.pop_size)
            self.log_file.write(f"Total pop imaged: {np.mean(self.agents.pop.imaged):.2f}\n")
            # self.writter.add_scalar('Population Avg Imaged',
            #                 self.agents.pop.total_imaged, gen*self.agents.pop.pop_size)
            # self.writter.add_scalar('Population Nadir Score',
            #                 self.agents.pop.nadir_score_history[-1], gen*self.pop_size)
            # self.writter.add_scalar('Population Memory Usage',
            #                 self.agents.pop.memory_usage[-1], gen*self.pop_size)
            # self.writter.add_scalar('Population Battery Usage',
            #                 self.agents.pop.battery_usage[-1], gen*self.pop_size)

            # total_actions = 0
            # action_count = defaultdict(int)

        # for action_taken in self.agents.pop.individuals[0].numbers:
        #     action_count[action_taken] += 1
        #     total_actions += 1

        # for action, count in action_count.items():
        #     action_percentage = (count / total_actions) * \
        #         100 if total_actions > 0 else 0
        #     self.writter.add_scalar(
        #         f'Action {action} Usage Percentage', action_percentage, gen*self.pop_size)
        #     print(f'Action {action} Usage Percentage',
        #           action_percentage, gen*self.pop_size)

        # for action_id in sorted(self.action_names.keys()):
        #     action_name = self.action_names[action_id]
        #     count = action_count[action_id]
        #     print(f'  {action_name}: {count} times')

            gen += 1
            

    # @torch.no_grad()
    # def render(self):
    #     """Render the model."""
    #     print("start rendering with rule-based policy")
        
    #     if self.manual_expand_dims:
    #         # this env needs manual expansion of the num_of_parallel_envs dimension
    #         for ep in range(self.algo_args["render"]["render_episodes"]):
    #             self.logger.episode_init(ep)
    #             if self.args['env']!='bsk':
    #                 eval_obs, _, eval_available_actions = self.envs.reset()
    #             else:
    #                 _obs = [sat.get_obs() for sat in self.envs.env.satellites]
    #                 eval_obs = self.envs._pad_observation(_obs)
    #                 eval_available_actions = self.envs.get_avail_actions()

    #             eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
    #             eval_available_actions = (
    #                 np.expand_dims(np.array(eval_available_actions), axis=0)
    #                 if eval_available_actions is not None
    #                 else None
    #             )
                
    #             eval_masks = np.ones(
    #                 (self.env_num, self.num_agents, 1), dtype=np.float32
    #             )
    #             rewards = 0
    #             self.logger.eval_init()
    #             render_step = 0

    #             while True:
    #                 eval_actions_collector = []
    #                 for agent_id in range(self.num_agents):
    #                     eval_actions = self.actor[agent_id].act(
    #                         agent_id,
    #                         eval_obs[:, agent_id],
    #                         eval_masks[:, agent_id],
    #                         eval_available_actions[:, agent_id]
    #                         if eval_available_actions is not None
    #                         else None,
    #                         deterministic=True,
    #                     )
    #                     eval_actions_collector.append((eval_actions))
    #                 eval_actions = [np.array(eval_actions_collector)]
    #                 (
    #                     eval_obs,
    #                     _,
    #                     eval_rewards,
    #                     eval_dones,
    #                     eval_infos,
    #                     eval_available_actions,
    #                 ) = self.envs.step(eval_actions[0])
    #                 rewards += eval_rewards[0][0]
    #                 eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
    #                 eval_available_actions = (
    #                     np.expand_dims(np.array(eval_available_actions), axis=0)
    #                     if eval_available_actions is not None
    #                     else None
    #                 )
    #                 if self.args['env']=='bsk':
    #                     if self.env_args['use_render']==False:
    #                         self.logger.log_render(eval_infos[0],render_step)
    #                 render_step += 1
    #                 if self.manual_render:
    #                     self.envs.render()
    #                 if self.manual_delay:
    #                     time.sleep(0.1)
    #                 if eval_dones[0]:
    #                     print(f"total reward of this episode: {rewards}")
    #                     # self.logger.eval_log(ep,rewards)
    #                     break
        
    def close(self):
        """Close environment, writter, and logger."""
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            # self.logger.close()
