import numpy as np
from harl.common.base_logger import BaseLogger
import time


class ClusterbskLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(ClusterbskLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
    def get_task_name(self):
        return self.env_args["key"]

    def per_step(self, data):
        """Process data per step."""
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        self.infos = infos
        dones_env = np.all(dones, axis=1)
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env
        for t in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[t]:
                self.done_episodes_rewards.append(self.train_episode_rewards[t])
                self.train_episode_rewards[t] = 0
    
    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):
        """Log information for each episode."""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.end = time.time()
        print(
            "Env {} Task {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"]["num_env_steps"],
                int(self.total_num_steps / (self.end - self.start)),
            )
        )

        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        self.log_train(actor_train_infos, critic_train_info)

        print(
            "Average step reward is {}.".format(
                critic_train_info["average_step_rewards"]
            )
        )

        if len(self.done_episodes_rewards) > 0:
            aver_episode_rewards = np.mean(self.done_episodes_rewards)
            print(
                "Some episodes done, average episode reward is {}.".format(
                    aver_episode_rewards
                )
            )
            self.writter.add_scalars(
                "train_episode_rewards",
                {"aver_rewards": aver_episode_rewards},
                self.total_num_steps,
            )
            self.done_episodes_rewards = []
        imaged = []
        for i, info in enumerate(self.infos):
            if "imaged" in info[0].keys():
                imaged.append(info[0]["imaged"])
        self.writter.add_scalars(
            "avg_total_imaged", {"Avg_total_imaged": np.mean(imaged)}, self.total_num_steps
        )
        print(f"Average total imaged: {np.mean(imaged):.2f}\n")
        
    def eval_init(self):
        super().eval_init()
        self.eval_imaged = []
        
    def eval_thread_done(self, tid):
        super().eval_thread_done(tid)
        self.eval_imaged.append(self.eval_infos[tid][0]["imaged"])


    def eval_log(self, eval_episode):
        """Log evaluation information."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_env_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
            "eval_imaged:": self.eval_imaged,
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        eval_std_rew = np.std(self.eval_episode_rewards)
        print("Evaluation average episode reward is {}.".format(eval_avg_rew))
        print("Evaluation std episode reward is {}.\n".format(eval_std_rew))
        eval_avg_imaged = np.mean(self.eval_imaged)
        eval_std_imaged = np.std(self.eval_imaged)
        print(f"Evaluation avg imaged is: {eval_avg_imaged:.2f}")
        print(f"Evaluation std imaged is: {eval_std_imaged:2f}\n")

        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_avg_rew]))
        )
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_std_rew])) + "\n"
        )
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_avg_imaged]))
        )
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_std_imaged])) + "\n"
        )
        self.log_file.flush()
    
    def log_env(self, env_infos):
        """Log environment information."""
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, self.total_num_steps)
                if k=='eval_average_episode_rewards':
                    self.writter.add_scalars('eval_std_episode_rewards', {k: np.std(v)}, self.total_num_steps)

    def log_render(self, env_infos, t):
        """Log environment information."""
        for k, v in env_infos.items():
            self.writter.add_scalars(k, {k: v}, t)
    