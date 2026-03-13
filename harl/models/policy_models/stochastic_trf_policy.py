import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.models.base.act import ACTLayer


from harl.utils.envs_tools import get_shape_from_obs_space


class StochasticTrfPolicy(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(StochasticTrfPolicy, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        # obs_shape = get_shape_from_obs_space(obs_space)
        # base = CNNBase if len(obs_shape) == 3 else MLPBase
        # self.base = base(args, obs_shape)

        # if self.use_naive_recurrent_policy or self.use_recurrent_policy:
        #     self.rnn = RNNLayer(
        #         self.hidden_sizes[-1],
        #         self.hidden_sizes[-1],
        #         self.recurrent_n,
        #         self.initialization_method,
        #     )
        if self.args["use_actLayer"]:
            from harl.models.base.encoder_actor_no_act import CrossActionEncoderPolicy
            self.base = CrossActionEncoderPolicy(
                args=args,
                obs_dim=4,
                embed_dim=self.hidden_sizes[-1],
                num_heads=1,
                num_layers=1
            )
            # self.hidden_state = self.base.init_hidden_state(b=1)

            self.act = ACTLayer(
                action_space,
                self.hidden_sizes[-1],
                self.initialization_method,
                self.gain,
                args,
            )
        else:
            if self.args["use_diff_transformer"]:
                # from harl.models.base.encoder_actor_diff import CrossActionEncoderPolicy
                # self.act = CrossActionEncoderPolicy(
                #     args=args,
                #     obs_dim=4,
                #     embed_dim=self.hidden_sizes[-1],
                #     num_heads=8,
                #     num_kv_heads=4,
                #     num_layers=1
                # )
                from harl.models.base.encoder_actor_mod import CrossActionEncoderPolicy
                self.act = CrossActionEncoderPolicy(
                    args=args,
                    obs_dim=4,
                    embed_dim=self.hidden_sizes[-1],
                    num_heads=1,
                    num_layers=1
                )
            else:
                from harl.models.base.encoder_actor import CrossActionEncoderPolicy
                self.act = CrossActionEncoderPolicy(
                    args=args,
                    obs_dim=4,
                    embed_dim=self.hidden_sizes[-1],
                    num_heads=1,
                    num_layers=1
                )

        self.to(device)
        self.action_space = action_space

    def forward(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        new_obs = obs.view(obs.shape[0], self.action_space.n, obs.shape[1] // self.action_space.n)

        if self.args["use_actLayer"]:
            if self.args["use_hidden_state"]:
                actor_features, rnn_states = self.base(new_obs,rnn_states)
            else:
                actor_features, _ = self.base(new_obs)

            actions, action_log_probs = self.act(
                actor_features, available_actions, deterministic
            )
        else:
            if self.args["use_hidden_state"]:
                actions, action_log_probs, rnn_states = self.act(
                    new_obs, rnn_states, available_actions, deterministic
                )
            else:
                actions, action_log_probs = self.act(
                    new_obs, available_actions, deterministic
                )

        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self, obs, rnn_states, action, masks, available_actions=None, active_masks=None
    ):
        """Compute action log probability, distribution entropy, and action distribution.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN/Transformer network, hidden states for RNN/Transformer.
            action: (np.ndarray / torch.Tensor) actions whose entropy and log probability to evaluate.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        Returns:
            action_log_probs: (torch.Tensor) log probabilities of the input actions.
            dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
            action_distribution: (torch.distributions) action distribution.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        
        new_obs = obs.view(obs.shape[0], self.action_space.n, obs.shape[1] // self.action_space.n)

        if self.args["use_actLayer"]:
            if self.args["use_hidden_state"]:
                actor_features, rnn_states = self.base(new_obs,rnn_states)
            else:
                actor_features, _ = self.base(new_obs)

            actor_features, _ = self.base(new_obs)
            action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(
                actor_features,
                action,
                available_actions,
                active_masks=active_masks if self.use_policy_active_masks else None,
            )
        else:
            if self.args["use_hidden_state"]:
                action_log_probs, dist_entropy, action_distribution, rnn_states = self.act.evaluate_actions(
                    new_obs,
                    action,
                    rnn_states,
                    available_actions,
                    active_masks=active_masks if self.use_policy_active_masks else None,
                )
            else:
                action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(
                    new_obs,
                    action,
                    available_actions,
                    active_masks=active_masks if self.use_policy_active_masks else None,
                )

        # if self.use_naive_recurrent_policy or self.use_recurrent_policy:
        #     actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        

        return action_log_probs, dist_entropy, action_distribution
    

