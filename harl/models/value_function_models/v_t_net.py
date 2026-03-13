import torch
import torch.nn as nn
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.models.base.encoder_critic import CrossAgentEncoderCritic
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.utils.models_tools import init, get_init_method


class VTNet(nn.Module):
    """V Network with transformer. Outputs value function predictions given global states."""

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        """Initialize VNet model.
        Args:
            args: (dict) arguments containing relevant model information.
            cent_obs_space: (gym.Space) centralized observation space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(VTNet, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = get_init_method(self.initialization_method)

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        # base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        # self.base = base(args, cent_obs_shape)

        # if self.use_naive_recurrent_policy or self.use_recurrent_policy:
        #     self.rnn = RNNLayer(
        #         self.hidden_sizes[-1],
        #         self.hidden_sizes[-1],
        #         self.recurrent_n,
        #         self.initialization_method,
        #     )

        # def init_(m):
        #     return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        self.n_agents = 3
        self.v_out = CrossAgentEncoderCritic(obs_dim=cent_obs_shape[0]//self.n_agents,
        embed_dim=64,
        num_heads=1,
        num_layers=1
    )
        self.agent_values = []
        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """Compute actions from the given inputs.
        Args:
            cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.
        Returns:
            values: (torch.Tensor) value function predictions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        converted_obs = cent_obs.view(cent_obs.shape[0],self.n_agents, cent_obs.shape[1] // self.n_agents)
        values, _ = self.v_out(converted_obs)
        self.agent_values = values

        return values.mean(), rnn_states
