import logging
import torch
import torch.nn as nn
from ..utils.mlp import MLPBase
from ..utils.gru import GRULayer
from ..utils.act import ACTLayer
from ..utils.utils import check

class ActorNet(nn.Module):
    """
    SAC Actor Network for LAG platform
    - Uses MLPBase for feature extraction
    - Optionally uses GRULayer for recurrent policy
    - Uses ACTLayer for continuous action distribution (DiagGaussian)
    """
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(ActorNet, self).__init__()
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers

        self.base = MLPBase(
            obs_space,
            args.hidden_size,
            args.activation_id,
            self.use_feature_normalization
        )

        input_size = self.base.output_size
        if self.use_recurrent_policy:
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size

        self.act_layer = ACTLayer(
            act_space=act_space,
            input_dim=input_size,
            hidden_size=args.act_hidden_size,
            activation_id=args.activation_id,
            gain=args.gain,
            device=self.device
        )

        # 初始化权重
        self._init_weights()
        self.to(self.device)

    def _init_weights(self):
        """
        Initialize weights for MLPBase, GRULayer, and ACTLayer
        """
        def init_module(module):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    # 使用 Xavier 初始化
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.constant_(m.bias, 0.01)
                elif isinstance(m, nn.GRU):
                    for name, param in m.named_parameters():
                        if 'weight' in name:
                            nn.init.xavier_uniform_(param)
                        elif 'bias' in name:
                            nn.init.constant_(param, 0.01)

        # 初始化 MLPBase
        init_module(self.base)

        # 初始化 GRULayer（如果使用）
        if self.use_recurrent_policy:
            init_module(self.rnn)

        # 初始化 ACTLayer
        init_module(self.act_layer)

    def forward(self, obs, rnn_states=None, masks=None, deterministic=False):
        """
        Input:
            obs: shape [batch_size, obs_dim]
            rnn_states: shape [batch_size, num_layers, hidden_size] (if recurrent)
            masks: shape [batch_size, 1] (if recurrent)
        Output:
            actions, log_probs, rnn_states (if recurrent, else None)
        """
        obs = check(obs).to(**self.tpdv)
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            logging.warning(f"Invalid obs detected: {obs}")
            obs = torch.clamp(obs, -10, 10)
        feat = self.base(obs)

        if self.use_recurrent_policy:
            if rnn_states is None:
                rnn_states = torch.zeros(
                    (self.recurrent_hidden_layers, obs.size(0), self.recurrent_hidden_size),
                    dtype=torch.float32, device=self.tpdv['device']
                )
            else:
                rnn_states = check(rnn_states).to(**self.tpdv)
            if masks is None:
                masks = torch.ones((obs.size(0), 1), dtype=torch.float32, device=self.tpdv['device'])
            else:
                masks = check(masks).to(**self.tpdv)
            if rnn_states.dim() == 3:
                rnn_states = rnn_states.transpose(0, 1)
            feat, new_rnn_states = self.rnn(feat, rnn_states, masks)
        else:
            new_rnn_states = None

        actions, log_probs = self.act_layer(feat, deterministic=deterministic, return_log_prob=True)
        if self.training and torch.rand(1).item() < 0.01:  # 约每 100 次打印一次
            logging.debug(f"Actions: {actions.detach().cpu().numpy()}")
        return actions, log_probs, new_rnn_states

    def evaluate_actions(self, obs, actions, rnn_states=None, masks=None):
        """
        Compute log probs and entropy for given actions
        """
        obs = check(obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            logging.warning(f"Invalid obs detected in evaluate_actions: {obs}")
            obs = torch.clamp(obs, -10, 10)
        if torch.isnan(actions).any() or torch.isinf(actions).any():
            logging.warning(f"Invalid actions detected in evaluate_actions: {actions}")
            actions = torch.clamp(actions, -10, 10)
        feat = self.base(obs)

        if self.use_recurrent_policy:
            if rnn_states is None:
                rnn_states = torch.zeros(
                    (self.recurrent_hidden_layers, obs.size(0), self.recurrent_hidden_size),
                    dtype=torch.float32, device=self.tpdv['device']
                )
            else:
                rnn_states = check(rnn_states).to(**self.tpdv)
            if masks is None:
                masks = torch.ones((obs.size(0), 1), dtype=torch.float32, device=self.tpdv['device'])
            else:
                masks = check(masks).to(**self.tpdv)
            if rnn_states.dim() == 3:
                rnn_states = rnn_states.transpose(0, 1)
            feat, new_rnn_states = self.rnn(feat, rnn_states, masks)
        else:
            new_rnn_states = None

        action_log_probs, dist_entropy = self.act_layer.evaluate_actions(feat, actions)
        return action_log_probs, dist_entropy, new_rnn_states