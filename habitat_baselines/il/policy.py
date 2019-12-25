import torch
import torch.nn as nn
import torch.optim as optim
from gym import Space

from habitat import Config
from habitat_baselines.common.utils import CategoricalNet
from habitat_baselines.rl.vln.ppo.policy import VLNBaselineNet


class ILPolicy(nn.Module):
    r"""habitat_baselines.rl.ppo.policy.Policy but for imitation learning.
    """

    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return action, action_log_probs, rnn_hidden_states

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        actions_logits = distribution.logits
        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return (
            action_log_probs,
            actions_logits,
            distribution_entropy,
            rnn_hidden_states,
        )


class VLNILBaselinePolicy(ILPolicy):
    def __init__(
        self, observation_space: Space, action_space: Space, vln_config: Config
    ):
        super().__init__(
            VLNBaselineNet(
                observation_space=observation_space, vln_config=vln_config
            ),
            action_space.n,
        )


class ILAgent(nn.Module):
    def __init__(self, net, lr=None, eps=None):
        super().__init__()
        self.net = net
        self.optimizer = optim.Adam(net.parameters(), lr=lr, eps=eps)
        # self.device = next(net.parameters()).device
        # https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss
        # https://datascience.stackexchange.com/questions/55962/pytorch-doing-a-cross-entropy-loss-when-the-predictions-already-have-probabiliti
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, *x):
        raise NotImplementedError

    def _compute_loss(self, actions_logits, gt_actions, gradient_weights):
        r"""Computes the loss of a batch of actions. Scale the gradient
        sample-wise.
        https://discuss.pytorch.org/t/how-to-scale-sample-wise-gradients/5203
        """

        def scale_gradients(logits_batch, gradient_weights):
            def hook(g):
                return (
                    g * gradient_weights
                )  # [batch x n] element-wise mul [batch x 1]

            logits_batch.register_hook(hook)

        gt_actions = gt_actions.squeeze()
        scale_gradients(actions_logits, gradient_weights)
        total_loss = self.loss_func(actions_logits, gt_actions)
        return total_loss

    def update(self, rollouts):
        # Reshaped for a single forward pass for all steps
        (
            obs_batch,
            recurrent_hidden_states_batch,
            gt_actions_batch,  # [batch x 1]
            prev_gt_actions_batch,
            masks_batch,
            episodes_over,
            old_action_log_probs_batch,
            actions_batch,
            gradient_weights,
        ) = rollouts.get_batch()

        (
            action_log_probs,  # [batch x 1]
            actions_logits,  # [batch x num_actions]
            dist_entropy,  #  scalar
            _,
        ) = self.net.evaluate_actions(
            obs_batch,
            recurrent_hidden_states_batch,
            prev_gt_actions_batch,
            masks_batch,
            actions_batch,
        )

        self.optimizer.zero_grad()

        total_loss = self._compute_loss(
            actions_logits, gt_actions_batch, gradient_weights
        )

        self.before_backward(total_loss)
        total_loss.backward()
        self.after_backward(total_loss)

        self.before_step()
        self.optimizer.step()
        self.after_step()
        return total_loss, dist_entropy

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass
