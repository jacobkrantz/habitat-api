import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from habitat_baselines.common.utils import Flatten


class ResNet50(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    def __init__(
        self, observation_space, output_size, device, activation="tanh"
    ):
        super().__init__()
        self.device = device
        self.resnet_layer_size = 2048
        linear_layer_input_size = 0
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            obs_size_0 = observation_space.spaces["rgb"].shape[0]
            obs_size_1 = observation_space.spaces["rgb"].shape[1]
            if obs_size_0 != 224 or obs_size_1 != 224:
                print(
                    f"WARNING: ResNet50: observation size {obs_size} is not conformant to expected ResNet input size [3x224x224]"
                )
            linear_layer_input_size += self.resnet_layer_size
        else:
            self._n_input_rgb = 0

        if self.is_blind:
            self.cnn = nn.Sequential()
            return

        self.cnn = models.resnet50(pretrained=True)

        # disable gradients for resnet, params frozen
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.layer_extract = self.cnn._modules.get("avgpool")
        self.cnn.eval()

        self.fc = nn.Linear(linear_layer_input_size, output_size)
        self.activation = nn.Tanh() if activation == "tanh" else nn.ReLU()

    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    def forward(self, observations):
        r"""Sends RGB observation through ResNet50 pre-trained on ImageNet.
        Sends through fully connected layer, activates with tanh, and returns
        final embedding.
        """

        def resnet_forward(observation):
            resnet_output = torch.zeros(
                observation.size(0), self.resnet_layer_size
            ).to(self.device)

            def hook(m, i, o):
                # self.resnet_output.resize_(d.size())
                resnet_output.copy_(torch.flatten(o, 1).data)

            # output: [BATCH x RESNET_DIM]
            h = self.layer_extract.register_forward_hook(hook)
            self.cnn(observation)
            h.remove()
            return resnet_output

        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
        rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
        rgb_observations = rgb_observations / 255.0  # normalize RGB
        resnet_output = resnet_forward(rgb_observations)
        return self.activation(self.fc(resnet_output))  # [BATCH x OUTPUT_DIM]
