import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from habitat_baselines.common.utils import Flatten


class ResNet50(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    def __init__(self, observation_space, output_size, device):
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

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
            linear_layer_input_size += self.resnet_layer_size
        else:
            self._n_input_depth = 0

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
        self.activation = nn.Tanh()

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations):
        r"""Sends RGB observation through pre-trained ResNet50. Repeats Depth
        channel 3 times then sends depth observation through ResNet50.
        Concatenates resulting vectors. Sends through fully connected layer,
        activates with tanh, and returns final embedding.

        Applies a tanh activation over the reduced ResNet output.
        Not sure if we should just have the fully connected linear layer or
        this added non-linearity on top of it.
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

        output = []
        if self._n_input_rgb > 0:
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
            rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            a = resnet_forward(rgb_observations)
            output.append(a)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"].permute(0, 3, 1, 2)

            # pre-trained ResNet can only handle 3-channel data. Work-around:
            #   Repeat the single channel values for three channels.
            depth_observations = depth_observations.repeat_interleave(
                repeats=3, dim=1
            )
            output.append(resnet_forward(depth_observations))

        output = torch.cat(output, dim=1)
        return self.activation(self.fc(output))  # [BATCH x OUTPUT_DIM]
