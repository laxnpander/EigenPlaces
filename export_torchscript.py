import torch
import logging
import torchvision
from torch import nn
from typing import Tuple
from eigenplaces_model import eigenplaces_network

model = eigenplaces_network.GeoLocalizationNet_("ResNet18", 256)

example = torch.rand(1, 3, 512, 512)
traced_net = torch.jit.trace(model, example, strict=False)
traced_net.eval()
traced_net.save("eigenPlaces.pt")
