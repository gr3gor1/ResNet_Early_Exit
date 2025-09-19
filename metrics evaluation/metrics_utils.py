import torch
import time
import torch.nn as nn

from collections import Counter
from codecarbon import EmissionsTracker
from ptflops import get_model_complexity_info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResNetExitWrapper(nn.Module):
    def __init__(self, original_model, exit_idx):
        super().__init__()
        self.model = original_model
        self.exit_idx = exit_idx

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.maxpool(x)

        layers = [self.model.layer0, self.model.layer1, self.model.layer2, self.model.layer3]
        exits = [self.model.exit0, self.model.exit1, self.model.exit2, self.model.exit3]

        for i, layer in enumerate(layers):
            x = layer(x)
            if i == self.exit_idx and self.exit_idx < 4:
                return exits[i](x)

        # final layer
        if self.exit_idx == 4:
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            return self.model.fc(x)

def FLOPs(model, early=False):
  if not early:
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
    flops = macs * 2
    return flops

  else:
    flops_by_exit = {}

    for exit_idx in range(5):
      wrapper = ResNetExitWrapper(model, exit_idx).to('cuda')
      macs, params = get_model_complexity_info(wrapper, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
      flops_by_exit[exit_idx] = macs * 2

    return flops_by_exit

def avg_FLOPs(flops, model, test_loader):
  total_flops = 0
  total_samples = 0

  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        preds, exit = model(images)
        predicted = preds.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        total_flops += flops[int(exit)]
        total_samples += 1

  avg_flops = total_flops / total_samples

  return avg_flops

def consumption(model, test_loader):
    tracker = EmissionsTracker(project_name="resnet_eval")
    tracker.start()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            if isinstance(output, tuple):
                preds, exit_id = output
            else:
                preds = output

            predicted = preds.argmax(dim=1)

    emissions = tracker.stop()

    energy_data = tracker.final_emissions_data
    energy_consumed = energy_data.energy_consumed
    gpu_energy_consumed = energy_data.gpu_energy
    gpu_model = energy_data.gpu_model
    emiss = energy_data.emissions

    return energy_consumed, gpu_energy_consumed, gpu_model, emiss

def exits(model, test_loader):
    exit_counter = Counter()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            preds, exit_id = model(images)   # changed name
            predicted = preds.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            exit_counter[int(exit_id)] += 1

    return exit_counter

def performance_metrics(model, test_loader, early=False):

  if early:
    start = time.time()

    with torch.no_grad():
      correct = 0
      total = 0
      for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        preds, exits = model(images)

        predicted = preds.argmax(dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    end = time.time()

    return 100 * correct / total, end-start

  start = time.time()

  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  end = time.time()

  return 100 * correct / total, end-start