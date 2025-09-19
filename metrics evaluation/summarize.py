import pandas as pd
import os
import argparse

from baseline_models.resnet18_34_backbone import ResNet
from baseline_models.resnet50_backbone import ResNet50
from early_exit_models.resnet50_EE import ResNetEE50
from early_exit_models.resnet18_34_EE import ResNetEE
from baseline_models.residual_blocks import ResidualBlock, ResidualBlock50
from training_testing.dataloaders_scripts import data_loader

from metrics_utils import *

def summarize(test_loader, path, dest):

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  results = []

  for filename in os.listdir(path):
    if filename.endswith(".pth") and "_EE" in filename:
        if "50" in filename:
            model = ResNetEE50(ResidualBlock50, [3, 4, 6, 3]).to(device)
            name = "ResNet50_EE"
        elif "34" in filename:
            model = ResNetEE(ResidualBlock, [3, 4, 6, 3]).to(device)
            name = "ResNet34_EE"
        else:
            model = ResNetEE(ResidualBlock, [2, 2, 2, 2]).to(device)
            name = "ResNet18_EE"

        state_dict = torch.load(os.path.join(path, filename))
        model.load_state_dict(state_dict)
        model.eval()

        accuracy, inference_time = performance_metrics(model, early=True, test_loader=test_loader)
        exit_counts = exits(model, test_loader)
        flops = FLOPs(model, early=True)
        avg_flops = avg_FLOPs(flops, model, test_loader)
        energy_consumed, gpu_energy_consumed, gpu_model, emi = consumption(model, test_loader)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Inference_Time": inference_time,
            "Exits": dict(exit_counts),
            "FLOPs_by_exit": flops,
            "Avg_FLOPs_per_sample": avg_flops,
            "Energy_Consumed": energy_consumed,
            "GPU_Energy_Consumed": gpu_energy_consumed,
            "GPU_Model": gpu_model,
            "Emissions": emi
        })

    elif filename.endswith(".pth") and "_EE" not in filename:
        if "50" in filename:
            model = ResNet50(ResidualBlock50, [3, 4, 6, 3]).to(device)
            name = "ResNet50"
        elif "34" in filename:
            model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)
            name = "ResNet34"
        else:
            model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
            name = "ResNet18"

        state_dict = torch.load(os.path.join(path, filename))
        model.load_state_dict(state_dict)
        model.eval()

        accuracy, inference_time = performance_metrics(model, early=False, test_loader=test_loader)
        flops = FLOPs(model)
        energy_consumed, gpu_energy_consumed, gpu_model, emi = consumption(model, test_loader)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Inference_Time": inference_time,
            "Exits": None,
            "FLOPs_by_exit": None,
            "Avg_FLOPs_per_sample": flops,
            "Energy_Consumed": energy_consumed,
            "GPU_Energy_Consumed": gpu_energy_consumed,
            "GPU_Model": gpu_model,
            "Emissions": emi
        })

  # Sort results by model name to ensure desired order
  sorted_results = sorted(results, key=lambda x: (x['Model'].replace('ResNet', '').replace('_EE', '')))

  df = pd.DataFrame(sorted_results)

  print(df)

  df.to_csv(dest + "model_results.csv", index=False)

  return

class Final:

    def __init__(self, src, dest, data):
        self.src = src
        self.dest = dest
        self.data = data

    def summary(self):
        test_loader = data_loader(data_dir=self.data,
                                  batch_size=1,
                                  test=True)

        summarize(test_loader, self.src, self.dest)

        return


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True, help="Where to store the data?")
    parser.add_argument("--src", type=str, required=True, help="Where to look for models?")
    parser.add_argument("--dest", type=str, required=True, help="Where to store the results?")

    args = parser.parse_args()

    src = args.src
    dest = args.dest
    data = args.data

    final = Final(src, dest, data)
    final.summary()

if __name__ == "__main__":
    main()