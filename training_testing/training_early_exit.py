import time
import ast
import torch
import torch.optim as optim
import torch.nn as nn
import argparse

from dataloaders_scripts import data_loader
from early_exit_models.resnet50_EE import ResNetEE50
from early_exit_models.resnet18_34_EE import ResNetEE

from baseline_models.residual_blocks import ResidualBlock, ResidualBlock50
from tqdm import tqdm

def trainingEE(layers, path, train_loader, itsfifty=False, variant=None):
    num_epochs = 20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if itsfifty:
        model = ResNetEE50(ResidualBlock50, layers).to(device)
    else:
        model = ResNetEE(ResidualBlock, layers).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    loss_weights = [0.9, 0.9, 0.8, 0.7, 0.3]
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()

    start = time.time()

    for epoch in range(num_epochs):
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                outputs = model(images)

                losses = [criterion(o, labels) for o in outputs]
                total_loss = sum(w * l for w, l in zip(loss_weights, losses))

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=total_loss.item())

    end = time.time()
    print(f"Elapsed time: {end - start:.4f} seconds")

    # Save the model
    if variant is not None:
        variant = str(variant)
        torch.save(model.state_dict(), path + "/model_weights_resnet" + str(variant) + "_EE.pth")
        print("Saved with success")

    return

class TrainEE:

    def __init__(self, dest, data, struct, variant):
        self.dest = dest
        self.data = data
        self.struct = ast.literal_eval(struct)

        self.train_loader, self.valid_loader = data_loader(data_dir=self.data,
                                                 batch_size=64)

        self.test_loader = data_loader(data_dir=self.data,
                                  batch_size=1,
                                  test=True)

        self.variant = variant
        self.fifty = False

        if self.variant == 50:
            self.fifty = True
        else:
            print("Not fifty")

    def trainEE(self):

        trainingEE(self.struct, self.dest, self.train_loader, self.fifty, variant=self.variant)

        return

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, required=True, help="Where to store the models?")
    parser.add_argument("--data", type=str, required=True, help="Where to store the data?")
    parser.add_argument("--struct", type=str, required=True, help="Where to store the data?")
    parser.add_argument("--variant", type=int, required=True,
                        help="What is the variant of ResNet (18,34,50)?")

    args = parser.parse_args()

    dest = args.dest
    data = args.data
    struct = args.struct
    variant = args.variant

    final = TrainEE(dest, data, struct, variant)
    final.trainEE()

if __name__ == "__main__":
    main()