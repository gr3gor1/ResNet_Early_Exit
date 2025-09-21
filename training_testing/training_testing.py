import time
import torch.nn as nn
import ast
import argparse

from baseline_models.resnet18_34_backbone import ResNet
from baseline_models.resnet50_backbone import ResNet50
from baseline_models.residual_blocks import ResidualBlock,ResidualBlock50

from dataloaders_scripts import *
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def training(layers,train_loader, valid_loader, path, itsfifty=False, variant=None):

  # Model Initialization
  if itsfifty:
    model = ResNet50(ResidualBlock50, layers).to(device)
  else:
    model = ResNet(ResidualBlock, layers).to(device)

  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)

  total_step = len(train_loader)

  start = time.time()

  scaler = torch.cuda.amp.GradScaler()

  num_epochs = 20

  for epoch in range(num_epochs):
    loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

  end = time.time()

  print(f"Elapsed time: {end - start:.4f} seconds")


  # Validation
  with torch.no_grad():
      correct = 0
      total = 0
      for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

  print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

  # Save the model
  if variant != None:
    variant = str(variant)
    torch.save(model.state_dict(), path + "/model_weights_resnet" + str(variant) + ".pth")
    print("Saved with success")


def testing(model, test_loader, early=False):

  if early == True:
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

    print(f"Accuracy of the network on the {total} test images: {accuracy:.2f} %")

    end = time.time()
    print(f"Elapsed time: {end - start:.4f} seconds")
    return

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

  print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))

  end = time.time()

  print(f"Elapsed time: {end - start:.4f} seconds")

  return

class Train:

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

        return

    def train(self):
        training(self.struct, self.train_loader, self.valid_loader, self.dest, self.fifty, variant=self.variant)

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

    final = Train(dest, data, struct, variant)
    final.train()

if __name__ == "__main__":
    main()