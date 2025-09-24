# Training ResNet Variants

## Baseline Models

### In order to train baseline models (ResNet18, ResNet34, ResNet50), without exits attached, we can utilize the training_testing.py script.

## EE Models

### In order to train EE variants we can utilize the training_early_exit.py script.

## Notes

### Both training scripts need a destination to save the model, a destination to save the data used for the dataloaders, the number of blocks per layer and the variant we are interested in.

### We can use a configuration similar to this one:

`--dest "C:\..\Desktop\ResNet_Early_Exit\dest" --data "C:\..\Desktop\ResNet_Early_Exit\data" 
--struct "[2,2,2,2]" --variant 18`

