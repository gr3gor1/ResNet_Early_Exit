# Models Evaluation

- **metrics_utils.py**: Provides functions to compute evaluation metrics such as accuracy, inference time, average FLOPs per sample, and energy consumption.
- **summarize.py**: Iterates over all models in a target directory and applies the evaluation functions from `metrics_utils.py` to produce performance summaries.

## Configuration

The configuration for `summarize.py` should look like this:

`--src "C:\Users\..\Desktop\ResNet_Early_Exit\pretrained_models/"
--dest "C:\Users\..\Desktop\ResNet_Early_Exit\data/" --data "C:\Users\..\Desktop\ResNet_Early_Exit\data/"`