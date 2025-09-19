# Summary

## Models Evaluation

### *metrics_utils.py* is used to compute metrics such as accuracy, inference time, average FLPs per sample, energy consumption etc.

### *summarize.py* is used to iterate through all the models of a target directory and utilize the evaluation functions of metrics_utils.py to provide additional insights.

### The configuration of summarize.py should look like this:

<code>
--src
"C:\Users\..\Desktop\ResNet_Early_Exit\pretrained_models/"
--dest
"C:\Users/../Desktop/ResNet_Early_Exit/data/"
--data
"C:\Users/../Desktop/ResNet_Early_Exit/data/"
</code>