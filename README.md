# ResNet Early-Exiting Experiment



## What is early exiting?

<p align="justify">Early exiting lets models stop computation once confident, reducing inference time, scaling effort to input complexity, and avoiding overthinking. While first explored in NLP and vision DNNs, it now shows promise in foundation and reasoning models, including BERT, LLaMA, and ViT.</p>

## Results

### Exit Distribution
![exit_distribution.png](diagrams/exit_distribution.png)
### FLOPs per Exit Layer
![FLOPs_per_exit_layer.png](diagrams/FLOPs_per_exit_layer.png)
### Average FLOPs per sample
![FLOPs_per_sample.png](diagrams/FLOPs_per_sample.png)
### Energy Consumption per GPU
![energy_consumption_per_gpu.png](diagrams/energy_consumption_per_gpu.png)
### Inference Time per GPU
![inference_time_per_gpu.png](diagrams/inference_time_per_gpu.png)
### Emissions per GPU
![emissions_per_gpu.png](diagrams/emissions_per_gpu.png)
