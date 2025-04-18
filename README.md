# Overview
## EdgeWiseLLM: 
### CUDA-Accelerated Large Language Model Inference with Configurable Performance and Optimal Decision-Making for Edge Devices
This project leverages pure C++ and CUDA technologies to unlock GPU acceleration potential and enhance inference performance on edge platforms. 

**EdgeWiseLLM** aims to tackle the challenges associated with deploying resource-intensive Large Language Models (LLMs) on edge devices, where hardware limitations severely restrict computational power and efficiency.
While Python-based libraries offer ease of use, they often fall short in fully harnessing GPU capabilities. 

Given the multitude of CUDA-based optimization techniques and diverse LLM architectural strategies, identifying the optimal inference configuration becomes a complex and time-consuming task.
EdgeWiseLLM addresses this by providing a configurable framework and a decision-making tool, enabling users to intuitively balance trade-offs in throughput, power consumption, bandwidth, memory usage, and model perplexity.
Thus, users can effectively and quickly tailor inference strategies to their specific hardware constraints and performance requirements.

The code and inspiration are based on [yalm](https://github.com/andrewkchan/yalm)

# Methodology
We designed EdgeWiseLLM around two core pillars:

Kernel-Level Optimization
Utilized thread block reorganization, memory coalescing, and shared memory to boost performance.
Applied sliding window attention and quantization (INT8, INT4) for faster token generation.
Implemented KV-sink management to reduce redundant computation and maximize cache reusability.

Decision-Making Framework
Developed a weighted scoring system that evaluates configurations against multiple edge-centric criteria.
Supports fine-tuning kernel parameters and benchmarking inference modes across different hardware (e.g., Jetson, RTX GPUs).

# Results
Optimization	Speedup (%)	Power Reduction (%)	Memory Footprint (↓)
Shared Memory & Coalescing	+55%	-	Moderate
Kernel Fusion + Quantization	+105%	Up to 30%	High
Config Selector (Best Case)	+120%	Up to 40%	Moderate–High
All benchmarks were run on RTX 2060 and Jetson Xavier NX platforms with 7B and 3B parameter LLMs (GPT-style), using token-level latency and throughput as primary metrics.



# conclusion
EdgeWiseLLM empowers developers and researchers to bring cutting-edge LLM capabilities to the edge—without sacrificing speed or flexibility. By exposing CUDA-based tunables and offering intuitive configuration guidance, it becomes possible to balance model performance with real-world resource limits.
