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
EdgeWiseLLM tackles the challenge of deploying resource-intensive LLMs on edge devices through two synergistic modules: CUDA kernel-level optimization and a dynamic decision-making framework. Below is a deeper dive:

## 1. Kernel-Level CUDA Optimization
EdgeWiseLLM implements several CUDA-level strategies to improve token-level inference performance:

### Thread Block Reorganization:
Inference workloads were profiled to identify underutilization in thread execution. Thread blocks were restructured to match the compute/memory access ratio and ensure full warp occupancy.

### Memory Coalescing & Shared Memory:
Data structures were re-aligned to enable coalesced global memory access. Frequently accessed tensors (e.g., attention scores and value matrices) were moved into shared memory, significantly reducing latency.

### Sliding Window Attention:
Inspired by Longformer and GPT optimizations, a sliding window kernel was implemented to truncate the attention window size, limiting quadratic complexity in long sequence inference.

### Quantization Support:
FP16 and FP32 quantized weights were used with appropriate dequantization strategies at runtime, striking a trade-off between precision and speed.

### KV-Cache Sink Optimization:
A specialized “KV sink” check was implemented to reuse cached key/value tensors across decoding steps, minimizing data re-fetch and redundant compute.

## 2. Configurable Inference Decision-Making
Given the broad design space of kernel configurations and hardware characteristics, EdgeWiseLLM introduces a configuration selector:

### Multi-Objective Scoring Function:
Each configuration is evaluated using a scoring metric that considers:

Inference throughput (tokens/sec), Latency (token generation time), memory throughput(GB/S), Power consumption (measured using nvml),  and Model perplexity on validation samples.

Weight-Sum Strategy:
Users specify a custom weight profile (e.g., 50% performance, 30% power, 20% memory), and the framework computes a ranked list of recommended configurations.

Batch Benchmarking Script:
A CLI tool enables rapid benchmarking of configurations across input batch sizes, model sizes, quantization levels, and GPU types (NVIDIA RTX 2060)

# 🧪 Experiment Setup

The following environment and test scenarios were used to evaluate **EdgeWiseLLM**:

| Component            | Details                                                   |
|----------------------|-----------------------------------------------------------|
| **Model**            | TinyLlama [model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)|
| **Precision Modes**  | FP16, FP32                                          |
| **Hardware**         | NVIDIA RTX 2060 (115W TDP)                          |
| **Framework**        | Pure C++ with CUDA                                 |
| **Metrics Tracked**  | Token latency, throughput, memory usage, power draw, perplexity |
| **Profiling Tools**  | NVIDIA Nsight Compute, Nsight Systems, `nvidia-smi`, `tegrastats` |
  
We validated correctness by comparing output logits against PyTorch baselines (within quantization-tolerant thresholds).

# conclusion
EdgeWiseLLM empowers developers and researchers to bring cutting-edge LLM capabilities to the edge—without sacrificing speed or flexibility. By exposing CUDA-based tunables and offering intuitive configuration guidance, it becomes possible to balance model performance with real-world resource limits.

🔗 Related Links
[GUI Demo](https://youtu.be/SVmaBCRmV_I)
📊 Results
[Paper](https://drive.google.com/file/d/1xcePLMSPCUD_SqZWu6xA6HGy0d6cT19O/view?usp=sharing)

