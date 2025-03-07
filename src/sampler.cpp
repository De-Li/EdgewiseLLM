#include "sampler.h"

#include <cfloat>

Sampler::Sampler(const std::shared_ptr<Config> config) {
  vocab_size = config->vocab_size;
  //deli: initialize nvml
  result = nvmlInit();
  nvml_init();
}

float Sampler::sample_prob(int index, const InferenceState& s) {
  const float* logits = s.logits();
  // Find max value to moderate the logits later on for numerical stability
  float max_val = -FLT_MAX;
  for (int i = 0; i < vocab_size; ++i) {
    if (logits[i] > max_val) {
      max_val = logits[i];
    }
  }
  float sum = 0;
  for (int i = 0; i < vocab_size; ++i) {
    sum += expf(logits[i] - max_val);
  }
  return expf(logits[index] - max_val) / sum;
}

int Sampler::sample_argmax(const InferenceState& s) {
  const float* logits = s.logits();
  int argmax = 0;
  float max_val = -FLT_MAX;
  for (int i = 0; i < vocab_size; ++i) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      argmax = i;
    }
  }
  return argmax;
}

int Sampler::nvml_init(){
  
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return 1;
    }

    // Get handle for the first GPU (index 0).
    // If you have multiple GPUs, you can adjust the index or loop over them.
    
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        std::cout << "Failed to get device handle: " << nvmlErrorString(result) << std::endl;
        nvmlShutdown();
        return 1;
    }
    return 0;
}

void Sampler::collect_GPU_power(){
  nvmlReturn_t ret = nvmlDeviceGetPowerUsage(device, &power_mW);
  if (ret == NVML_SUCCESS) {
      // power_mW is the current power draw in milliwatts
      measurement.push_back(power_mW);
      ++validSamples;
  } else {
      std::cout << "Failed to get power usage: " << nvmlErrorString(ret) << std::endl;
  }
}

float Sampler::get_avg_power(){
  int sum = 0;
  for(auto measure:measurement){
    sum += measure;
  }
  return (float)sum/validSamples;
}