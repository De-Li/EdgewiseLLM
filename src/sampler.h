#pragma once

#include "model.h"

#include <memory>
#include <iostream>

//GPU power guage
#include <chrono>
#include <thread>
#include <nvml.h>

struct Sampler {
  int vocab_size;

  Sampler(const std::shared_ptr<Config> config);

  // Return the probability score corresponding to `logits[index]`.
  // This is equivalent to taking the softmax of the logits and returning
  // the value at index `index`.
  float sample_prob(int index, const InferenceState& s);
  // Return the index of the maximum value in `logits`.
  int sample_argmax(const InferenceState& s);

  int nvml_init();

  void collect_GPU_power();

  float get_avg_power();

  private:
    std::vector<unsigned int> measurement;
    nvmlReturn_t result;
    nvmlDevice_t device;
    unsigned int power_mW = 0;
    unsigned int validSamples = 0;
};