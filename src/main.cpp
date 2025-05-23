#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>

#include "fmt/format.h"

#include "codec.h"
#include "model.h"
#include "sampler.h"
#include "time.h"
#include "tokenizer.h"

void error_usage() {
	fprintf(stderr, "Usage:   main <checkpoint> [options]\n");
	fprintf(stderr, "Example: main model.yalm -i \"Q: What is the meaning of life?\"\n");
	fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -d [cpu,cuda] which device to use (default - cuda)\n");
  fprintf(stderr, "  -m [completion,perplexity] which mode to run in (default - completion)\n");
  fprintf(stderr, "  Choose one:\n");
	fprintf(stderr, "    -i <string> input prompt\n");
  fprintf(stderr, "    -f <filepath> input file with prompt\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Completion mode options:\n");
  fprintf(stderr, "  -n <int>    number of steps to run for in completion mode, default 256. 0 = max_seq_len, -1 = infinite\n");
	exit(1);
}

// Utility to compute sample mean and std
void compute_mean_std(const std::vector<double> &values, double &mean, double &std_dev) {
    double sum = 0.0;
    for (double val : values) {
        sum += val;
    }
    mean = sum / values.size();

    double sq_sum = 0.0;
    for (double val : values) {
        sq_sum += (val - mean) * (val - mean);
    }
    // Sample standard deviation uses denominator (N-1)
    std_dev = values.size() > 1 ? std::sqrt(sq_sum / (values.size() - 1)) : 0.0;
}


int main(int argc, char* argv[]) {
  std::string checkpoint_path = "";    // e.g. out/model.bin
  // Options
  std::string device = "cuda";         // cpu or cuda
  std::string mode = "completion";     // completion or perplexity
  std::string prompt = "";             // prompt string
  std::string prompt_path = "";        // prompt file path
  // Completion mode options
  int num_steps = 4096;                 // number of steps to run for

  //deli:for avg experiment
  const int NUM_RUNS = 10;
  std::vector<Stats> all_stats;
  all_stats.reserve(NUM_RUNS);

	if (argc >= 2) {
		checkpoint_path = argv[1];
	} else {
		error_usage();
	}
	for (int i = 2; i < argc;) {
		// do some basic validation
		if (i + 1 >= argc) {
			error_usage();
		} // must have arg after flag
		if (argv[i][0] != '-') {
			error_usage();
		} // must start with dash
		if (strlen(argv[i]) != 2) {
			error_usage();
		} // must be -x (one dash, one letter)

		// read in the args
		if (argv[i][1] == 'm') {
      if (i + 1 >= argc) {
        error_usage();
      }
      mode = argv[i + 1];
      if (std::string("completion").starts_with(mode)) {
        mode = "completion";
      } else if (std::string("perplexity").starts_with(mode)) {
        mode = "perplexity";
      } else {
        error_usage();
      }
      i += 2;
    } else if (argv[i][1] == 'd') {
      if (i + 1 >= argc) {
        error_usage();
      }
      device = argv[i + 1];
      if (std::string("cpu").starts_with(device)) {
        device = "cpu";
      } else if (std::string("cuda").starts_with(device)) {
        device = "cuda";
      } else {
        error_usage();
      }
      i += 2;
    } else if (argv[i][1] == 'i') {
      if (i + 1 >= argc) {
        error_usage();
      }
      prompt = argv[i + 1];
      i += 2;
		} else if (argv[i][1] == 'f') {
      if (i + 1 >= argc) {
        error_usage();
      }
      prompt_path = argv[i + 1];
      i += 2;
    } else if (argv[i][1] == 'n') {
      if (i + 1 >= argc) {
        error_usage();
      }
      num_steps = std::stoi(argv[i + 1]);
      i += 2;
    } else {
			error_usage();
		}
	}
  int has_prompt = prompt.size() > 0 ? 1 : 0;
  int has_prompt_path = prompt_path.size() > 0 ? 1 : 0;
  if ((has_prompt + has_prompt_path) != 1) {
    error_usage();
  } else if (has_prompt_path) {
    std::ifstream file(prompt_path);
    if (!file.is_open()) {
      std::cerr << "Error: could not open file " << prompt_path << std::endl;
      return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    prompt = buffer.str();
  }

  YALMData model_data;
  model_data.from_file(checkpoint_path);
  Model model(model_data);
  InferenceState state(model.config);
  Sampler sampler(model.config);
  Tokenizer tokenizer(model_data);

  if (num_steps == 0) {
    // `-n 0` means use the full context length
    num_steps = model.config->max_seq_len;
  }
  if (device == "cuda") {
    std::cout << "Using CUDA" << std::endl;
    model.cuda();
    state.cuda();
  }

  std::cout << "Do one inference as warmup." << std::endl;
  // Do one inference as warmup.
  // On CPU, this ensures all tensors are loaded into memory via mmap.
  // On GPU, this ensures all tensors are loaded into device memory and 
  // kernels are compiled + instantiated.
  model.forward(state, 0, 0);

  std::vector<int> encoding;
  {
    uint64_t encode_start_ms = get_timestamp_ms();
    encoding = tokenizer.encode(prompt, true);
    uint64_t encode_end_ms = get_timestamp_ms();

    std::cout << tokenizer.encoding_to_debug_string(encoding) << std::endl;
    uint64_t encoding_ms = encode_end_ms - encode_start_ms;
    std::cout << fmt::format(
      "Encoding stats: ({} tokens, throughput: {:.5}tok/s, latency: {:.5}s/tok, total: {:.5}s)\n",
      encoding.size(),
      encoding.size() / (encoding_ms / 1000.0),
      (encoding_ms / 1000.0) / encoding.size(),
      encoding_ms / 1000.0
    ) << std::endl;
  }

  if (mode == "completion") {
    for (int run = 0; run < NUM_RUNS; run++) {
      uint64_t start_ms = get_timestamp_ms();
      uint64_t GPU_pwr_start_ms = get_timestamp_ms();
      size_t read_bytes = 0;
      // Hydrate KV cache by forwarding model on all prompt tokens and discarding output.
      // This also generates output logits for the last token.
      for (size_t pos = 0; pos < encoding.size(); pos++) {
        int token_id = encoding[pos];
        InferenceMode inferMode = pos + 1 == encoding.size() ? 
          InferenceMode::OUTPUT_LOGITS : InferenceMode::HYDRATE_KV_CACHE;
        model.forward(state, token_id, pos, inferMode);
        read_bytes += model.config->active_bytes(pos);
      }
      uint64_t end_hydrate_ms = get_timestamp_ms();
      // For N steps:
      // - Sample + decode output logits
      // - Forward the model
      for (int i = 0; i < num_steps || num_steps == -1; i++) {
        int token_id = sampler.sample_argmax(state);
        std::string token_str = tokenizer.decode_one(encoding.back(), token_id);
        std::cout << token_str << std::flush;
        encoding.push_back(token_id);
        if (token_id == tokenizer.eos_id || token_id == tokenizer.eot_id) {
          break;
        }
        model.forward(state, token_id, encoding.size() - 1);
        read_bytes += model.config->active_bytes(encoding.size() - 1);
        if(get_timestamp_ms() - GPU_pwr_start_ms >= 100){
        sampler.collect_GPU_power();
        GPU_pwr_start_ms = get_timestamp_ms();
      }
      }
      std::cout << "\n" << std::endl;
      std::cout << "avg pwr consumption: " << sampler.get_avg_power() << std::endl;  
      std::cout << "\n" << std::endl;
      uint64_t end_ms = get_timestamp_ms();
      double elapsed_s = (end_ms - start_ms) / 1000.0;
      std::cout << fmt::format(
        "Generation stats:\n"
        "  {} tokens\n"
        "  throughput: {:.5}tok/s\n"
        "  latency: {:.5}s/tok\n"
        "  hydrate: {:.5}s\n"
        "  bandwidth: {:.5}GB/s\n"
        "  total: {:.5}s\n"
        "  avg pwr consumption: {:.5}mW\n",
        encoding.size(),
        encoding.size() / elapsed_s,
        elapsed_s / encoding.size(),
        (end_hydrate_ms - start_ms) / 1000.0,
        ((double)read_bytes / 1e9) / elapsed_s,
        elapsed_s,
        sampler.get_avg_power()
      ) << std::endl;
      Stats current_run;
        current_run.tokens     = encoding.size();
        current_run.throughput = encoding.size() / elapsed_s;
        current_run.latency    = elapsed_s / encoding.size();
        current_run.hydrate    = (end_hydrate_ms - start_ms) / 1000.0;
        current_run.bandwidth  = ((double)read_bytes / 1e9) / elapsed_s;
        current_run.total      = elapsed_s;
        current_run.avg_power  = sampler.get_avg_power();
        all_stats.push_back(current_run);
        encoding = tokenizer.encode(prompt, true);
    }
    {
      // Gather each metric into a vector
      std::vector<double> tokens, throughputs, latencies, hydrates, bandwidths, totals, powers;
      tokens.reserve(NUM_RUNS);
      throughputs.reserve(NUM_RUNS);
      latencies.reserve(NUM_RUNS);
      hydrates.reserve(NUM_RUNS);
      bandwidths.reserve(NUM_RUNS);
      totals.reserve(NUM_RUNS);
      powers.reserve(NUM_RUNS);

      for (auto &st : all_stats) {
          tokens.push_back((double)st.tokens);
          throughputs.push_back(st.throughput);
          latencies.push_back(st.latency);
          hydrates.push_back(st.hydrate);
          bandwidths.push_back(st.bandwidth);
          totals.push_back(st.total);
          powers.push_back(st.avg_power);
      }

      double mean_tokens, std_tokens;
      compute_mean_std(tokens, mean_tokens, std_tokens);

      double mean_thr, std_thr;
      compute_mean_std(throughputs, mean_thr, std_thr);

      double mean_lat, std_lat;
      compute_mean_std(latencies, mean_lat, std_lat);

      double mean_hyd, std_hyd;
      compute_mean_std(hydrates, mean_hyd, std_hyd);

      double mean_bnd, std_bnd;
      compute_mean_std(bandwidths, mean_bnd, std_bnd);

      double mean_tot, std_tot;
      compute_mean_std(totals, mean_tot, std_tot);

      double mean_pwr, std_pwr;
      compute_mean_std(powers, mean_pwr, std_pwr);

      std::cout << "===== Final Averages and Std Dev (over 10 runs) =====\n";
      std::cout << fmt::format(
          "Tokens: mean = {:.5f}, std = {:.5f}\n"
          "Throughput (tok/s): mean = {:.5f}, std = {:.5f}\n"
          "Latency (s/tok): mean = {:.5f}, std = {:.5f}\n"
          "Hydrate (s): mean = {:.5f}, std = {:.5f}\n"
          "Bandwidth (GB/s): mean = {:.5f}, std = {:.5f}\n"
          "Total time (s): mean = {:.5f}, std = {:.5f}\n"
          "Avg power (mW): mean = {:.5f}, std = {:.5f}\n",
          mean_tokens,   std_tokens,
          mean_thr,      std_thr,
          mean_lat,      std_lat,
          mean_hyd,      std_hyd,
          mean_bnd,      std_bnd,
          mean_tot,      std_tot,
          mean_pwr,      std_pwr
      );
            std::cout << fmt::format(
          "{:.5f}\n {:.5f}\n"
          "{:.5f}\n {:.5f}\n"
          "{:.5f}\n {:.5f}\n"
          "{:.5f}\n {:.5f}\n"
          "{:.5f}\n {:.5f}\n"
          "{:.5f}\n {:.5f}\n"
          "{:.5f}\n {:.5f}\n",
          mean_tokens,   
          mean_thr,      
          mean_lat,      
          mean_hyd,      
          mean_bnd,     
          mean_tot,      
          mean_pwr,
          std_tokens,
          std_thr,
          std_lat,
          std_hyd,
          std_bnd,
          std_tot,
          std_pwr     
      );
    }
  } else {
    double sum_logprob = 0.0;
    double ss_logprob = 0.0;
    // Generates output logits for all tokens in the prompt and sum log probs to
    // compute perplexity.
    uint64_t start_ms = get_timestamp_ms();
    size_t read_bytes = 0;
    size_t N = encoding.size() - 1;
    for (size_t pos = 0; pos + 1 < encoding.size(); pos++) {
      std::cout << "\r Computing perplexity..." << pos + 1 << "/" << N << std::flush;
      
      int token_id = encoding[pos];
      model.forward(state, token_id, pos);
      read_bytes += model.config->active_bytes(pos);

      double logprob = std::log(sampler.sample_prob(encoding[pos + 1], state));
      sum_logprob += logprob;
      ss_logprob += logprob * logprob;
    }
    std::cout << std::endl;
    uint64_t end_ms = get_timestamp_ms();
    double elapsed_s = (end_ms - start_ms)/1000.0;
    double perplexity = std::exp(-sum_logprob / N);
    double perplexity_error = perplexity * std::sqrt(
      (ss_logprob - sum_logprob * sum_logprob / N) / N / N
    );
    std::cout << fmt::format(
      "Stats:\n"
      "  {} tokens\n"
      "  perplexity: {:.5} ± {:.5}\n"
      "  throughput: {:.5}tok/s\n"
      "  latency: {:.5}s/tok\n"
      "  bandwidth: {:.5}GB/s\n"
      "  total: {:.5}s\n",
      N,
      perplexity,
      perplexity_error,
      N / elapsed_s,
      elapsed_s / N,
      ((double)read_bytes / 1e9) / elapsed_s,
      elapsed_s
    ) << std::endl;
  }

  return 0;
}