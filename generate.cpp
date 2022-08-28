#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include <cmath>

namespace {

template<typename Rng>
void
generate_dataset(const std::string& path, const float frequency, Rng& rng, bool has_anomaly = false)
{
  const int sample_count = ((48'000) / 1024) * 1024;

  const float sample_rate = 1.0f / 48'000.0f;

  const float signal_amplitude = 0.3f;

  const float noise_amplitude = 0.05f;

  const float anomaly_amplitude = 0.3f;

  std::uniform_real_distribution<float> noise_dist(-noise_amplitude, noise_amplitude);

  std::ofstream data_file(path.c_str());

  for (int i = 0; i < sample_count; i++) {

    const float t = i * sample_rate;

    const float signal = (signal_amplitude * std::sin(t * frequency)) + noise_dist(rng);

    const float anomaly = (anomaly_amplitude * std::sin(t * frequency * 3));

    const float sum = signal + (has_anomaly ? anomaly : 0);

    const float normalized_signal = (sum + 1.0f) * 0.5f;

    data_file << normalized_signal << std::endl;
  }
}

} // namespace

int
main()
{
  std::seed_seq seed{ 1234, 42, 3141 };
  std::mt19937 rng(seed);
  generate_dataset("data/train.csv", 440.0f, rng);
  generate_dataset("data/test.csv", 440.0f, rng);
  generate_dataset("data/anomaly.csv", 440.0f, rng, true);
  return 0;
}