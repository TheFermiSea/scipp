#include <benchmark/benchmark.h>
#include <omp.h>

#include <vector>

#include "legacy_cow_ptr.h"
#include "index.h"

class Histogram {
private:
  cow_ptr<std::vector<double>> m_y;

public:
  Histogram(const gsl::index size)
      : m_y(std::make_unique<std::vector<double>>(size)) {}

  Histogram &operator+=(const Histogram &other) {
    auto &ys = m_y.access();
    auto &other_ys = *other.m_y;
    for (gsl::index i = 0; i < ys.size(); ++i)
      ys[i] += other_ys[i];
  }
};

static void BM_Histogram_plus_equals(benchmark::State &state) {
  const int64_t count = 100000000 / state.range(0);
  std::vector<Histogram> histograms(count, 0);
  std::vector<Histogram> histograms2(count, 0);
  const auto size = histograms.size();
  for (gsl::index i = 0; i < size; ++i) {
    // Create without sharing.
    histograms[i] = Histogram(state.range(0));
    histograms2[i] = Histogram(state.range(0));
  }
// Warmup.
#pragma omp parallel for num_threads(state.range(1))
  for (gsl::index i = 0; i < size; ++i) {
    histograms[i] += histograms2[i];
  }
  for (auto _ : state) {
#pragma omp parallel for num_threads(state.range(1))
    for (gsl::index i = 0; i < size; ++i) {
      histograms[i] += histograms2[i];
    }
  }
  state.SetItemsProcessed(state.iterations() * count);
  state.SetBytesProcessed(state.iterations() * count * state.range(0) * 3 *
                          sizeof(double));
}

static void
BM_Histogram_plus_equals_allocation_from_threads(benchmark::State &state) {
  const int64_t count = 100000000 / state.range(0);
  std::vector<Histogram> histograms(count, 0);
  std::vector<Histogram> histograms2(count, 0);
  const auto size = histograms.size();
#pragma omp parallel for num_threads(state.range(1))
  for (gsl::index i = 0; i < size; ++i) {
    // Create without sharing.
    histograms[i] = Histogram(state.range(0));
    histograms2[i] = Histogram(state.range(0));
  }
// Warmup.
#pragma omp parallel for num_threads(state.range(1))
  for (gsl::index i = 0; i < size; ++i) {
    histograms[i] += histograms2[i];
  }
  for (auto _ : state) {
#pragma omp parallel for num_threads(state.range(1))
    for (gsl::index i = 0; i < size; ++i) {
      histograms[i] += histograms2[i];
    }
  }
  state.SetItemsProcessed(state.iterations() * count);
  state.SetBytesProcessed(state.iterations() * count * state.range(0) * 3 *
                          sizeof(double));
}

static void BM_bare_plus_equals(benchmark::State &state) {
  const int64_t count = 100000000 / state.range(0);
  std::vector<double> histograms(count * state.range(0));
  std::vector<double> histograms2(count * state.range(0));
  const auto size = histograms.size();
#pragma omp parallel for num_threads(state.range(1))
  for (gsl::index i = 0; i < size; ++i) {
    histograms[i] += histograms2[i];
  }
  for (auto _ : state) {
#pragma omp parallel for num_threads(state.range(1))
    for (gsl::index i = 0; i < size; ++i) {
      histograms[i] += histograms2[i];
    }
  }
  state.SetItemsProcessed(state.iterations() * count);
  state.SetBytesProcessed(state.iterations() * count * state.range(0) * 3 *
                          sizeof(double));
}

static void BM_bare_plus_equals_no_fork_join(benchmark::State &state) {
  const int64_t count = 1000000;
  gsl::index repeat = 64;
  int num_threads = state.range(1);
  // std::vector<std::vector<double>> histograms(num_threads,
  // std::vector<double>(count * state.range(0)));
  // std::vector<std::vector<double>> histograms2(num_threads,
  // std::vector<double>(count * state.range(0)));
  std::vector<double> histograms(count * state.range(0));
  std::vector<double> histograms2(count * state.range(0));
  for (auto _ : state) {
#pragma omp parallel num_threads(state.range(1))
    {
      const int thread = omp_get_thread_num();
      const int threads = omp_get_num_threads();
      // gsl::index start = 0;
      // gsl::index end = histograms[thread].size();
      gsl::index start = thread * (histograms.size() / threads);
      gsl::index end = start + histograms.size() / threads;
      for (gsl::index r = 0; r < repeat; ++r) {
        for (gsl::index i = start; i < end; ++i) {
          histograms[i] += histograms2[i];
          // histograms[thread][i] += histograms2[thread][i];
        }
      }
    }
  }
  state.SetItemsProcessed(state.iterations() * count * repeat);
  state.SetBytesProcessed(state.iterations() * count * state.range(0) * 3 *
                          sizeof(double) * repeat);
}

BENCHMARK(BM_Histogram_plus_equals)
    ->Args({100, 1})
    ->Args({100, 2})
    ->Args({100, 4})
    ->Args({100, 8})
    ->Args({100, 12})
    ->Args({100, 24})
    ->Args({1000, 1})
    ->Args({1000, 2})
    ->Args({1000, 4})
    ->Args({1000, 8})
    ->Args({1000, 12})
    ->Args({1000, 24})
    ->UseRealTime();

BENCHMARK(BM_Histogram_plus_equals_allocation_from_threads)
    ->Args({100, 1})
    ->Args({100, 2})
    ->Args({100, 4})
    ->Args({100, 8})
    ->Args({100, 12})
    ->Args({100, 24})
    ->Args({1000, 1})
    ->Args({1000, 2})
    ->Args({1000, 4})
    ->Args({1000, 8})
    ->Args({1000, 12})
    ->Args({1000, 24})
    ->UseRealTime();

BENCHMARK(BM_bare_plus_equals)
    ->Args({1000, 1})
    ->Args({1000, 2})
    ->Args({1000, 4})
    ->Args({1000, 8})
    ->Args({1000, 12})
    ->Args({1000, 24})
    ->UseRealTime();

BENCHMARK(BM_bare_plus_equals_no_fork_join)
    ->Args({100, 1})
    ->Args({100, 2})
    ->Args({100, 4})
    ->Args({100, 8})
    ->Args({100, 12})
    ->Args({100, 24})
    ->UseRealTime();

BENCHMARK_MAIN();
