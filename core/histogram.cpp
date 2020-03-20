// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "scipp/core/histogram.h"
#include "scipp/common/numeric.h"
#include "scipp/core/dataset.h"
#include "scipp/core/except.h"
#include "scipp/core/transform_subspan.h"

#include "dataset_operations_common.h"

namespace scipp::core {

static constexpr auto make_histogram =
    [](auto &data, const auto &events, const auto &weights, const auto &edges) {
      constexpr auto value = [](const auto &v, const scipp::index i) {
        if constexpr (is_ValueAndVariance_v<std::decay_t<decltype(v)>>) {
          static_cast<void>(i);
          return v.value;
        } else
          return v.values[i];
      };
      constexpr auto variance = [](const auto &v, const scipp::index i) {
        if constexpr (is_ValueAndVariance_v<std::decay_t<decltype(v)>>) {
          static_cast<void>(i);
          return v.variance;
        } else
          return v.variances[i];
      };

      // Special implementation for linear bins. Gives a 1x to 20x speedup
      // for few and many events per histogram, respectively.
      if (scipp::numeric::is_linspace(edges)) {
        const auto [offset, nbin, scale] = linear_edge_params(edges);
        for (scipp::index j = 0; j < scipp::size(events); ++j) {
          const auto x = events[j];
          const double bin = (x - offset) * scale;
          if (bin >= 0.0 && bin < nbin) {
            const auto b = static_cast<scipp::index>(bin);
            const auto w = value(weights, j);
            const auto e = variance(weights, j);
            data.value[b] += w;
            data.variance[b] += e;
          }
        }
      } else {
        expect::histogram::sorted_edges(edges);
        for (scipp::index j = 0; j < scipp::size(events); ++j) {
          const auto x = events[j];
          auto it = std::upper_bound(edges.begin(), edges.end(), x);
          if (it != edges.end() && it != edges.begin()) {
            const auto b = --it - edges.begin();
            const auto w = value(weights, j);
            const auto e = variance(weights, j);
            data.value[b] += w;
            data.variance[b] += e;
          }
        }
      }
    };

static constexpr auto make_histogram_unit = [](const units::Unit &sparse_unit,
                                               const units::Unit &weights_unit,
                                               const units::Unit &edge_unit) {
  if (sparse_unit != edge_unit)
    throw except::UnitError("Bin edges must have same unit as the sparse "
                            "input coordinate.");
  if (weights_unit != units::counts && weights_unit != units::dimensionless)
    throw except::UnitError("Weights of sparse data must be "
                            "`units::counts` or `units::dimensionless`.");
  return weights_unit;
};

namespace histogram_detail {
template <class Out, class Coord, class Edge>
using args = std::tuple<span<Out>, event_list<Coord>, span<const Edge>>;
}
namespace histogram_weighted_detail {
template <class Out, class Coord, class Weight, class Edge>
using args = std::tuple<span<Out>, event_list<Coord>, Weight, span<const Edge>>;
}

DataArray histogram(const DataArrayConstView &sparse,
                    const VariableConstView &binEdges) {
  auto dim = binEdges.dims().inner();

  auto result = apply_and_drop_dim(
      sparse,
      [](const DataArrayConstView &sparse_, const Dim dim_,
         const VariableConstView &binEdges_) {
        using namespace histogram_weighted_detail;
        // This supports scalar weights as well as event_list weights.
        return transform_subspan<
            std::tuple<args<double, double, double, double>,
                       args<double, float, double, double>,
                       args<double, float, double, float>,
                       args<double, double, float, double>,
                       args<double, double, event_list<double>, double>,
                       args<double, float, event_list<double>, double>,
                       args<double, float, event_list<double>, float>,
                       args<double, double, event_list<float>, double>>>(
            dim_, binEdges_.dims()[dim_] - 1, sparse_.coords()[dim_],
            sparse_.data(), binEdges_,
            overloaded{make_histogram, make_histogram_unit,
                       transform_flags::expect_variance_arg<0>,
                       transform_flags::expect_no_variance_arg<1>,
                       transform_flags::expect_variance_arg<2>,
                       transform_flags::expect_no_variance_arg<3>});
      },
      dim, binEdges);
  result.setCoord(dim, binEdges);
  return result;
}

DataArray histogram(const DataArrayConstView &sparse,
                    const Variable &binEdges) {
  return histogram(sparse, VariableConstView(binEdges));
}

Dataset histogram(const Dataset &dataset, const VariableConstView &bins) {
  auto out(Dataset(DatasetConstView::makeViewWithEmptyIndexes(dataset)));
  const Dim dim = bins.dims().inner();
  out.setCoord(dim, bins);
  for (const auto &item : dataset) {
    if (is_events(item.coords()[dim]))
      out.setData(item.name(), histogram(item, bins));
  }
  return out;
}

Dataset histogram(const Dataset &dataset, const Dim &dim) {
  auto bins = dataset.coords()[dim];
  if (is_events(bins))
    throw except::BinEdgeError("Expected bin edges, got event data.");
  return histogram(dataset, bins);
}

/// Return true if the data array respresents a histogram for given dim.
bool is_histogram(const DataArrayConstView &a, const Dim dim) {
  const auto dims = a.dims();
  const auto coords = a.coords();
  return dims.contains(dim) && coords.contains(dim) &&
         coords[dim].dims().contains(dim) &&
         coords[dim].dims()[dim] == dims[dim] + 1;
}

} // namespace scipp::core
