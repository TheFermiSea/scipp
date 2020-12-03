// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <vector>

#include <scipp/dataset/dataset.h>
#include <scipp/variable/util.h>
#include <scipp/variable/variable.h>

using scipp::variable::SortOrder;

namespace scipp::dataset {
SCIPP_DATASET_EXPORT Variable
sort(const VariableConstView &var, const VariableConstView &key,
     const SortOrder &order = SortOrder::Ascending);
SCIPP_DATASET_EXPORT DataArray
sort(const DataArrayConstView &array, const VariableConstView &key,
     const SortOrder &order = SortOrder::Ascending);
SCIPP_DATASET_EXPORT DataArray
sort(const DataArrayConstView &array, const Dim &key,
     const SortOrder &order = SortOrder::Ascending);
SCIPP_DATASET_EXPORT Dataset
sort(const DatasetConstView &dataset, const VariableConstView &key,
     const SortOrder &order = SortOrder::Ascending);
SCIPP_DATASET_EXPORT Dataset
sort(const DatasetConstView &dataset, const Dim &key,
     const SortOrder &order = SortOrder::Ascending);

} // namespace scipp::dataset
