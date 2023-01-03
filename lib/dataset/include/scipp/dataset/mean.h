// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

/*
 * These functions are not generated by CMake because they require
 * slightly different operations from other reductions.
 */

#include "scipp/dataset/dataset.h"

namespace scipp::dataset {

SCIPP_DATASET_EXPORT DataArray mean(const DataArray &a, const Dim dim);
SCIPP_DATASET_EXPORT DataArray mean(const DataArray &a);
SCIPP_DATASET_EXPORT Dataset mean(const Dataset &d, const Dim dim);
SCIPP_DATASET_EXPORT Dataset mean(const Dataset &d);

} // namespace scipp::dataset
