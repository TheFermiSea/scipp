// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/variable/creation.h"
#include "test_macros.h"
#include "test_variables.h"

using namespace scipp;

TEST_P(DenseVariablesTest, empty_like_fail_if_sizes) {
  const auto var = GetParam();
  EXPECT_THROW_DROP_RESULT(
      empty_like(var, {}, makeVariable<scipp::index>(Values{12})),
      except::TypeError);
}

TEST_P(DenseVariablesTest, empty_like_default_shape) {
  const auto var = GetParam();
  const auto empty = empty_like(var);
  EXPECT_EQ(empty.dtype(), var.dtype());
  EXPECT_EQ(empty.dims(), var.dims());
  EXPECT_EQ(empty.unit(), var.unit());
  EXPECT_EQ(empty.hasVariances(), var.hasVariances());
}

TEST_P(DenseVariablesTest, empty_like_slice_default_shape) {
  const auto var = GetParam();
  if (var.dims().contains(Dim::X)) {
    const auto empty = empty_like(var.slice({Dim::X, 0}));
    EXPECT_EQ(empty.dtype(), var.dtype());
    EXPECT_EQ(empty.dims(), var.slice({Dim::X, 0}).dims());
    EXPECT_EQ(empty.unit(), var.unit());
    EXPECT_EQ(empty.hasVariances(), var.hasVariances());
  }
}

TEST_P(DenseVariablesTest, empty_like) {
  const auto var = GetParam();
  const Dimensions dims(Dim::X, 4);
  const auto empty = empty_like(var, dims);
  EXPECT_EQ(empty.dtype(), var.dtype());
  EXPECT_EQ(empty.dims(), dims);
  EXPECT_EQ(empty.unit(), var.unit());
  EXPECT_EQ(empty.hasVariances(), var.hasVariances());
}

TEST(CreationTest, full_like_double) {
  const auto var = makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m,
                                        Values{1, 2}, Variances{3, 4});
  EXPECT_EQ(full_like(var, variable::FillValue::Zero),
            makeVariable<double>(var.dims(), var.unit(), Values{0, 0},
                                 Variances{0, 0}));
  EXPECT_EQ(full_like(var, variable::FillValue::True),
            makeVariable<bool>(var.dims(), var.unit(), Values{true, true}));
  EXPECT_EQ(full_like(var, variable::FillValue::False),
            makeVariable<bool>(var.dims(), var.unit(), Values{false, false}));
  EXPECT_EQ(full_like(var, variable::FillValue::Max),
            makeVariable<double>(var.dims(), var.unit(),
                                 Values{std::numeric_limits<double>::max(),
                                        std::numeric_limits<double>::max()},
                                 Variances{0, 0}));
  EXPECT_EQ(full_like(var, variable::FillValue::Min),
            makeVariable<double>(var.dims(), var.unit(),
                                 Values{std::numeric_limits<double>::min(),
                                        std::numeric_limits<double>::min()},
                                 Variances{0, 0}));
}

TEST(CreationTest, full_like_int) {
  const auto var =
      makeVariable<int64_t>(Dims{Dim::X}, Shape{2}, units::m, Values{1, 2});
  EXPECT_EQ(full_like(var, variable::FillValue::Zero),
            makeVariable<int64_t>(var.dims(), var.unit(), Values{0, 0}));
  EXPECT_EQ(full_like(var, variable::FillValue::True),
            makeVariable<bool>(var.dims(), var.unit(), Values{true, true}));
  EXPECT_EQ(full_like(var, variable::FillValue::False),
            makeVariable<bool>(var.dims(), var.unit(), Values{false, false}));
  EXPECT_EQ(full_like(var, variable::FillValue::Max),
            makeVariable<int64_t>(var.dims(), var.unit(),
                                  Values{std::numeric_limits<int64_t>::max(),
                                         std::numeric_limits<int64_t>::max()}));
  EXPECT_EQ(full_like(var, variable::FillValue::Min),
            makeVariable<int64_t>(var.dims(), var.unit(),
                                  Values{std::numeric_limits<int64_t>::min(),
                                         std::numeric_limits<int64_t>::min()}));
}
