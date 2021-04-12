// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/dataset/dataset.h"
#include "scipp/dataset/except.h"

using namespace scipp;
using namespace scipp::dataset;

struct AssignTest : public ::testing::Test {
  AssignTest() {}

protected:
  Dimensions dims{Dim::X, 3};
  Variable data = makeVariable<double>(dims, Values{1, 2, 3});
  Variable x = makeVariable<double>(dims, Values{1, 1, 3});
  Variable mask = makeVariable<bool>(dims, Values{true, false, true});
  DataArray array{data, {{Dim::X, copy(x)}}, {{"mask", mask}}};
};

TEST_F(AssignTest, self) {
  const auto original = copy(array);
  EXPECT_EQ(array.assign(array), original);
}

TEST_F(AssignTest, coord_fail) {
  const auto original = copy(array);
  EXPECT_THROW(array.assign(array.slice({Dim::X, 0, 1})),
               except::CoordMismatchError);
  EXPECT_EQ(array, original);
  EXPECT_THROW(array.slice({Dim::X, 0, 1}).assign(array.slice({Dim::X, 2, 3})),
               except::CoordMismatchError);
  EXPECT_EQ(array, original);
}

TEST_F(AssignTest, mask_propagation) {
  const auto original = copy(array);
  // Mask values get copied
  array.slice({Dim::X, 0}).assign(original.slice({Dim::X, 1}));
  EXPECT_EQ(array.masks()["mask"],
            makeVariable<bool>(dims, Values{false, false, true}));
  array.slice({Dim::X, 0}).assign(original.slice({Dim::X, 2}));
  EXPECT_EQ(array.masks()["mask"],
            makeVariable<bool>(dims, Values{true, false, true}));
  // Mask not in source is preserved unchanged
  array.masks().set("other", mask);
  array.slice({Dim::X, 0}).assign(original.slice({Dim::X, 1}));
  EXPECT_EQ(array.masks()["mask"], mask);
  // Extra mask is added
  auto extra_mask = copy(array);
  extra_mask.masks().set("extra", mask);
  array.assign(extra_mask.slice({Dim::X, 1}));
  EXPECT_TRUE(array.masks().contains("extra"));
  // Extra masks added to mask dict of slice => silently dropped
  extra_mask.masks().set("dropped", mask);
  EXPECT_NO_THROW(
      array.slice({Dim::X, 0}).assign(extra_mask.slice({Dim::X, 1})));
  EXPECT_FALSE(array.masks().contains("dropped"));
}
