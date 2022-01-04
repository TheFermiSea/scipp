// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "test_macros.h"

#include "scipp/variable/astype.h"
#include "scipp/variable/bins.h"

using namespace scipp;

template <class T> class AsTypeTest : public ::testing::Test {};

using type_pairs =
    ::testing::Types<std::pair<float, double>, std::pair<double, float>,
                     std::pair<int32_t, float>, std::pair<double, double>>;
TYPED_TEST_SUITE(AsTypeTest, type_pairs);

TYPED_TEST(AsTypeTest, dense) {
  using T1 = typename TypeParam::first_type;
  using T2 = typename TypeParam::second_type;
  Variable var1;
  Variable var2;
  if constexpr (core::canHaveVariances<T1>() && core::canHaveVariances<T2>()) {
    var1 = makeVariable<T1>(Values{1}, Variances{1});
    var2 = makeVariable<T2>(Values{1}, Variances{1});
    ASSERT_EQ(astype(var1, core::dtype<T2>), var2);
  }

  var1 = makeVariable<T1>(Values{1});
  var2 = makeVariable<T2>(Values{1});
  ASSERT_EQ(astype(var1, core::dtype<T2>), var2);
  var1 =
      makeVariable<T1>(Dims{Dim::X}, Shape{3}, units::m, Values{1.0, 2.0, 3.0});
  var2 =
      makeVariable<T2>(Dims{Dim::X}, Shape{3}, units::m, Values{1.0, 2.0, 3.0});

  ASSERT_EQ(astype(var1, core::dtype<T2>), var2);
  ASSERT_FALSE(astype(var1, core::dtype<T2>, CopyPolicy::Always).is_same(var1));
  ASSERT_EQ(astype(var1, core::dtype<T2>, CopyPolicy::TryAvoid).is_same(var1),
            (std::is_same_v<T1, T2>));
}

TYPED_TEST(AsTypeTest, binned) {
  using T1 = typename TypeParam::first_type;
  using T2 = typename TypeParam::second_type;
  auto var1 =
      makeVariable<T1>(Dims{Dim::X}, Shape{3}, units::m, Values{1.0, 2.0, 3.0});
  auto var2 =
      makeVariable<T2>(Dims{Dim::X}, Shape{3}, units::m, Values{1.0, 2.0, 3.0});
  auto indices =
      makeVariable<scipp::index_pair>(Values{scipp::index_pair{0, 3}});
  auto binned1 = make_bins(indices, Dim::X, var1);
  auto binned2 = make_bins(indices, Dim::X, var2);
  ASSERT_EQ(astype(binned1, core::dtype<T2>), binned2);
  ASSERT_FALSE(
      astype(binned1, core::dtype<T2>, CopyPolicy::Always).is_same(binned1));
  ASSERT_EQ(
      astype(binned1, core::dtype<T2>, CopyPolicy::TryAvoid).is_same(binned1),
      (std::is_same_v<T1, T2>));
}

TEST(AsTypeTest, buffer_handling) {
  const auto var = makeVariable<float>(Values{1});
  const auto force_copy = astype(var, dtype<float>);
  EXPECT_FALSE(force_copy.is_same(var));
  EXPECT_EQ(force_copy, var);
  const auto force_copy_explicit =
      astype(var, dtype<float>, CopyPolicy::Always);
  EXPECT_FALSE(force_copy_explicit.is_same(var));
  EXPECT_EQ(force_copy_explicit, var);
  const auto no_copy = astype(var, dtype<float>, CopyPolicy::TryAvoid);
  EXPECT_TRUE(no_copy.is_same(var));
  EXPECT_EQ(no_copy, var);
  const auto required_copy = astype(var, dtype<double>, CopyPolicy::TryAvoid);
  EXPECT_FALSE(required_copy.is_same(var));
}
