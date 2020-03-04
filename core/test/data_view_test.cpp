// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include <numeric>

#include "scipp/core/dataset.h"
#include "scipp/core/dimensions.h"
#include "scipp/core/event.h"
#include "test_macros.h"

using namespace scipp;
using namespace scipp::core;

// Using typed tests for common functionality of DataArrayView and
// DataArrayConstView.
template <typename T> class DataArrayViewTest : public ::testing::Test {
protected:
  using dataset_type = std::conditional_t<std::is_same_v<T, DataArrayView>,
                                          Dataset, const Dataset>;
};

using DataArrayViewTypes = ::testing::Types<DataArrayView, DataArrayConstView>;
TYPED_TEST_SUITE(DataArrayViewTest, DataArrayViewTypes);

TYPED_TEST(DataArrayViewTest, name_ignored_in_comparison) {
  const auto var = makeVariable<double>(Values{1.0});
  Dataset d;
  d.setData("a", var);
  d.setData("b", var);
  typename TestFixture::dataset_type &d_ref(d);
  EXPECT_EQ(d_ref["a"], d_ref["b"]);
}

TYPED_TEST(DataArrayViewTest, sparse_sparseDim) {
  Dataset d;
  typename TestFixture::dataset_type &d_ref(d);

  d.setData("dense", makeVariable<double>(Values{double{}}));
  ASSERT_FALSE(is_events(d_ref["dense"]));

  d.setData("sparse_data", makeVariable<event_list<double>>(Dims{}, Shape{}));
  ASSERT_TRUE(is_events(d_ref["sparse_data"]));

  // TODO Event data can have non-list data (only coords would be event_list),
  // what should is_events return in that case?
}

TYPED_TEST(DataArrayViewTest, dims) {
  Dataset d;
  const auto dense = makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{1, 2});
  const auto sparse =
      makeVariable<event_list<double>>(Dims{Dim::X, Dim::Y}, Shape{1, 2});
  typename TestFixture::dataset_type &d_ref(d);

  d.setData("dense", dense);
  ASSERT_EQ(d_ref["dense"].dims(), dense.dims());

  d.setData("sparse_data", sparse);
  ASSERT_EQ(d_ref["sparse_data"].dims(), sparse.dims());
}

TYPED_TEST(DataArrayViewTest, dims_with_extra_coords) {
  Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  const auto x = makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3});
  const auto y = makeVariable<double>(Dims{Dim::Y}, Shape{3}, Values{4, 5, 6});
  const auto var = makeVariable<double>(Dims{Dim::X}, Shape{3});
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y);
  d.setData("a", var);

  ASSERT_EQ(d_ref["a"].dims(), var.dims());
}

TYPED_TEST(DataArrayViewTest, unit) {
  Dataset d;
  typename TestFixture::dataset_type &d_ref(d);

  d.setData("dense", makeVariable<double>(Values{double{}}));
  EXPECT_EQ(d_ref["dense"].unit(), units::dimensionless);
}

// Reenable this test when proper unaligned support available
TYPED_TEST(DataArrayViewTest, DISABLED_unit_access_fails_without_values) {
  Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  DatasetAxis x;
  x.unaligned().set("sparse",
                    makeVariable<event_list<double>>(Dims{}, Shape{}));
  d.coords().set(Dim::X, x);
  // TODO Must set unaligned data, otherwise the item does not exist.
  EXPECT_THROW(d_ref["sparse"].unit(), except::SparseDataError);
}

TYPED_TEST(DataArrayViewTest, coords) {
  Dataset d;
  const auto var = makeVariable<double>(Dims{Dim::X}, Shape{3});
  d.coords().set(Dim::X, var);
  d.setData("a", var);

  typename TestFixture::dataset_type &d_ref(d);
  ASSERT_NO_THROW(d_ref["a"].coords());
  // TODO Require better means to compare DatasetCoords and DataArrayCoords
  ASSERT_EQ(d_ref["a"].coords().size(), d.coords().size());
  for (const auto &[dim, coord] : d_ref["a"].coords())
    EXPECT_EQ(d.coords()[dim].data(), coord.data());
}

TYPED_TEST(DataArrayViewTest, coords_sparse) {
  Dataset d;
  const auto var = makeVariable<event_list<double>>(Dims{Dim::X}, Shape{3});
  d.coords().set(Dim::Y, var);
  d.setData("a", var);

  typename TestFixture::dataset_type &d_ref(d);
  ASSERT_NO_THROW(d_ref["a"].coords());
  // TODO Require better means to compare DatasetCoords and DataArrayCoords
  ASSERT_EQ(d_ref["a"].coords().size(), d.coords().size());
  for (const auto &[dim, coord] : d_ref["a"].coords())
    EXPECT_EQ(d.coords()[dim].data(), coord.data());
  ASSERT_EQ(d_ref["a"].coords().size(), 1);
  ASSERT_NO_THROW(d_ref["a"].coords()[Dim::Y]);
  ASSERT_EQ(d_ref["a"].coords()[Dim::Y], var);
}

// Reenable this when proper unaligned views are implemented
TYPED_TEST(DataArrayViewTest, DISABLED_coords_unaligned) {
  Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  const auto x = makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3});
  const auto y = makeVariable<double>(Dims{Dim::Y}, Shape{3}, Values{4, 5, 6});
  const auto sparse = makeVariable<event_list<double>>(Dims{Dim::X}, Shape{3});
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y);
  d.setUnaligned("a", x);
  DatasetAxis y_axis;
  y_axis.unaligned().set("a", sparse);
  d.coords().set(Dim::Y, sparse);

  ASSERT_NO_THROW(d_ref["a"].coords());
  // TODO Would it be sensible to define such a comparison?
  // ASSERT_NE(d_ref["a"].coords(), d.coords());
  ASSERT_EQ(d_ref["a"].coords().size(), 2);
  ASSERT_NO_THROW(d_ref["a"].coords()[Dim::X]);
  ASSERT_NO_THROW(d_ref["a"].coords()[Dim::Y]);
  ASSERT_EQ(d_ref["a"].coords()[Dim::X], x);
  ASSERT_NE(d_ref["a"].coords()[Dim::Y], y);
  ASSERT_EQ(d_ref["a"].coords()[Dim::Y], sparse);
}

TYPED_TEST(DataArrayViewTest, coords_sparse_shadow_even_if_no_coord) {
  Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  const auto x = makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3});
  const auto y = makeVariable<double>(Dims{Dim::Y}, Shape{3}, Values{4, 5, 6});
  const auto sparse = makeVariable<event_list<double>>(Dims{Dim::X}, Shape{3});
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y);
  d.setData("a", sparse);

  ASSERT_NO_THROW(d_ref["a"].coords());
  // Dim::Y is sparse, so the global (non-sparse) Y coordinate does not make
  // sense and is thus hidden.
  // TODO Would it be sensible to define such a comparison?
  // ASSERT_NE(d_ref["a"].coords(), d.coords());
  ASSERT_EQ(d_ref["a"].coords().size(), 1);
  ASSERT_NO_THROW(d_ref["a"].coords()[Dim::X]);
  ASSERT_ANY_THROW(d_ref["a"].coords()[Dim::Y]);
  ASSERT_EQ(d_ref["a"].coords()[Dim::X], x);
}

TYPED_TEST(DataArrayViewTest, coords_contains_only_relevant) {
  Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  const auto x = makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3});
  const auto y = makeVariable<double>(Dims{Dim::Y}, Shape{3}, Values{4, 5, 6});
  const auto var = makeVariable<double>(Dims{Dim::X}, Shape{3});
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y);
  d.setData("a", var);
  const auto coords = d_ref["a"].coords();

  // TODO Would it be sensible to define such a comparison?
  // ASSERT_NE(coords, d.coords());
  ASSERT_EQ(coords.size(), 1);
  ASSERT_NO_THROW(coords[Dim::X]);
  ASSERT_EQ(coords[Dim::X], x);
}

TYPED_TEST(DataArrayViewTest, coords_contains_only_relevant_2d_dropped) {
  Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  const auto x = makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3});
  const auto y = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
  const auto var = makeVariable<double>(Dims{Dim::X}, Shape{3});
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y);
  d.setData("a", var);
  const auto coords = d_ref["a"].coords();

  // TODO Would it be sensible to define such a comparison?
  // ASSERT_NE(coords, d.coords());
  ASSERT_EQ(coords.size(), 1);
  ASSERT_NO_THROW(coords[Dim::X]);
  ASSERT_EQ(coords[Dim::X], x);
}

TYPED_TEST(DataArrayViewTest,
           coords_contains_only_relevant_2d_not_dropped_inconsistency) {
  Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  const auto x = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
  const auto y = makeVariable<double>(Dims{Dim::Y}, Shape{3});
  const auto var = makeVariable<double>(Dims{Dim::X}, Shape{3});
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y);
  d.setData("a", var);
  const auto coords = d_ref["a"].coords();

  // This is a very special case which is probably unlikely to occur in
  // practice. If the coordinate depends on extra dimensions and the data is
  // not, it implies that the coordinate cannot be for this data item, so it
  // should be dropped... HOWEVER, the current implementation DOES NOT DROP IT.
  // Should that be changed?
  // TODO Would it be sensible to define such a comparison?
  // ASSERT_NE(coords, d.coords());
  ASSERT_EQ(coords.size(), 1);
  ASSERT_NO_THROW(coords[Dim::X]);
  ASSERT_EQ(coords[Dim::X], x);
}

TYPED_TEST(DataArrayViewTest, hasData_hasVariances) {
  Dataset d;
  typename TestFixture::dataset_type &d_ref(d);

  d.setData("a", makeVariable<double>(Values{double{}}));
  d.setData("b", makeVariable<double>(Values{1}, Variances{1}));

  ASSERT_TRUE(d_ref["a"].hasData());
  ASSERT_FALSE(d_ref["a"].hasVariances());

  ASSERT_TRUE(d_ref["b"].hasData());
  ASSERT_TRUE(d_ref["b"].hasVariances());
}

TYPED_TEST(DataArrayViewTest, values_variances) {
  Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  const auto var = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1, 2},
                                        Variances{3, 4});
  d.setData("a", var);

  ASSERT_EQ(d_ref["a"].data(), var);
  ASSERT_TRUE(equals(d_ref["a"].template values<double>(), {1, 2}));
  ASSERT_TRUE(equals(d_ref["a"].template variances<double>(), {3, 4}));
  ASSERT_ANY_THROW(d_ref["a"].template values<float>());
  ASSERT_ANY_THROW(d_ref["a"].template variances<float>());
}

TYPED_TEST(DataArrayViewTest, unaligned_coord_with_no_data) {
  Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  DatasetAxis x;
  x.unaligned().set("a", makeVariable<event_list<double>>(Dims{}, Shape{}));
  d.coords().set(Dim::X, x);

  // TODO Unaligned coord does not imply creation of data item. This feels a bit
  // weird, is there a better way to handle this?
  EXPECT_FALSE(d_ref.contains("a"));
}
