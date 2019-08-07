// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
#include "test_macros.h"
#include <gtest/gtest.h>

#include <numeric>
#include <set>

#include "scipp/core/dataset.h"
#include "scipp/core/dimensions.h"

#include "dataset_test_common.h"

using namespace scipp;
using namespace scipp::core;

// Any dataset functionality that is also available for Dataset(Const)Proxy is
// to be tested in dataset_proxy_test.cpp, not here!

TEST(DatasetTest, construct_default) { ASSERT_NO_THROW(Dataset d); }

TEST(DatasetTest, clear) {
  DatasetFactory3D factory;
  auto dataset = factory.make();
  ASSERT_FALSE(dataset.empty());
  ASSERT_FALSE(dataset.coords().empty());
  ASSERT_FALSE(dataset.labels().empty());
  ASSERT_FALSE(dataset.attrs().empty());
  ASSERT_NO_THROW(dataset.clear());
  ASSERT_TRUE(dataset.empty());
  ASSERT_FALSE(dataset.coords().empty());
  ASSERT_FALSE(dataset.labels().empty());
  ASSERT_FALSE(dataset.attrs().empty());
}

TEST(DatasetTest, erase_single) {
  DatasetFactory3D factory;
  auto dataset = factory.make();
  ASSERT_EQ(1, dataset.erase("data_xyz"));
  ASSERT_FALSE(dataset.contains("data_xyz"));
}

TEST(DatasetTest, setCoord) {
  Dataset d;
  const auto var = makeVariable<double>({Dim::X, 3});

  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.coords().size(), 0);

  ASSERT_NO_THROW(d.setCoord(Dim::X, var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.coords().size(), 1);

  ASSERT_NO_THROW(d.setCoord(Dim::Y, var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.coords().size(), 2);

  ASSERT_NO_THROW(d.setCoord(Dim::X, var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.coords().size(), 2);
}

TEST(DatasetTest, setLabels) {
  Dataset d;
  const auto var = makeVariable<double>({Dim::X, 3});

  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.labels().size(), 0);

  ASSERT_NO_THROW(d.setLabels("a", var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.labels().size(), 1);

  ASSERT_NO_THROW(d.setLabels("b", var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.labels().size(), 2);

  ASSERT_NO_THROW(d.setLabels("a", var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.labels().size(), 2);
}

TEST(DatasetTest, setAttr) {
  Dataset d;
  const auto var = makeVariable<double>({Dim::X, 3});

  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.attrs().size(), 0);

  ASSERT_NO_THROW(d.setAttr("a", var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.attrs().size(), 1);

  ASSERT_NO_THROW(d.setAttr("b", var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.attrs().size(), 2);

  ASSERT_NO_THROW(d.setAttr("a", var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.attrs().size(), 2);
}

TEST(DatasetTest, setData_with_and_without_variances) {
  Dataset d;
  const auto var = makeVariable<double>({Dim::X, 3});

  ASSERT_NO_THROW(d.setData("a", var));
  ASSERT_EQ(d.size(), 1);

  ASSERT_NO_THROW(d.setData("b", var));
  ASSERT_EQ(d.size(), 2);

  ASSERT_NO_THROW(d.setData("a", var));
  ASSERT_EQ(d.size(), 2);

  ASSERT_NO_THROW(
      d.setData("a", makeVariable<double>({Dim::X, 3}, {1, 1, 1}, {0, 0, 0})));
  ASSERT_EQ(d.size(), 2);
}

TEST(DatasetTest, setLabels_with_name_matching_data_name) {
  Dataset d;
  d.setData("a", makeVariable<double>({Dim::X, 3}));
  d.setData("b", makeVariable<double>({Dim::X, 3}));

  // It is possible to set labels with a name matching data. However, there is
  // no special meaning attached to this. In particular it is *not* linking the
  // labels to that data item.
  ASSERT_NO_THROW(d.setLabels("a", makeVariable<double>({})));
  ASSERT_EQ(d.size(), 2);
  ASSERT_EQ(d.labels().size(), 1);
  ASSERT_EQ(d["a"].labels().size(), 1);
  ASSERT_EQ(d["b"].labels().size(), 1);
}

TEST(DatasetTest, setSparseCoord_not_sparse_fail) {
  Dataset d;
  const auto var = makeVariable<double>({Dim::X, 3});

  ASSERT_ANY_THROW(d.setSparseCoord("a", var));
}

TEST(DatasetTest, setSparseCoord) {
  Dataset d;
  const auto var =
      makeVariable<double>({Dim::X, Dim::Y}, {3, Dimensions::Sparse});

  ASSERT_NO_THROW(d.setSparseCoord("a", var));
  ASSERT_EQ(d.size(), 1);
  ASSERT_NO_THROW(d["a"]);
}

TEST(DatasetTest, setSparseLabels_missing_values_or_coord) {
  Dataset d;
  const auto sparse = makeVariable<double>({Dim::X}, {Dimensions::Sparse});

  ASSERT_ANY_THROW(d.setSparseLabels("a", "x", sparse));
  d.setSparseCoord("a", sparse);
  ASSERT_NO_THROW(d.setSparseLabels("a", "x", sparse));
}

TEST(DatasetTest, setSparseLabels_not_sparse_fail) {
  Dataset d;
  const auto dense = makeVariable<double>({});
  const auto sparse = makeVariable<double>({Dim::X}, {Dimensions::Sparse});

  d.setSparseCoord("a", sparse);
  ASSERT_ANY_THROW(d.setSparseLabels("a", "x", dense));
}

TEST(DatasetTest, setSparseLabels) {
  Dataset d;
  const auto sparse = makeVariable<double>({Dim::X}, {Dimensions::Sparse});
  d.setSparseCoord("a", sparse);

  ASSERT_NO_THROW(d.setSparseLabels("a", "x", sparse));
  ASSERT_EQ(d.size(), 1);
  ASSERT_NO_THROW(d["a"]);
  ASSERT_EQ(d["a"].labels().size(), 1);
}

TEST(DatasetTest, iterators_return_types) {
  Dataset d;
  ASSERT_TRUE((std::is_same_v<decltype(d.begin()->second), DataProxy>));
  ASSERT_TRUE((std::is_same_v<decltype(d.end()->second), DataProxy>));
}

TEST(DatasetTest, const_iterators_return_types) {
  const Dataset d;
  ASSERT_TRUE((std::is_same_v<decltype(d.begin()->second), DataConstProxy>));
  ASSERT_TRUE((std::is_same_v<decltype(d.end()->second), DataConstProxy>));
}

TEST(DatasetTest, set_dense_data_with_sparse_coord) {
  auto sparse_variable =
      makeVariable<double>({Dim::Y, Dim::X}, {2, Dimensions::Sparse});
  auto dense_variable = makeVariable<double>({Dim::Y, Dim::X}, {2, 2});

  Dataset a;
  a.setData("sparse_coord_and_val", dense_variable);
  ASSERT_THROW(a.setSparseCoord("sparse_coord_and_val", sparse_variable),
               except::DimensionError);

  // Setting coords first yields same response.
  Dataset b;
  b.setSparseCoord("sparse_coord_and_val", sparse_variable);
  ASSERT_THROW(b.setData("sparse_coord_and_val", dense_variable),
               except::DimensionError);
}

TEST(DatasetTest, construct_from_proxy) {
  DatasetFactory3D factory;
  const auto dataset = factory.make();
  const DatasetConstProxy proxy(dataset);
  Dataset from_proxy(proxy);
  ASSERT_EQ(from_proxy, dataset);
}

TEST(DatasetTest, construct_from_slice) {
  DatasetFactory3D factory;
  const auto dataset = factory.make();
  const auto slice = dataset.slice({Dim::X, 1});
  Dataset from_slice(slice);
  ASSERT_EQ(from_slice, dataset.slice({Dim::X, 1}));
}

TEST(DatasetTest, slice_temporary) {
  DatasetFactory3D factory;
  auto dataset = factory.make().slice({Dim::X, 1});
  ASSERT_TRUE((std::is_same_v<decltype(dataset), Dataset>));
}
