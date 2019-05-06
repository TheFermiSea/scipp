// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
#include "test_macros.h"
#include <gtest/gtest.h>

#include <numeric>

#include "dataset_next.h"
#include "dimensions.h"

using namespace scipp;
using namespace scipp::core;
using namespace scipp::core::next;

TEST(DatasetNext, construct_default) { ASSERT_NO_THROW(next::Dataset d); }

TEST(DatasetNext, empty) {
  next::Dataset d;
  ASSERT_TRUE(d.empty());
  ASSERT_EQ(d.size(), 0);
}

TEST(DatasetNext, coords) {
  next::Dataset d;
  ASSERT_NO_THROW(d.coords());
}

TEST(DatasetNext, labels) {
  next::Dataset d;
  ASSERT_NO_THROW(d.labels());
}

TEST(DatasetNext, attrs) {
  next::Dataset d;
  ASSERT_NO_THROW(d.attrs());
}

TEST(DatasetNext, bad_item_access) {
  next::Dataset d;
  ASSERT_ANY_THROW(d[""]);
  ASSERT_ANY_THROW(d["abc"]);
}

TEST(DatasetNext, setCoord) {
  next::Dataset d;
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

TEST(DatasetNext, setLabels) {
  next::Dataset d;
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

TEST(DatasetNext, setAttr) {
  next::Dataset d;
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

TEST(DatasetNext, setValues_setVariances) {
  next::Dataset d;
  const auto var = makeVariable<double>({Dim::X, 3});

  ASSERT_NO_THROW(d.setValues("a", var));
  ASSERT_EQ(d.size(), 1);

  ASSERT_NO_THROW(d.setValues("b", var));
  ASSERT_EQ(d.size(), 2);

  ASSERT_NO_THROW(d.setValues("a", var));
  ASSERT_EQ(d.size(), 2);

  ASSERT_NO_THROW(d.setVariances("a", var));
  ASSERT_EQ(d.size(), 2);

  ASSERT_ANY_THROW(d.setVariances("c", var));
}
TEST(DatasetNext, setLabels_with_name_matching_data_name) {
  next::Dataset d;
  d.setValues("a", makeVariable<double>({Dim::X, 3}));
  d.setValues("b", makeVariable<double>({Dim::X, 3}));

  // It is possible to set labels with a name matching data. However, there is
  // no special meaning attached to this. In particular it is *not* linking the
  // labels to that data item.
  ASSERT_NO_THROW(d.setLabels("a", makeVariable<double>({})));
  ASSERT_EQ(d.size(), 2);
  ASSERT_EQ(d.labels().size(), 1);
  ASSERT_EQ(d["a"].labels().size(), 1);
  ASSERT_EQ(d["b"].labels().size(), 1);
}

TEST(DatasetNext, setVariances_dtype_mismatch) {
  next::Dataset d;
  d.setValues("", makeVariable<double>({}));
  ASSERT_ANY_THROW(d.setVariances("", makeVariable<float>({})));
  ASSERT_NO_THROW(d.setVariances("", makeVariable<double>({})));
}

TEST(DatasetNext, setVariances_unit_mismatch) {
  next::Dataset d;
  auto values = makeVariable<double>({});
  values.setUnit(units::m);
  d.setValues("", values);
  auto variances = makeVariable<double>({});
  ASSERT_ANY_THROW(d.setVariances("", variances));
  variances.setUnit(units::m);
  ASSERT_ANY_THROW(d.setVariances("", variances));
  variances.setUnit(units::m * units::m);
  ASSERT_NO_THROW(d.setVariances("", variances));
}

TEST(DatasetNext, setVariances_dimensions_mismatch) {
  next::Dataset d;
  d.setValues("", makeVariable<double>({}));
  ASSERT_ANY_THROW(d.setVariances("", makeVariable<double>({Dim::X, 1})));
  ASSERT_NO_THROW(d.setVariances("", makeVariable<double>({})));
}

TEST(DatasetNext, setVariances_sparseDim_mismatch) {
  next::Dataset d;
  d.setValues("", makeSparseVariable<double>({}, Dim::X));
  ASSERT_ANY_THROW(d.setVariances("", makeVariable<double>({Dim::X, 1})));
  ASSERT_ANY_THROW(d.setVariances("", makeVariable<double>({})));
  ASSERT_ANY_THROW(d.setVariances("", makeSparseVariable<double>({}, Dim::Y)));
  ASSERT_ANY_THROW(
      d.setVariances("", makeSparseVariable<double>({Dim::X, 1}, Dim::X)));
  ASSERT_NO_THROW(d.setVariances("", makeSparseVariable<double>({}, Dim::X)));
}

TEST(DatasetNext, setValues_dtype_mismatch) {
  next::Dataset d;
  d.setValues("", makeVariable<double>({}));
  d.setVariances("", makeVariable<double>({}));
  ASSERT_ANY_THROW(d.setValues("", makeVariable<float>({})));
  ASSERT_NO_THROW(d.setValues("", makeVariable<double>({})));
}

TEST(DatasetNext, setValues_dimensions_mismatch) {
  next::Dataset d;
  d.setValues("", makeVariable<double>({}));
  d.setVariances("", makeVariable<double>({}));
  ASSERT_ANY_THROW(d.setValues("", makeVariable<double>({Dim::X, 1})));
  ASSERT_NO_THROW(d.setValues("", makeVariable<double>({})));
}

TEST(DatasetNext, setValues_sparseDim_mismatch) {
  next::Dataset d;
  d.setValues("", makeSparseVariable<double>({}, Dim::X));
  d.setVariances("", makeSparseVariable<double>({}, Dim::X));
  ASSERT_ANY_THROW(d.setValues("", makeVariable<double>({Dim::X, 1})));
  ASSERT_ANY_THROW(d.setValues("", makeVariable<double>({})));
  ASSERT_ANY_THROW(d.setValues("", makeSparseVariable<double>({}, Dim::Y)));
  ASSERT_ANY_THROW(
      d.setValues("", makeSparseVariable<double>({Dim::X, 1}, Dim::X)));
  ASSERT_NO_THROW(d.setValues("", makeSparseVariable<double>({}, Dim::X)));
}

TEST(DatasetNext, setSparseCoord_not_sparse_fail) {
  next::Dataset d;
  const auto var = makeVariable<double>({Dim::X, 3});

  ASSERT_ANY_THROW(d.setSparseCoord("a", var));
}

TEST(DatasetNext, setSparseCoord) {
  next::Dataset d;
  const auto var = makeSparseVariable<double>({Dim::X, 3}, Dim::Y);

  ASSERT_NO_THROW(d.setSparseCoord("a", var));
  ASSERT_EQ(d.size(), 1);
  ASSERT_NO_THROW(d["a"]);
}

TEST(DatasetNext, setSparseLabels_missing_values_or_coord) {
  next::Dataset d;
  const auto sparse = makeSparseVariable<double>({}, Dim::X);

  ASSERT_ANY_THROW(d.setSparseLabels("a", "x", sparse));
  d.setSparseCoord("a", sparse);
  ASSERT_NO_THROW(d.setSparseLabels("a", "x", sparse));
}

TEST(DatasetNext, setSparseLabels_not_sparse_fail) {
  next::Dataset d;
  const auto dense = makeVariable<double>({});
  const auto sparse = makeSparseVariable<double>({}, Dim::X);

  d.setSparseCoord("a", sparse);
  ASSERT_ANY_THROW(d.setSparseLabels("a", "x", dense));
}

TEST(DatasetNext, setSparseLabels) {
  next::Dataset d;
  const auto sparse = makeSparseVariable<double>({}, Dim::X);
  d.setSparseCoord("a", sparse);

  ASSERT_NO_THROW(d.setSparseLabels("a", "x", sparse));
  ASSERT_EQ(d.size(), 1);
  ASSERT_NO_THROW(d["a"]);
  ASSERT_EQ(d["a"].labels().size(), 1);
}

TEST(DatasetNext, iterators_empty_dataset) {
  next::Dataset d;
  ASSERT_NO_THROW(d.begin());
  ASSERT_NO_THROW(d.end());
  EXPECT_EQ(d.begin(), d.end());
}

TEST(DatasetNext, iterators_only_coords) {
  next::Dataset d;
  d.setCoord(Dim::X, makeVariable<double>({}));
  ASSERT_NO_THROW(d.begin());
  ASSERT_NO_THROW(d.end());
  EXPECT_EQ(d.begin(), d.end());
}

TEST(DatasetNext, iterators_only_labels) {
  next::Dataset d;
  d.setLabels("a", makeVariable<double>({}));
  ASSERT_NO_THROW(d.begin());
  ASSERT_NO_THROW(d.end());
  EXPECT_EQ(d.begin(), d.end());
}

TEST(DatasetNext, iterators_only_attrs) {
  next::Dataset d;
  d.setAttr("a", makeVariable<double>({}));
  ASSERT_NO_THROW(d.begin());
  ASSERT_NO_THROW(d.end());
  EXPECT_EQ(d.begin(), d.end());
}

TEST(DatasetNext, iterators) {
  next::Dataset d;
  d.setValues("a", makeVariable<double>({}));
  d.setValues("b", makeVariable<float>({}));
  d.setValues("c", makeVariable<int64_t>({}));

  ASSERT_NO_THROW(d.begin());
  ASSERT_NO_THROW(d.end());

  auto it = d.begin();
  ASSERT_NE(it, d.end());
  EXPECT_EQ(it->first, "a");

  ASSERT_NO_THROW(++it);
  ASSERT_NE(it, d.end());
  EXPECT_EQ(it->first, "b");

  ASSERT_NO_THROW(++it);
  ASSERT_NE(it, d.end());
  EXPECT_EQ(it->first, "c");

  ASSERT_NO_THROW(++it);
  ASSERT_EQ(it, d.end());
}

TEST(DatasetNext, iterators_return_types) {
  next::Dataset d;
  ASSERT_TRUE((std::is_same_v<decltype(d.begin()->second), DataProxy>));
  ASSERT_TRUE((std::is_same_v<decltype(d.end()->second), DataProxy>));
}

TEST(DatasetNext, const_iterators_return_types) {
  const next::Dataset d;
  ASSERT_TRUE((std::is_same_v<decltype(d.begin()->second), DataConstProxy>));
  ASSERT_TRUE((std::is_same_v<decltype(d.end()->second), DataConstProxy>));
}

template <class T, class T2>
auto variable(const Dimensions &dims, const units::Unit unit,
              const std::initializer_list<T2> &data) {
  auto var = makeVariable<T>(dims, data);
  var.setUnit(unit);
  return var;
}

class Dataset_comparison_operators : public ::testing::Test {
private:
  template <class A, class B>
  void expect_eq_impl(const A &a, const B &b) const {
    EXPECT_TRUE(a == b);
    EXPECT_TRUE(b == a);
    EXPECT_FALSE(a != b);
    EXPECT_FALSE(b != a);
  }
  template <class A, class B>
  void expect_ne_impl(const A &a, const B &b) const {
    EXPECT_TRUE(a != b);
    EXPECT_TRUE(b != a);
    EXPECT_FALSE(a == b);
    EXPECT_FALSE(b == a);
  }

protected:
  Dataset_comparison_operators()
      : sparse_variable(
            makeSparseVariable<double>({{Dim::Y, 3}, {Dim::Z, 2}}, Dim::X)) {
    dataset.setCoord(Dim::X, makeVariable<double>({Dim::X, 4}));
    dataset.setCoord(Dim::Y, makeVariable<double>({Dim::Y, 2}));

    dataset.setLabels("labels", makeVariable<int>({Dim::X, 4}));

    dataset.setAttr("attr", makeVariable<int>({}));

    dataset.setValues("val_and_var",
                      makeVariable<double>({{Dim::Y, 3}, {Dim::X, 4}}));
    dataset.setVariances("val_and_var",
                         makeVariable<double>({{Dim::Y, 3}, {Dim::X, 4}}));

    dataset.setValues("val", makeVariable<double>({Dim::X, 4}));

    dataset.setSparseCoord("sparse_coord", sparse_variable);
    dataset.setValues("sparse_coord_and_val", sparse_variable);
    dataset.setSparseCoord("sparse_coord_and_val", sparse_variable);
  }
  void expect_eq(const next::Dataset &a, const next::Dataset &b) const {
    expect_eq_impl(a, DatasetConstProxy(b));
    expect_eq_impl(DatasetConstProxy(a), b);
    expect_eq_impl(DatasetConstProxy(a), DatasetConstProxy(b));
  }
  void expect_ne(const next::Dataset &a, const next::Dataset &b) const {
    expect_ne_impl(a, DatasetConstProxy(b));
    expect_ne_impl(DatasetConstProxy(a), b);
    expect_ne_impl(DatasetConstProxy(a), DatasetConstProxy(b));
  }

  next::Dataset dataset;
  Variable sparse_variable;
};

auto make_empty() { return next::Dataset(); };

template <class T, class T2>
auto make_1_coord(const Dim dim, const Dimensions &dims, const units::Unit unit,
                  const std::initializer_list<T2> &data) {
  auto d = make_empty();
  d.setCoord(dim, variable<T>(dims, unit, data));
  return d;
}

template <class T, class T2>
auto make_1_labels(const std::string &name, const Dimensions &dims,
                   const units::Unit unit,
                   const std::initializer_list<T2> &data) {
  auto d = make_empty();
  d.setLabels(name, variable<T>(dims, unit, data));
  return d;
}

template <class T, class T2>
auto make_1_attr(const std::string &name, const Dimensions &dims,
                 const units::Unit unit,
                 const std::initializer_list<T2> &data) {
  auto d = make_empty();
  d.setAttr(name, variable<T>(dims, unit, data));
  return d;
}

template <class T, class T2>
auto make_1_values(const std::string &name, const Dimensions &dims,
                   const units::Unit unit,
                   const std::initializer_list<T2> &data) {
  auto d = make_empty();
  d.setValues(name, variable<T>(dims, unit, data));
  return d;
}

template <class T, class T2>
auto make_1_values_and_variances(const std::string &name,
                                 const Dimensions &dims, const units::Unit unit,
                                 const std::initializer_list<T2> &values,
                                 const std::initializer_list<T2> &variances) {
  auto d = make_empty();
  d.setValues(name, variable<T>(dims, unit, values));
  d.setVariances(name, variable<T>(dims, unit * unit, variances));
  return d;
}

// Baseline checks: Does dataset comparison pick up arbitrary mismatch of
// individual items? Strictly speaking many of these are just retesting the
// comparison of Variable, but it ensures that the content is actually compared
// and thus serves as a baseline for the follow-up tests.
TEST_F(Dataset_comparison_operators, single_coord) {
  auto d = make_1_coord<double>(Dim::X, {Dim::X, 3}, units::m, {1, 2, 3});
  expect_eq(d, d);
  expect_ne(d, make_empty());
  expect_ne(d, make_1_coord<float>(Dim::X, {Dim::X, 3}, units::m, {1, 2, 3}));
  expect_ne(d, make_1_coord<double>(Dim::Y, {Dim::X, 3}, units::m, {1, 2, 3}));
  expect_ne(d, make_1_coord<double>(Dim::X, {Dim::Y, 3}, units::m, {1, 2, 3}));
  expect_ne(d, make_1_coord<double>(Dim::X, {Dim::X, 2}, units::m, {1, 2}));
  expect_ne(d, make_1_coord<double>(Dim::X, {Dim::X, 3}, units::s, {1, 2, 3}));
  expect_ne(d, make_1_coord<double>(Dim::X, {Dim::X, 3}, units::m, {1, 2, 4}));
}

TEST_F(Dataset_comparison_operators, single_labels) {
  auto d = make_1_labels<double>("a", {Dim::X, 3}, units::m, {1, 2, 3});
  expect_eq(d, d);
  expect_ne(d, make_empty());
  expect_ne(d, make_1_labels<float>("a", {Dim::X, 3}, units::m, {1, 2, 3}));
  expect_ne(d, make_1_labels<double>("b", {Dim::X, 3}, units::m, {1, 2, 3}));
  expect_ne(d, make_1_labels<double>("a", {Dim::Y, 3}, units::m, {1, 2, 3}));
  expect_ne(d, make_1_labels<double>("a", {Dim::X, 2}, units::m, {1, 2}));
  expect_ne(d, make_1_labels<double>("a", {Dim::X, 3}, units::s, {1, 2, 3}));
  expect_ne(d, make_1_labels<double>("a", {Dim::X, 3}, units::m, {1, 2, 4}));
}

TEST_F(Dataset_comparison_operators, single_attr) {
  auto d = make_1_attr<double>("a", {Dim::X, 3}, units::m, {1, 2, 3});
  expect_eq(d, d);
  expect_ne(d, make_empty());
  expect_ne(d, make_1_attr<float>("a", {Dim::X, 3}, units::m, {1, 2, 3}));
  expect_ne(d, make_1_attr<double>("b", {Dim::X, 3}, units::m, {1, 2, 3}));
  expect_ne(d, make_1_attr<double>("a", {Dim::Y, 3}, units::m, {1, 2, 3}));
  expect_ne(d, make_1_attr<double>("a", {Dim::X, 2}, units::m, {1, 2}));
  expect_ne(d, make_1_attr<double>("a", {Dim::X, 3}, units::s, {1, 2, 3}));
  expect_ne(d, make_1_attr<double>("a", {Dim::X, 3}, units::m, {1, 2, 4}));
}

TEST_F(Dataset_comparison_operators, single_values) {
  auto d = make_1_values<double>("a", {Dim::X, 3}, units::m, {1, 2, 3});
  expect_eq(d, d);
  expect_ne(d, make_empty());
  expect_ne(d, make_1_values<float>("a", {Dim::X, 3}, units::m, {1, 2, 3}));
  expect_ne(d, make_1_values<double>("b", {Dim::X, 3}, units::m, {1, 2, 3}));
  expect_ne(d, make_1_values<double>("a", {Dim::Y, 3}, units::m, {1, 2, 3}));
  expect_ne(d, make_1_values<double>("a", {Dim::X, 2}, units::m, {1, 2}));
  expect_ne(d, make_1_values<double>("a", {Dim::X, 3}, units::s, {1, 2, 3}));
  expect_ne(d, make_1_values<double>("a", {Dim::X, 3}, units::m, {1, 2, 4}));
}

TEST_F(Dataset_comparison_operators, single_values_and_variances) {
  auto d = make_1_values_and_variances<double>("a", {Dim::X, 3}, units::m,
                                               {1, 2, 3}, {4, 5, 6});
  expect_eq(d, d);
  expect_ne(d, make_empty());
  expect_ne(d, make_1_values_and_variances<float>("a", {Dim::X, 3}, units::m,
                                                  {1, 2, 3}, {4, 5, 6}));
  expect_ne(d, make_1_values_and_variances<double>("b", {Dim::X, 3}, units::m,
                                                   {1, 2, 3}, {4, 5, 6}));
  expect_ne(d, make_1_values_and_variances<double>("a", {Dim::Y, 3}, units::m,
                                                   {1, 2, 3}, {4, 5, 6}));
  expect_ne(d, make_1_values_and_variances<double>("a", {Dim::X, 2}, units::m,
                                                   {1, 2}, {4, 5}));
  expect_ne(d, make_1_values_and_variances<double>("a", {Dim::X, 3}, units::s,
                                                   {1, 2, 3}, {4, 5, 6}));
  expect_ne(d, make_1_values_and_variances<double>("a", {Dim::X, 3}, units::m,
                                                   {1, 2, 4}, {4, 5, 6}));
  expect_ne(d, make_1_values_and_variances<double>("a", {Dim::X, 3}, units::m,
                                                   {1, 2, 3}, {4, 5, 7}));
}
// End baseline checks.

TEST_F(Dataset_comparison_operators, empty) {
  const auto empty = make_empty();
  expect_eq(empty, empty);
}

TEST_F(Dataset_comparison_operators, self) {
  expect_eq(dataset, dataset);
  const auto copy(dataset);
  expect_eq(copy, dataset);
}

TEST_F(Dataset_comparison_operators, extra_coord) {
  auto extra = dataset;
  extra.setCoord(Dim::Z, makeVariable<double>({Dim::Z, 2}));
  expect_ne(extra, dataset);
}

TEST_F(Dataset_comparison_operators, extra_labels) {
  auto extra = dataset;
  extra.setLabels("extra", makeVariable<double>({Dim::Z, 2}));
  expect_ne(extra, dataset);
}

TEST_F(Dataset_comparison_operators, extra_attr) {
  auto extra = dataset;
  extra.setAttr("extra", makeVariable<double>({Dim::Z, 2}));
  expect_ne(extra, dataset);
}

TEST_F(Dataset_comparison_operators, extra_data) {
  auto extra = dataset;
  extra.setValues("extra", makeVariable<double>({Dim::Z, 2}));
  expect_ne(extra, dataset);
}

TEST_F(Dataset_comparison_operators, extra_variance) {
  auto extra = dataset;
  extra.setVariances("val", makeVariable<double>({Dim::X, 4}));
  expect_ne(extra, dataset);
}

TEST_F(Dataset_comparison_operators, extra_sparse_values) {
  auto extra = dataset;
  extra.setValues("sparse_coord", sparse_variable);
  expect_ne(extra, dataset);
}

TEST_F(Dataset_comparison_operators, extra_sparse_label) {
  auto extra = dataset;
  extra.setSparseLabels("sparse_coord_and_val", "extra", sparse_variable);
  expect_ne(extra, dataset);
}

TEST_F(Dataset_comparison_operators, different_coord_insertion_order) {
  auto a = make_empty();
  auto b = make_empty();
  a.setCoord(Dim::X, dataset.coords()[Dim::X]);
  a.setCoord(Dim::Y, dataset.coords()[Dim::Y]);
  b.setCoord(Dim::Y, dataset.coords()[Dim::Y]);
  b.setCoord(Dim::X, dataset.coords()[Dim::X]);
  expect_eq(a, b);
}

TEST_F(Dataset_comparison_operators, different_label_insertion_order) {
  auto a = make_empty();
  auto b = make_empty();
  a.setLabels("x", dataset.coords()[Dim::X]);
  a.setLabels("y", dataset.coords()[Dim::Y]);
  b.setLabels("y", dataset.coords()[Dim::Y]);
  b.setLabels("x", dataset.coords()[Dim::X]);
  expect_eq(a, b);
}

TEST_F(Dataset_comparison_operators, different_attr_insertion_order) {
  auto a = make_empty();
  auto b = make_empty();
  a.setAttr("x", dataset.coords()[Dim::X]);
  a.setAttr("y", dataset.coords()[Dim::Y]);
  b.setAttr("y", dataset.coords()[Dim::Y]);
  b.setAttr("x", dataset.coords()[Dim::X]);
  expect_eq(a, b);
}

TEST_F(Dataset_comparison_operators, different_data_insertion_order) {
  auto a = make_empty();
  auto b = make_empty();
  a.setValues("x", dataset.coords()[Dim::X]);
  a.setValues("y", dataset.coords()[Dim::Y]);
  b.setValues("y", dataset.coords()[Dim::Y]);
  b.setValues("x", dataset.coords()[Dim::X]);
  expect_eq(a, b);
}

class Dataset3DTest : public ::testing::Test {
protected:
  Dataset3DTest() {
    dataset.setCoord(Dim::Time, scalar());
    dataset.setCoord(Dim::X, x());
    dataset.setCoord(Dim::Y, y());
    dataset.setCoord(Dim::Z, xyz());

    dataset.setLabels("labels_x", x());
    dataset.setLabels("labels_xy", xy());
    dataset.setLabels("labels_z", z());

    dataset.setAttr("attr_scalar", scalar());
    dataset.setAttr("attr_x", x());

    dataset.setValues("data_x", x());
    dataset.setVariances("data_x", x());

    dataset.setValues("data_xy", xy());
    dataset.setVariances("data_xy", xy());

    dataset.setValues("data_zyx", zyx());
    dataset.setVariances("data_zyx", zyx());

    dataset.setValues("data_xyz", xyz());

    dataset.setValues("data_scalar", scalar());
  }

  Variable scalar() const { return makeVariable<double>({}, {1000}); }
  Variable x(const scipp::index lx = 4) const {
    std::vector<double> data(lx);
    std::iota(data.begin(), data.end(), 1);
    return makeVariable<double>({Dim::X, lx}, data);
  }
  Variable y(const scipp::index ly = 5) const {
    std::vector<double> data(ly);
    std::iota(data.begin(), data.end(), 5);
    return makeVariable<double>({Dim::Y, ly}, data);
  }
  Variable z() const {
    return makeVariable<double>({Dim::Z, 6}, {10, 11, 12, 13, 14, 15});
  }
  Variable xy() const {
    std::vector<double> data(4 * 5);
    std::iota(data.begin(), data.end(), 16);
    auto var = makeVariable<double>({{Dim::X, 4}, {Dim::Y, 5}}, data);
    return var;
  }
  Variable xyz(const scipp::index lz = 6) const {
    std::vector<double> data(4 * 5 * lz);
    std::iota(data.begin(), data.end(), 4 * 5 + 16);
    auto var =
        makeVariable<double>({{Dim::X, 4}, {Dim::Y, 5}, {Dim::Z, lz}}, data);
    return var;
  }
  Variable zyx() const {
    std::vector<double> data(4 * 5 * 6);
    std::iota(data.begin(), data.end(), 4 * 5 + 4 * 5 * 6 + 16);
    auto var =
        makeVariable<double>({{Dim::Z, 6}, {Dim::Y, 5}, {Dim::X, 4}}, data);
    return var;
  }

  next::Dataset dataset;
};

class Dataset3DTest_slice_x : public Dataset3DTest,
                              public ::testing::WithParamInterface<int> {
protected:
  next::Dataset reference(const scipp::index pos) {
    next::Dataset d;
    d.setCoord(Dim::Time, scalar());
    d.setCoord(Dim::Y, y());
    d.setCoord(Dim::Z, xyz().slice({Dim::X, pos}));
    d.setLabels("labels_xy", xy().slice({Dim::X, pos}));
    d.setLabels("labels_z", z());
    d.setAttr("attr_scalar", scalar());
    d.setValues("data_x", x().slice({Dim::X, pos}));
    d.setVariances("data_x", x().slice({Dim::X, pos}));
    d.setValues("data_xy", xy().slice({Dim::X, pos}));
    d.setVariances("data_xy", xy().slice({Dim::X, pos}));
    d.setValues("data_zyx", zyx().slice({Dim::X, pos}));
    d.setVariances("data_zyx", zyx().slice({Dim::X, pos}));
    d.setValues("data_xyz", xyz().slice({Dim::X, pos}));
    return d;
  }
};
class Dataset3DTest_slice_y : public Dataset3DTest,
                              public ::testing::WithParamInterface<int> {};
class Dataset3DTest_slice_z : public Dataset3DTest,
                              public ::testing::WithParamInterface<int> {};

class Dataset3DTest_slice_range_x : public Dataset3DTest,
                                    public ::testing::WithParamInterface<
                                        std::pair<scipp::index, scipp::index>> {
};
class Dataset3DTest_slice_range_y : public Dataset3DTest,
                                    public ::testing::WithParamInterface<
                                        std::pair<scipp::index, scipp::index>> {
protected:
  next::Dataset reference(const scipp::index begin, const scipp::index end) {
    next::Dataset d;
    d.setCoord(Dim::Time, scalar());
    d.setCoord(Dim::X, x());
    d.setCoord(Dim::Y, y().slice({Dim::Y, begin, end}));
    d.setCoord(Dim::Z, xyz().slice({Dim::Y, begin, end}));
    d.setLabels("labels_x", x());
    d.setLabels("labels_xy", xy().slice({Dim::Y, begin, end}));
    d.setLabels("labels_z", z());
    d.setAttr("attr_scalar", scalar());
    d.setAttr("attr_x", x());
    d.setValues("data_xy", xy().slice({Dim::Y, begin, end}));
    d.setVariances("data_xy", xy().slice({Dim::Y, begin, end}));
    d.setValues("data_zyx", zyx().slice({Dim::Y, begin, end}));
    d.setVariances("data_zyx", zyx().slice({Dim::Y, begin, end}));
    d.setValues("data_xyz", xyz().slice({Dim::Y, begin, end}));
    return d;
  }
};
class Dataset3DTest_slice_range_z : public Dataset3DTest,
                                    public ::testing::WithParamInterface<
                                        std::pair<scipp::index, scipp::index>> {
protected:
  next::Dataset reference(const scipp::index begin, const scipp::index end) {
    next::Dataset d;
    d.setCoord(Dim::Time, scalar());
    d.setCoord(Dim::X, x());
    d.setCoord(Dim::Y, y());
    d.setCoord(Dim::Z, xyz().slice({Dim::Z, begin, end}));
    d.setLabels("labels_x", x());
    d.setLabels("labels_xy", xy());
    d.setLabels("labels_z", z().slice({Dim::Z, begin, end}));
    d.setAttr("attr_scalar", scalar());
    d.setAttr("attr_x", x());
    d.setValues("data_zyx", zyx().slice({Dim::Z, begin, end}));
    d.setVariances("data_zyx", zyx().slice({Dim::Z, begin, end}));
    d.setValues("data_xyz", xyz().slice({Dim::Z, begin, end}));
    return d;
  }
};

template <int max> constexpr auto nonnegative_cartesian_products() {
  using scipp::index;
  const auto size = max + 1;
  std::array<std::pair<index, index>, (size * size + size) / 2> pairs;
  index i = 0;
  for (index first = 0; first <= max; ++first)
    for (index second = first + 0; second <= max; ++second) {
      pairs[i].first = first;
      pairs[i].second = second;
      ++i;
    }
  return pairs;
}

constexpr auto ranges_x = nonnegative_cartesian_products<4>();
constexpr auto ranges_y = nonnegative_cartesian_products<5>();
constexpr auto ranges_z = nonnegative_cartesian_products<6>();

INSTANTIATE_TEST_CASE_P(AllPositions, Dataset3DTest_slice_x,
                        ::testing::Range(0, 4));
INSTANTIATE_TEST_CASE_P(AllPositions, Dataset3DTest_slice_y,
                        ::testing::Range(0, 5));
INSTANTIATE_TEST_CASE_P(AllPositions, Dataset3DTest_slice_z,
                        ::testing::Range(0, 6));

INSTANTIATE_TEST_CASE_P(NonEmptyRanges, Dataset3DTest_slice_range_x,
                        ::testing::ValuesIn(ranges_x));
INSTANTIATE_TEST_CASE_P(NonEmptyRanges, Dataset3DTest_slice_range_y,
                        ::testing::ValuesIn(ranges_y));
INSTANTIATE_TEST_CASE_P(NonEmptyRanges, Dataset3DTest_slice_range_z,
                        ::testing::ValuesIn(ranges_z));

TEST_P(Dataset3DTest_slice_x, slice) {
  const auto pos = GetParam();
  EXPECT_EQ(dataset.slice({Dim::X, pos}), reference(pos));
}

TEST_P(Dataset3DTest_slice_x, slice_bin_edges) {
  const auto pos = GetParam();
  auto datasetWithEdges = dataset;
  datasetWithEdges.setCoord(Dim::X, x(5));
  EXPECT_EQ(datasetWithEdges.slice({Dim::X, pos}), reference(pos));
  EXPECT_EQ(datasetWithEdges.slice({Dim::X, pos}),
            dataset.slice({Dim::X, pos}));
}

TEST_P(Dataset3DTest_slice_y, slice) {
  const auto pos = GetParam();
  next::Dataset reference;
  reference.setCoord(Dim::Time, scalar());
  reference.setCoord(Dim::X, x());
  reference.setCoord(Dim::Z, xyz().slice({Dim::Y, pos}));
  reference.setLabels("labels_x", x());
  reference.setLabels("labels_z", z());
  reference.setAttr("attr_scalar", scalar());
  reference.setAttr("attr_x", x());
  reference.setValues("data_xy", xy().slice({Dim::Y, pos}));
  reference.setVariances("data_xy", xy().slice({Dim::Y, pos}));
  reference.setValues("data_zyx", zyx().slice({Dim::Y, pos}));
  reference.setVariances("data_zyx", zyx().slice({Dim::Y, pos}));
  reference.setValues("data_xyz", xyz().slice({Dim::Y, pos}));

  EXPECT_EQ(dataset.slice({Dim::Y, pos}), reference);
}

TEST_P(Dataset3DTest_slice_z, slice) {
  const auto pos = GetParam();
  next::Dataset reference;
  reference.setCoord(Dim::Time, scalar());
  reference.setCoord(Dim::X, x());
  reference.setCoord(Dim::Y, y());
  reference.setLabels("labels_x", x());
  reference.setLabels("labels_xy", xy());
  reference.setAttr("attr_scalar", scalar());
  reference.setAttr("attr_x", x());
  reference.setValues("data_zyx", zyx().slice({Dim::Z, pos}));
  reference.setVariances("data_zyx", zyx().slice({Dim::Z, pos}));
  reference.setValues("data_xyz", xyz().slice({Dim::Z, pos}));

  EXPECT_EQ(dataset.slice({Dim::Z, pos}), reference);
}

TEST_P(Dataset3DTest_slice_range_x, slice) {
  const auto[begin, end] = GetParam();
  next::Dataset reference;
  reference.setCoord(Dim::Time, scalar());
  reference.setCoord(Dim::X, x().slice({Dim::X, begin, end}));
  reference.setCoord(Dim::Y, y());
  reference.setCoord(Dim::Z, xyz().slice({Dim::X, begin, end}));
  reference.setLabels("labels_x", x().slice({Dim::X, begin, end}));
  reference.setLabels("labels_xy", xy().slice({Dim::X, begin, end}));
  reference.setLabels("labels_z", z());
  reference.setAttr("attr_scalar", scalar());
  reference.setAttr("attr_x", x().slice({Dim::X, begin, end}));
  reference.setValues("data_x", x().slice({Dim::X, begin, end}));
  reference.setVariances("data_x", x().slice({Dim::X, begin, end}));
  reference.setValues("data_xy", xy().slice({Dim::X, begin, end}));
  reference.setVariances("data_xy", xy().slice({Dim::X, begin, end}));
  reference.setValues("data_zyx", zyx().slice({Dim::X, begin, end}));
  reference.setVariances("data_zyx", zyx().slice({Dim::X, begin, end}));
  reference.setValues("data_xyz", xyz().slice({Dim::X, begin, end}));

  EXPECT_EQ(dataset.slice({Dim::X, begin, end}), reference);
}

TEST_P(Dataset3DTest_slice_range_y, slice) {
  const auto[begin, end] = GetParam();
  EXPECT_EQ(dataset.slice({Dim::Y, begin, end}), reference(begin, end));
}

TEST_P(Dataset3DTest_slice_range_y, slice_with_edges) {
  const auto[begin, end] = GetParam();
  auto datasetWithEdges = dataset;
  datasetWithEdges.setCoord(Dim::Y, y(6));
  auto referenceWithEdges = reference(begin, end);
  referenceWithEdges.setCoord(Dim::Y, y(6).slice({Dim::Y, begin, end + 1}));
  EXPECT_EQ(datasetWithEdges.slice({Dim::Y, begin, end}), referenceWithEdges);
}

TEST_P(Dataset3DTest_slice_range_y, slice_with_z_edges) {
  const auto[begin, end] = GetParam();
  auto datasetWithEdges = dataset;
  datasetWithEdges.setCoord(Dim::Z, xyz(7));
  auto referenceWithEdges = reference(begin, end);
  referenceWithEdges.setCoord(Dim::Z, xyz(7).slice({Dim::Y, begin, end}));
  EXPECT_EQ(datasetWithEdges.slice({Dim::Y, begin, end}), referenceWithEdges);
}

TEST_P(Dataset3DTest_slice_range_z, slice) {
  const auto[begin, end] = GetParam();
  EXPECT_EQ(dataset.slice({Dim::Z, begin, end}), reference(begin, end));
}

TEST_P(Dataset3DTest_slice_range_z, slice_with_edges) {
  const auto[begin, end] = GetParam();
  auto datasetWithEdges = dataset;
  datasetWithEdges.setCoord(Dim::Z, xyz(7));
  auto referenceWithEdges = reference(begin, end);
  referenceWithEdges.setCoord(Dim::Z, xyz(7).slice({Dim::Z, begin, end + 1}));
  EXPECT_EQ(datasetWithEdges.slice({Dim::Z, begin, end}), referenceWithEdges);
}

TEST_F(Dataset3DTest, nested_slice) {
  for (const auto dim : {Dim::X, Dim::Y, Dim::Z}) {
    EXPECT_EQ(dataset.slice({dim, 1, 3}, {dim, 1}), dataset.slice({dim, 2}));
  }
}

TEST_F(Dataset3DTest, nested_slice_range) {
  for (const auto dim : {Dim::X, Dim::Y, Dim::Z}) {
    EXPECT_EQ(dataset.slice({dim, 1, 3}, {dim, 0, 2}),
              dataset.slice({dim, 1, 3}));
    EXPECT_EQ(dataset.slice({dim, 1, 3}, {dim, 1, 2}),
              dataset.slice({dim, 2, 3}));
  }
}

TEST_F(Dataset3DTest, commutative_slice) {
  EXPECT_EQ(dataset.slice({Dim::X, 1, 3}, {Dim::Y, 2}),
            dataset.slice({Dim::Y, 2}, {Dim::X, 1, 3}));
  EXPECT_EQ(dataset.slice({Dim::X, 1, 3}, {Dim::Y, 2}, {Dim::Z, 3, 4}),
            dataset.slice({Dim::Y, 2}, {Dim::Z, 3, 4}, {Dim::X, 1, 3}));
  EXPECT_EQ(dataset.slice({Dim::X, 1, 3}, {Dim::Y, 2}, {Dim::Z, 3, 4}),
            dataset.slice({Dim::Z, 3, 4}, {Dim::Y, 2}, {Dim::X, 1, 3}));
  EXPECT_EQ(dataset.slice({Dim::X, 1, 3}, {Dim::Y, 2}, {Dim::Z, 3, 4}),
            dataset.slice({Dim::Z, 3, 4}, {Dim::X, 1, 3}, {Dim::Y, 2}));
}

TEST_F(Dataset3DTest, commutative_slice_range) {
  EXPECT_EQ(dataset.slice({Dim::X, 1, 3}, {Dim::Y, 2, 4}),
            dataset.slice({Dim::Y, 2, 4}, {Dim::X, 1, 3}));
  EXPECT_EQ(dataset.slice({Dim::X, 1, 3}, {Dim::Y, 2, 4}, {Dim::Z, 3, 4}),
            dataset.slice({Dim::Y, 2, 4}, {Dim::Z, 3, 4}, {Dim::X, 1, 3}));
  EXPECT_EQ(dataset.slice({Dim::X, 1, 3}, {Dim::Y, 2, 4}, {Dim::Z, 3, 4}),
            dataset.slice({Dim::Z, 3, 4}, {Dim::Y, 2, 4}, {Dim::X, 1, 3}));
  EXPECT_EQ(dataset.slice({Dim::X, 1, 3}, {Dim::Y, 2, 4}, {Dim::Z, 3, 4}),
            dataset.slice({Dim::Z, 3, 4}, {Dim::X, 1, 3}, {Dim::Y, 2, 4}));
}

template <typename T> class CoordsProxyTest : public ::testing::Test {
protected:
  std::conditional_t<std::is_same_v<T, next::CoordsProxy>, next::Dataset,
                     const next::Dataset> &
  access(auto &dataset) {
    return dataset;
  }
};

using CoordsProxyTypes =
    ::testing::Types<next::CoordsProxy, next::CoordsConstProxy>;
TYPED_TEST_CASE(CoordsProxyTest, CoordsProxyTypes);

TYPED_TEST(CoordsProxyTest, empty) {
  next::Dataset d;
  const auto coords = TestFixture::access(d).coords();
  ASSERT_TRUE(coords.empty());
  ASSERT_EQ(coords.size(), 0);
}

TYPED_TEST(CoordsProxyTest, bad_item_access) {
  next::Dataset d;
  const auto coords = TestFixture::access(d).coords();
  ASSERT_ANY_THROW(coords[Dim::X]);
}

TYPED_TEST(CoordsProxyTest, item_access) {
  next::Dataset d;
  const auto x = makeVariable<double>({Dim::X, 3}, {1, 2, 3});
  const auto y = makeVariable<double>({Dim::Y, 2}, {4, 5});
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y);

  const auto coords = TestFixture::access(d).coords();
  ASSERT_EQ(coords[Dim::X], x);
  ASSERT_EQ(coords[Dim::Y], y);
}

TYPED_TEST(CoordsProxyTest, iterators_empty_coords) {
  next::Dataset d;
  const auto coords = TestFixture::access(d).coords();

  ASSERT_NO_THROW(coords.begin());
  ASSERT_NO_THROW(coords.end());
  EXPECT_EQ(coords.begin(), coords.end());
}

TYPED_TEST(CoordsProxyTest, iterators) {
  next::Dataset d;
  const auto x = makeVariable<double>({Dim::X, 3}, {1, 2, 3});
  const auto y = makeVariable<double>({Dim::Y, 2}, {4, 5});
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y);
  const auto coords = TestFixture::access(d).coords();

  ASSERT_NO_THROW(coords.begin());
  ASSERT_NO_THROW(coords.end());

  auto it = coords.begin();
  ASSERT_NE(it, coords.end());
  EXPECT_EQ(it->first, Dim::X);
  EXPECT_EQ(it->second, x);

  ASSERT_NO_THROW(++it);
  ASSERT_NE(it, coords.end());
  EXPECT_EQ(it->first, Dim::Y);
  EXPECT_EQ(it->second, y);

  ASSERT_NO_THROW(++it);
  ASSERT_EQ(it, coords.end());
}

TYPED_TEST(CoordsProxyTest, slice) {
  next::Dataset d;
  const auto x = makeVariable<double>({Dim::X, 3}, {1, 2, 3});
  const auto y = makeVariable<double>({Dim::Y, 2}, {1, 2});
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y);
  const auto coords = TestFixture::access(d).coords();

  const auto sliceX = coords.slice({Dim::X, 1});
  EXPECT_ANY_THROW(sliceX[Dim::X]);
  EXPECT_EQ(sliceX[Dim::Y], y);

  const auto sliceDX = coords.slice({Dim::X, 1, 2});
  EXPECT_EQ(sliceDX[Dim::X], x.slice({Dim::X, 1, 2}));
  EXPECT_EQ(sliceDX[Dim::Y], y);

  const auto sliceY = coords.slice({Dim::Y, 1});
  EXPECT_EQ(sliceY[Dim::X], x);
  EXPECT_ANY_THROW(sliceY[Dim::Y]);

  const auto sliceDY = coords.slice({Dim::Y, 1, 2});
  EXPECT_EQ(sliceDY[Dim::X], x);
  EXPECT_EQ(sliceDY[Dim::Y], y.slice({Dim::Y, 1, 2}));
}

auto make_dataset_2d_coord_x_1d_coord_y() {
  next::Dataset d;
  const auto x =
      makeVariable<double>({{Dim::X, 3}, {Dim::Y, 2}}, {1, 2, 3, 4, 5, 6});
  const auto y = makeVariable<double>({Dim::Y, 2}, {1, 2});
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y);
  return d;
}

TYPED_TEST(CoordsProxyTest, slice_2D_coord) {
  auto d = make_dataset_2d_coord_x_1d_coord_y();
  const auto coords = TestFixture::access(d).coords();

  const auto sliceX = coords.slice({Dim::X, 1});
  EXPECT_ANY_THROW(sliceX[Dim::X]);
  EXPECT_EQ(sliceX[Dim::Y], coords[Dim::Y]);

  const auto sliceDX = coords.slice({Dim::X, 1, 2});
  EXPECT_EQ(sliceDX[Dim::X], coords[Dim::X].slice({Dim::X, 1, 2}));
  EXPECT_EQ(sliceDX[Dim::Y], coords[Dim::Y]);

  const auto sliceY = coords.slice({Dim::Y, 1});
  EXPECT_EQ(sliceY[Dim::X], coords[Dim::X].slice({Dim::Y, 1}));
  EXPECT_ANY_THROW(sliceY[Dim::Y]);

  const auto sliceDY = coords.slice({Dim::Y, 1, 2});
  EXPECT_EQ(sliceDY[Dim::X], coords[Dim::X].slice({Dim::Y, 1, 2}));
  EXPECT_EQ(sliceDY[Dim::Y], coords[Dim::Y].slice({Dim::Y, 1, 2}));
}

auto check_slice_of_slice = [](const auto &dataset, const auto slice) {
  EXPECT_EQ(slice[Dim::X],
            dataset.coords()[Dim::X].slice({Dim::X, 1, 3}).slice({Dim::Y, 1}));
  EXPECT_ANY_THROW(slice[Dim::Y]);
};

TYPED_TEST(CoordsProxyTest, slice_of_slice) {
  auto d = make_dataset_2d_coord_x_1d_coord_y();
  const auto cs = TestFixture::access(d).coords();

  check_slice_of_slice(d, cs.slice({Dim::X, 1, 3}).slice({Dim::Y, 1}));
  check_slice_of_slice(d, cs.slice({Dim::Y, 1}).slice({Dim::X, 1, 3}));
  check_slice_of_slice(d, cs.slice({Dim::X, 1, 3}, {Dim::Y, 1}));
  check_slice_of_slice(d, cs.slice({Dim::Y, 1}, {Dim::X, 1, 3}));
}

auto check_slice_of_slice_range = [](const auto &dataset, const auto slice) {
  EXPECT_EQ(
      slice[Dim::X],
      dataset.coords()[Dim::X].slice({Dim::X, 1, 3}).slice({Dim::Y, 1, 2}));
  EXPECT_EQ(slice[Dim::Y], dataset.coords()[Dim::Y].slice({Dim::Y, 1, 2}));
};

TYPED_TEST(CoordsProxyTest, slice_of_slice_range) {
  auto d = make_dataset_2d_coord_x_1d_coord_y();
  const auto cs = TestFixture::access(d).coords();

  check_slice_of_slice_range(d, cs.slice({Dim::X, 1, 3}).slice({Dim::Y, 1, 2}));
  check_slice_of_slice_range(d, cs.slice({Dim::Y, 1, 2}).slice({Dim::X, 1, 3}));
  check_slice_of_slice_range(d, cs.slice({Dim::X, 1, 3}, {Dim::Y, 1, 2}));
  check_slice_of_slice_range(d, cs.slice({Dim::Y, 1, 2}, {Dim::X, 1, 3}));
}

TEST(CoordsConstProxy, slice_return_type) {
  const next::Dataset d;
  ASSERT_TRUE((std::is_same_v<decltype(d.coords().slice({Dim::X, 0})),
                              CoordsConstProxy>));
}

TEST(CoordsProxy, slice_return_type) {
  next::Dataset d;
  ASSERT_TRUE(
      (std::is_same_v<decltype(d.coords().slice({Dim::X, 0})), CoordsProxy>));
}

TEST(MutableCoordsProxyTest, item_write) {
  next::Dataset d;
  const auto x = makeVariable<double>({Dim::X, 3}, {1, 2, 3});
  const auto y = makeVariable<double>({Dim::Y, 2}, {4, 5});
  const auto x_reference = makeVariable<double>({Dim::X, 3}, {1.5, 2.0, 3.0});
  const auto y_reference = makeVariable<double>({Dim::Y, 2}, {4.5, 5.0});
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y);

  const auto coords = d.coords();
  coords[Dim::X].values<double>()[0] += 0.5;
  coords[Dim::Y].values<double>()[0] += 0.5;
  ASSERT_EQ(coords[Dim::X], x_reference);
  ASSERT_EQ(coords[Dim::Y], y_reference);
}

TEST(CoordsProxy, modify_slice) {
  auto d = make_dataset_2d_coord_x_1d_coord_y();
  const auto coords = d.coords();

  const auto slice = coords.slice({Dim::X, 1, 2});
  for (auto &x : slice[Dim::X].values<double>())
    x = 0.0;

  const auto reference =
      makeVariable<double>({{Dim::X, 3}, {Dim::Y, 2}}, {1, 2, 0, 0, 5, 6});
  EXPECT_EQ(d.coords()[Dim::X], reference);
}

TEST(CoordsConstProxy, slice_bin_edges_with_2D_coord) {
  next::Dataset d;
  const auto x = makeVariable<double>({{Dim::Y, 2}, {Dim::X, 2}}, {1, 2, 3, 4});
  const auto y_edges = makeVariable<double>({Dim::Y, 3}, {1, 2, 3});
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y_edges);
  const auto coords = d.coords();

  const auto sliceX = coords.slice({Dim::X, 1});
  EXPECT_ANY_THROW(sliceX[Dim::X]);
  EXPECT_EQ(sliceX[Dim::Y], coords[Dim::Y]);

  const auto sliceDX = coords.slice({Dim::X, 1, 2});
  EXPECT_EQ(sliceDX[Dim::X].dims(), Dimensions({{Dim::Y, 2}, {Dim::X, 1}}));
  EXPECT_EQ(sliceDX[Dim::Y], coords[Dim::Y]);

  const auto sliceY = coords.slice({Dim::Y, 1});
  // TODO Would it be more consistent to preserve X with 0 thickness?
  EXPECT_ANY_THROW(sliceY[Dim::X]);
  EXPECT_ANY_THROW(sliceY[Dim::Y]);

  const auto sliceY_edge = coords.slice({Dim::Y, 1, 2});
  EXPECT_EQ(sliceY_edge[Dim::X], coords[Dim::X].slice({Dim::Y, 1, 1}));
  EXPECT_EQ(sliceY_edge[Dim::Y], coords[Dim::Y].slice({Dim::Y, 1, 2}));

  const auto sliceY_bin = coords.slice({Dim::Y, 1, 3});
  EXPECT_EQ(sliceY_bin[Dim::X], coords[Dim::X].slice({Dim::Y, 1, 2}));
  EXPECT_EQ(sliceY_bin[Dim::Y], coords[Dim::Y].slice({Dim::Y, 1, 3}));
}

// Using typed tests for common functionality of DataProxy and DataConstProxy.
template <typename T> class DataProxyTest : public ::testing::Test {
protected:
  using dataset_type = std::conditional_t<std::is_same_v<T, next::DataProxy>,
                                          next::Dataset, const next::Dataset>;
};

using DataProxyTypes = ::testing::Types<next::DataProxy, next::DataConstProxy>;
TYPED_TEST_CASE(DataProxyTest, DataProxyTypes);

TYPED_TEST(DataProxyTest, isSparse_sparseDim) {
  next::Dataset d;
  typename TestFixture::dataset_type &d_ref(d);

  d.setValues("dense", makeVariable<double>({}));
  ASSERT_FALSE(d_ref["dense"].isSparse());
  ASSERT_EQ(d_ref["dense"].sparseDim(), Dim::Invalid);

  d.setValues("sparse_data", makeSparseVariable<double>({}, Dim::X));
  ASSERT_TRUE(d_ref["sparse_data"].isSparse());
  ASSERT_EQ(d_ref["sparse_data"].sparseDim(), Dim::X);

  d.setSparseCoord("sparse_coord", makeSparseVariable<double>({}, Dim::X));
  ASSERT_TRUE(d_ref["sparse_coord"].isSparse());
  ASSERT_EQ(d_ref["sparse_coord"].sparseDim(), Dim::X);
}

TYPED_TEST(DataProxyTest, dims) {
  next::Dataset d;
  const auto dense = makeVariable<double>({{Dim::X, 1}, {Dim::Y, 2}});
  const auto sparse =
      makeSparseVariable<double>({{Dim::X, 1}, {Dim::Y, 2}}, Dim::Z);
  typename TestFixture::dataset_type &d_ref(d);

  d.setValues("dense", dense);
  ASSERT_EQ(d_ref["dense"].dims(), dense.dims());

  // Sparse dimension is currently not included in dims(). It is unclear whether
  // this is the right choice. An unfinished idea involves returning
  // std::tuple<std::span<const Dim>, std::optional<Dim>> instead, using `auto [
  // dims, sparse ] = data.dims();`.
  d.setValues("sparse_data", sparse);
  ASSERT_EQ(d_ref["sparse_data"].dims(), dense.dims());
  ASSERT_EQ(d_ref["sparse_data"].dims(), sparse.dims());

  d.setSparseCoord("sparse_coord", sparse);
  ASSERT_EQ(d_ref["sparse_coord"].dims(), dense.dims());
  ASSERT_EQ(d_ref["sparse_coord"].dims(), sparse.dims());
}

TYPED_TEST(DataProxyTest, dims_with_extra_coords) {
  next::Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  const auto x = makeVariable<double>({Dim::X, 3}, {1, 2, 3});
  const auto y = makeVariable<double>({Dim::Y, 3}, {4, 5, 6});
  const auto var = makeVariable<double>({Dim::X, 3});
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y);
  d.setValues("a", var);

  ASSERT_EQ(d_ref["a"].dims(), var.dims());
}

TYPED_TEST(DataProxyTest, unit) {
  next::Dataset d;
  typename TestFixture::dataset_type &d_ref(d);

  d.setValues("dense", makeVariable<double>({}));
  EXPECT_EQ(d_ref["dense"].unit(), units::dimensionless);
}

TYPED_TEST(DataProxyTest, unit_access_fails_without_values) {
  next::Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  d.setSparseCoord("sparse", makeSparseVariable<double>({}, Dim::X));
  EXPECT_ANY_THROW(d_ref["sparse"].unit());
}

TYPED_TEST(DataProxyTest, coords) {
  next::Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  const auto var = makeVariable<double>({Dim::X, 3});
  d.setCoord(Dim::X, var);
  d.setValues("a", var);

  ASSERT_NO_THROW(d_ref["a"].coords());
  ASSERT_EQ(d_ref["a"].coords(), d.coords());
}

TYPED_TEST(DataProxyTest, coords_sparse) {
  next::Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  const auto var = makeSparseVariable<double>({Dim::X, 3}, Dim::Y);
  d.setSparseCoord("a", var);

  ASSERT_NO_THROW(d_ref["a"].coords());
  ASSERT_NE(d_ref["a"].coords(), d.coords());
  ASSERT_EQ(d_ref["a"].coords().size(), 1);
  ASSERT_NO_THROW(d_ref["a"].coords()[Dim::Y]);
  ASSERT_EQ(d_ref["a"].coords()[Dim::Y], var);
}

TYPED_TEST(DataProxyTest, coords_sparse_shadow) {
  next::Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  const auto x = makeVariable<double>({Dim::X, 3}, {1, 2, 3});
  const auto y = makeVariable<double>({Dim::Y, 3}, {4, 5, 6});
  const auto sparse = makeSparseVariable<double>({Dim::X, 3}, Dim::Y);
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y);
  d.setSparseCoord("a", sparse);

  ASSERT_NO_THROW(d_ref["a"].coords());
  ASSERT_NE(d_ref["a"].coords(), d.coords());
  ASSERT_EQ(d_ref["a"].coords().size(), 2);
  ASSERT_NO_THROW(d_ref["a"].coords()[Dim::X]);
  ASSERT_NO_THROW(d_ref["a"].coords()[Dim::Y]);
  ASSERT_EQ(d_ref["a"].coords()[Dim::X], x);
  ASSERT_NE(d_ref["a"].coords()[Dim::Y], y);
  ASSERT_EQ(d_ref["a"].coords()[Dim::Y], sparse);
}

TYPED_TEST(DataProxyTest, coords_sparse_shadow_even_if_no_coord) {
  next::Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  const auto x = makeVariable<double>({Dim::X, 3}, {1, 2, 3});
  const auto y = makeVariable<double>({Dim::Y, 3}, {4, 5, 6});
  const auto sparse = makeSparseVariable<double>({Dim::X, 3}, Dim::Y);
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y);
  d.setValues("a", sparse);

  ASSERT_NO_THROW(d_ref["a"].coords());
  // Dim::Y is sparse, so the global (non-sparse) Y coordinate does not make
  // sense and is thus hidden.
  ASSERT_NE(d_ref["a"].coords(), d.coords());
  ASSERT_EQ(d_ref["a"].coords().size(), 1);
  ASSERT_NO_THROW(d_ref["a"].coords()[Dim::X]);
  ASSERT_ANY_THROW(d_ref["a"].coords()[Dim::Y]);
  ASSERT_EQ(d_ref["a"].coords()[Dim::X], x);
}

TYPED_TEST(DataProxyTest, coords_contains_only_relevant) {
  next::Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  const auto x = makeVariable<double>({Dim::X, 3}, {1, 2, 3});
  const auto y = makeVariable<double>({Dim::Y, 3}, {4, 5, 6});
  const auto var = makeVariable<double>({Dim::X, 3});
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y);
  d.setValues("a", var);
  const auto coords = d_ref["a"].coords();

  ASSERT_NE(coords, d.coords());
  ASSERT_EQ(coords.size(), 1);
  ASSERT_NO_THROW(coords[Dim::X]);
  ASSERT_EQ(coords[Dim::X], x);
}

TYPED_TEST(DataProxyTest, coords_contains_only_relevant_2d_dropped) {
  next::Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  const auto x = makeVariable<double>({Dim::X, 3}, {1, 2, 3});
  const auto y = makeVariable<double>({{Dim::Y, 3}, {Dim::X, 3}});
  const auto var = makeVariable<double>({Dim::X, 3});
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y);
  d.setValues("a", var);
  const auto coords = d_ref["a"].coords();

  ASSERT_NE(coords, d.coords());
  ASSERT_EQ(coords.size(), 1);
  ASSERT_NO_THROW(coords[Dim::X]);
  ASSERT_EQ(coords[Dim::X], x);
}

TYPED_TEST(DataProxyTest,
           coords_contains_only_relevant_2d_not_dropped_inconsistency) {
  next::Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  const auto x = makeVariable<double>({{Dim::Y, 3}, {Dim::X, 3}});
  const auto y = makeVariable<double>({Dim::Y, 3});
  const auto var = makeVariable<double>({Dim::X, 3});
  d.setCoord(Dim::X, x);
  d.setCoord(Dim::Y, y);
  d.setValues("a", var);
  const auto coords = d_ref["a"].coords();

  // This is a very special case which is probably unlikely to occur in
  // practice. If the coordinate depends on extra dimensions and the data is
  // not, it implies that the coordinate cannot be for this data item, so it
  // should be dropped... HOWEVER, the current implementation DOES NOT DROP IT.
  // Should that be changed?
  ASSERT_NE(coords, d.coords());
  ASSERT_EQ(coords.size(), 1);
  ASSERT_NO_THROW(coords[Dim::X]);
  ASSERT_EQ(coords[Dim::X], x);
}

TYPED_TEST(DataProxyTest, hasValues_hasVariances) {
  next::Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  const auto var = makeVariable<double>({});

  d.setValues("a", var);
  d.setValues("b", var);
  d.setVariances("b", var);

  ASSERT_TRUE(d_ref["a"].hasValues());
  ASSERT_FALSE(d_ref["a"].hasVariances());

  ASSERT_TRUE(d_ref["b"].hasValues());
  ASSERT_TRUE(d_ref["b"].hasVariances());
}

TYPED_TEST(DataProxyTest, values_variances) {
  next::Dataset d;
  typename TestFixture::dataset_type &d_ref(d);
  const auto var = makeVariable<double>({Dim::X, 2}, {1, 2});
  d.setValues("a", var);
  d.setVariances("a", var);

  ASSERT_EQ(d_ref["a"].values(), var);
  ASSERT_EQ(d_ref["a"].variances(), var);
  ASSERT_TRUE(equals(d_ref["a"].template values<double>(), {1, 2}));
  ASSERT_TRUE(equals(d_ref["a"].template variances<double>(), {1, 2}));
  ASSERT_ANY_THROW(d_ref["a"].template values<float>());
  ASSERT_ANY_THROW(d_ref["a"].template variances<float>());
}

template <typename T> class DataProxy3DTest : public Dataset3DTest {
protected:
  using dataset_type = std::conditional_t<std::is_same_v<T, next::DataProxy>,
                                          next::Dataset, const next::Dataset>;

  dataset_type &dataset() { return Dataset3DTest::dataset; }
};

TYPED_TEST_CASE(DataProxy3DTest, DataProxyTypes);

// We have tests that ensure that Dataset::slice is correct (and its item access
// returns the correct data), so we rely on that for verifying the results of
// slicing DataProxy.
TYPED_TEST(DataProxy3DTest, slice_single) {
  auto &d = TestFixture::dataset();
  for (const auto[name, item] : d) {
    for (const auto dim : {Dim::X, Dim::Y, Dim::Z}) {
      if (item.dims().contains(dim)) {
        EXPECT_ANY_THROW(item.slice({dim, -1}));
        for (scipp::index i = 0; i < item.dims()[dim]; ++i)
          EXPECT_EQ(item.slice({dim, i}), d.slice({dim, i})[name]);
        EXPECT_ANY_THROW(item.slice({dim, item.dims()[dim]}));
      } else {
        EXPECT_ANY_THROW(item.slice({dim, 0}));
      }
    }
  }
}

TYPED_TEST(DataProxy3DTest, slice_length_0) {
  auto &d = TestFixture::dataset();
  for (const auto[name, item] : d) {
    for (const auto dim : {Dim::X, Dim::Y, Dim::Z}) {
      if (item.dims().contains(dim)) {
        EXPECT_ANY_THROW(item.slice({dim, -1, -1}));
        for (scipp::index i = 0; i < item.dims()[dim]; ++i)
          EXPECT_EQ(item.slice({dim, i, i + 0}),
                    d.slice({dim, i, i + 0})[name]);
        EXPECT_ANY_THROW(
            item.slice({dim, item.dims()[dim], item.dims()[dim] + 0}));
      } else {
        EXPECT_ANY_THROW(item.slice({dim, 0, 0}));
      }
    }
  }
}

TYPED_TEST(DataProxy3DTest, slice_length_1) {
  auto &d = TestFixture::dataset();
  for (const auto[name, item] : d) {
    for (const auto dim : {Dim::X, Dim::Y, Dim::Z}) {
      if (item.dims().contains(dim)) {
        EXPECT_ANY_THROW(item.slice({dim, -1, 0}));
        for (scipp::index i = 0; i < item.dims()[dim]; ++i)
          EXPECT_EQ(item.slice({dim, i, i + 1}),
                    d.slice({dim, i, i + 1})[name]);
        EXPECT_ANY_THROW(
            item.slice({dim, item.dims()[dim], item.dims()[dim] + 1}));
      } else {
        EXPECT_ANY_THROW(item.slice({dim, 0, 0}));
      }
    }
  }
}

TYPED_TEST(DataProxy3DTest, slice) {
  auto &d = TestFixture::dataset();
  for (const auto[name, item] : d) {
    for (const auto dim : {Dim::X, Dim::Y, Dim::Z}) {
      if (item.dims().contains(dim)) {
        EXPECT_ANY_THROW(item.slice({dim, -1, 1}));
        for (scipp::index i = 0; i < item.dims()[dim] - 1; ++i)
          EXPECT_EQ(item.slice({dim, i, i + 2}),
                    d.slice({dim, i, i + 2})[name]);
        EXPECT_ANY_THROW(
            item.slice({dim, item.dims()[dim], item.dims()[dim] + 2}));
      } else {
        EXPECT_ANY_THROW(item.slice({dim, 0, 2}));
      }
    }
  }
}
