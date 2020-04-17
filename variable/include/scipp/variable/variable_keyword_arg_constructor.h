// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <limits>
#include <type_traits>

namespace scipp::variable {

// The structs needed for keyword-like variable constructor are introduced
// below. Tags are used to match the corresponding arguments treating the
// arbitrary order of arguments in the constructor, and not mixing values and
// variances. Structures Values and Variances just forwards the arguments for
// constructing internal variable structure - array storage.

namespace detail {

template <class U> struct vector_base {
  std::vector<U> data;
  template <class... Args>
  vector_base(Args &&... args) : data(std::forward<Args>(args)...) {}
  template <class T>
  vector_base(std::initializer_list<T> init) : data(init.begin(), init.end()) {}
};

template <class U> struct vector : public vector_base<U> {
  using vector_base<U>::vector_base;
  template <class T> vector(T &&count, U &&value = U()) = delete;
};

template <template <class...> class Derived, class... Args> struct arg_tuple {
  std::tuple<std::decay_t<Args>...> tuple;
  arg_tuple(Args &&... args) : tuple(std::forward<Args>(args)...) {}
};

} // namespace detail

using Shape = detail::vector<scipp::index>;
using Dims = detail::vector<Dim>;

template <class... Args>
struct Values : public detail::arg_tuple<Values, Args...> {
  using detail::arg_tuple<Values, Args...>::arg_tuple;
  template <class T>
  Values(std::initializer_list<T> init)
      : detail::arg_tuple<Values, Args...>(init) {}
};
template <class... Args>
struct Variances : public detail::arg_tuple<Variances, Args...> {
  using detail::arg_tuple<Variances, Args...>::arg_tuple;
  template <class T>
  Variances(std::initializer_list<T> init)
      : detail::arg_tuple<Variances, Args...>(init) {}
};

template <class... Args> Values(Args &&... args)->Values<Args...>;
template <class T>
Values(std::initializer_list<T>)->Values<std::initializer_list<T>>;

template <class... Args> Variances(Args &&... args)->Variances<Args...>;
template <class T>
Variances(std::initializer_list<T>)->Variances<std::initializer_list<T>>;

namespace detail {

void throw_keyword_arg_constructor_bad_dtype(const DType dtype);

template <class ElemT> struct ArgParser {
  template <class Tuple>
  static void parse(const units::Unit &unit, Tuple &args) {
    std::get<units::Unit>(args) = unit;
  }

  template <class Tuple>
  static void parse(const Dimensions &dims, Tuple &args) {
    std::get<Dimensions>(args) = dims;
  }

  template <class Tuple> static void parse(const Dims &labels, Tuple &args) {
    std::vector<scipp::index> shape(labels.data.size(),
                                    std::numeric_limits<scipp::index>::max());
    std::get<Dimensions>(args) = Dimensions(labels.data, shape);
  }

  template <class Tuple> static void parse(const Shape &shape, Tuple &args) {
    const auto &labels = std::get<Dimensions>(args).labels();
    std::get<Dimensions>(args) =
        Dimensions({labels.begin(), labels.end()}, shape.data);
  }

  template <class Tuple, class... Args>
  static void parse(Values<Args...> &&values, Tuple &args) {
    if constexpr (std::is_constructible_v<element_array<ElemT>, Args...>)
      std::get<2>(args) =
          std::make_from_tuple<element_array<ElemT>>(std::move(values.tuple));
    else
      throw_keyword_arg_constructor_bad_dtype(core::dtype<ElemT>);
  }

  template <class Tuple, class... Args>
  static void parse(Variances<Args...> &&variances, Tuple &args) {
    if constexpr (std::is_constructible_v<element_array<ElemT>, Args...>)
      std::get<3>(args) = std::make_from_tuple<element_array<ElemT>>(
          std::move(variances.tuple));
    else
      throw_keyword_arg_constructor_bad_dtype(core::dtype<ElemT>);
  }
};

} // namespace detail
} // namespace scipp::variable
