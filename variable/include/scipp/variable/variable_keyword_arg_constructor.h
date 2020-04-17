// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Igor Gudich

#ifndef SCIPP_VARIABLE_KEYWORD_ARG_CONSTRUCTOR_H
#define SCIPP_VARIABLE_KEYWORD_ARG_CONSTRUCTOR_H

#include <limits>
#include <type_traits>

namespace scipp::variable {

// The structs needed for keyword-like variable constructor are introduced
// below. Tags are used to match the corresponding arguments treating the
// arbitrary order of arguments in the constructor, and not mixing values and
// variances. Structures Values and Variances just forwards the arguments for
// constructing internal variable structure - array storage.

namespace detail {
template <int N, typename... Ts>
using nthDecayType =
    typename std::decay_t<std::tuple_element_t<N, std::tuple<Ts...>>>;

template <class... Args> constexpr bool has_last_arg_int64_t() {
  constexpr size_t n = sizeof...(Args);
  if constexpr (n == 0)
    return false;
  else
    return std::is_same_v<nthDecayType<n - 1, Args...>, int64_t>;
}

template <class U> struct vector_like {
  std::vector<U> data;
  template <class... Args>
  vector_like(Args &&... args) : data(make(std::forward<Args>(args)...)) {}

  template <class T>
  vector_like(std::initializer_list<T> init) : data(init.begin(), init.end()) {}

private:
  // This is to override the std::vector(size_t num_elems, const Type& element)
  // insted of [elem, elem, ..., elem] we want [Type(num_elems), element]
  template <class... Args> static std::vector<U> make(Args &&... args) {
    if constexpr (has_last_arg_int64_t<Args...>())
      return std::vector{U(std::forward<Args>(args))...};
    else
      return std::vector<U>(std::forward<Args>(args)...);
  }
};

struct ValuesTag {};

struct VariancesTag {};

template <class... Ts> auto makeArgsTuple(Ts &&... ts) {
  return std::tuple<std::decay_t<Ts>...>(std::forward<Ts>(ts)...);
}

template <class T> auto makeArgsTuple(std::initializer_list<T> init) {
  using iter = typename std::initializer_list<T>::iterator;
  return std::make_tuple<iter, iter>(init.begin(), init.end());
}

} // namespace detail

using Shape = detail::vector_like<scipp::index>;
using Dims = detail::vector_like<Dim>;

template <class... Args>
using ArgsTuple = decltype(detail::makeArgsTuple(std::declval<Args>()...));

template <class... Args> struct Values : detail::ValuesTag {
  ArgsTuple<Args...> tuple;
  Values(Args &&... args)
      : tuple(detail::makeArgsTuple(std::forward<Args>(args)...)) {}
  template <class T>
  Values(std::initializer_list<T> init)
      : tuple(detail::makeArgsTuple(init.begin(), init.end())) {}
};
template <class... Args> Values(Args &&... args)->Values<Args...>;
template <class T>
Values(std::initializer_list<T>)
    ->Values<typename std::initializer_list<T>::iterator,
             typename std::initializer_list<T>::iterator>;

template <class... Args> struct Variances : detail::VariancesTag {
  ArgsTuple<Args...> tuple;
  Variances(Args &&... args)
      : tuple(detail::makeArgsTuple(std::forward<Args>(args)...)) {}
  template <class T>
  Variances(std::initializer_list<T> init)
      : tuple(detail::makeArgsTuple(init.begin(), init.end())) {}
};
template <class... Args> Variances(Args &&... args)->Variances<Args...>;
template <class T>
Variances(std::initializer_list<T>)
    ->Variances<typename std::initializer_list<T>::iterator,
                typename std::initializer_list<T>::iterator>;

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

#endif // SCIPP_VARIABLE_KEYWORD_ARG_CONSTRUCTOR_H
