#ifndef DATA_ARRAY_H
#define DATA_ARRAY_H

#include <string>
#include <type_traits>

#include <gsl/span>

#include "cow_ptr.h"
#include "dimensions.h"
#include "unit.h"
#include "index.h"
#include "variable.h"

class DataArrayConcept {
public:
  virtual ~DataArrayConcept() = default;
  virtual std::unique_ptr<DataArrayConcept> clone() const = 0;
  virtual gsl::index size() const = 0;
  virtual void resize(const gsl::index) = 0;
  virtual void copyFrom(const gsl::index chunkSize, const gsl::index chunkStart,
                        const gsl::index chunkSkip,
                        const DataArrayConcept &other) = 0;
};

template <class T> class DataArrayModel : public DataArrayConcept {
public:
  DataArrayModel(T const &model) : m_model(model) {}
  std::unique_ptr<DataArrayConcept> clone() const override {
    return std::make_unique<DataArrayModel<T>>(m_model);
  }
  gsl::index size() const override { return m_model.size(); }
  void resize(const gsl::index size) override { m_model.resize(size); }

  void copyFrom(const gsl::index chunkSize, const gsl::index chunkStart,
                const gsl::index chunkSkip,
                const DataArrayConcept &other) override {
    const auto source = dynamic_cast<const DataArrayModel<T> &>(other);
    auto in = source.m_model.begin();
    auto in_end = source.m_model.end();
    auto out = m_model.begin() + chunkStart * chunkSize;
    auto out_end = m_model.end();
    while (in != in_end && out != out_end) {
      for (gsl::index i = 0; i < chunkSize; ++i) {
        *out = *in;
        ++out;
        ++in;
      }
      out += (chunkSkip - 1) * chunkSize;
    }
  }

  T m_model;
};

DataArray concatenate(const Dimension dim, const DataArray &a1,
                      const DataArray &a2);

class DataArray {
public:
  template <class T>
  DataArray(uint32_t id, Dimensions dimensions, T object)
      : m_type(id), m_unit{Unit::Id::Dimensionless},
        m_dimensions(std::move(dimensions)),
        m_object(std::make_unique<DataArrayModel<T>>(std::move(object))) {
    if (m_dimensions.volume() != m_object->size())
      throw std::runtime_error("Creating DataArray: data size does not match "
                               "volume given by dimension extents");
  }

  const std::string &name() const { return m_name; }
  void setName(const std::string &name) {
    if (isCoord())
      throw std::runtime_error("Coordinate variable cannot have a name.");
    m_name = name;
  }

  const Unit &unit() const { return m_unit; }
  void setUnit(const Unit &unit) {
    // TODO
    // Some variables are special, e.g., Data::Tof, which must always have a
    // time-of-flight-related unit. We need some sort of check here. Is there a
    // better mechanism to implement this that does not require gatekeeping here
    // but expresses itself on the interface instead? Does it make sense to
    // handle all unit changes by conversion functions?
    m_unit = unit;
  }

  gsl::index size() const { return m_object->size(); }

  const Dimensions &dimensions() const { return m_dimensions; }
  void setDimensions(const Dimensions &dimensions) {
    m_object.access().resize(dimensions.volume());
    m_dimensions = dimensions;
  }

  const DataArrayConcept &data() const { return *m_object; }
  DataArrayConcept &data() { return m_object.access(); }

  template <class Tag> bool valueTypeIs() const {
    return tag_id<Tag> == m_type;
  }

  uint16_t type() const { return m_type; }
  bool isCoord() const { return m_type < std::tuple_size<Coord::tags>::value; }

  template <class Tag> auto get() const {
    return gsl::make_span(cast<std::remove_const_t<variable_type_t<Tag>>>());
  }

  template <class Tag>
  auto get(std::enable_if_t<std::is_const<Tag>::value> * = nullptr) {
    return const_cast<const DataArray *>(this)->get<Tag>();
  }

  template <class Tag>
  auto get(std::enable_if_t<!std::is_const<Tag>::value> * = nullptr) {
    return gsl::make_span(cast<std::remove_const_t<variable_type_t<Tag>>>());
  }

private:
  template <class T> const T &cast() const {
    return dynamic_cast<const DataArrayModel<T> &>(*m_object).m_model;
  }

  template <class T> T &cast() {
    return dynamic_cast<DataArrayModel<T> &>(m_object.access()).m_model;
  }

  uint16_t m_type;
  std::string m_name;
  Unit m_unit;
  Dimensions m_dimensions;
  cow_ptr<DataArrayConcept> m_object;
};

template <class Tag, class... Args>
DataArray makeDataArray(Dimensions dimensions, Args &&... args) {
  return DataArray(tag_id<Tag>, std::move(dimensions),
                   variable_type_t<Tag>(std::forward<Args>(args)...));
}

template <class Tag, class T>
DataArray makeDataArray(Dimensions dimensions,
                        std::initializer_list<T> values) {
  return DataArray(tag_id<Tag>, std::move(dimensions),
                   variable_type_t<Tag>(values));
}

inline DataArray concatenate(const Dimension dim, const DataArray &a1,
                             const DataArray &a2) {
  if (a1.type() != a2.type())
    throw std::runtime_error(
        "Cannot concatenate DataArrays: Data types do not match.");
  if (a1.unit() != a2.unit())
    throw std::runtime_error(
        "Cannot concatenate DataArrays: Units do not match.");
  if (a1.name() != a2.name())
    throw std::runtime_error(
        "Cannot concatenate DataArrays: Names do not match.");
  const auto &dims1 = a1.dimensions();
  const auto &dims2 = a2.dimensions();
  if (!(dims1 == dims2))
    throw std::runtime_error(
        "Cannot concatenate DataArrays: Dimensions do not match.");
  // Should we permit creation of ragged outputs if one dimension does not
  // match?
  auto out(a1);
  auto dims(dims1);
  if (dims.contains(dim)) {
    dims.resize(dim, dims1.size(dim) + dims2.size(dim));
    out.setDimensions(dims);
    auto offset = dims1.offset(dim) * dims1.size(dim);
    out.data().copyFrom(offset, 0, 2, a1.data());
    out.data().copyFrom(offset, 1, 2, a2.data());
  } else {
    dims.add(dim, 2);
    out.setDimensions(dims);
    out.data().copyFrom(a1.size(), 1, 2, a2.data());
  }
  return out;
}

#endif // DATA_ARRAY_H
