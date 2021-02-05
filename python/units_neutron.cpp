// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "scipp/core/dtype.h"
#include "scipp/core/tag_util.h"
#include "scipp/units/unit.h"
#include "scipp/variable/operations.h"
#include "scipp/variable/variable.h"

#include "dtype.h"
#include "pybind11.h"

using namespace scipp;
namespace py = pybind11;

namespace {
template <class T> struct MultScalarUnit {
  static Variable apply(const py::object &scalar, const units::Unit &unit) {
    return py::cast<T>(scalar) * unit;
  }
};

Variable doMultScalarUnit(const units::Unit &unit, const py::object &scalar,
                          const py::dtype &type) {
  return scipp::core::CallDType<double, float, int64_t, int32_t>::apply<
      MultScalarUnit>(scipp_dtype(type), scalar, unit);
}

template <class T> struct DivScalarUnit {
  static Variable apply(const py::object &scalar, const units::Unit &unit) {
    return py::cast<T>(scalar) / unit;
  }
};

Variable doDivScalarUnit(const units::Unit &unit, const py::object &scalar,
                         const py::dtype &type) {
  return scipp::core::CallDType<double, float, int64_t, int32_t>::apply<
      DivScalarUnit>(scipp_dtype(type), scalar, unit);
}
} // namespace

void init_units_neutron(py::module &m) {
  py::class_<units::Dim>(m, "Dim", "Dimension label")
      .def(py::init<const std::string &>())
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(hash(py::self))
      .def("__repr__", [](const Dim &dim) { return dim.name(); });

  py::class_<units::Unit>(m, "Unit", "A physical unit.")
      .def(py::init())
      .def(py::init<const std::string &>())
      .def("__repr__", [](const units::Unit &u) { return u.name(); })
      .def_property_readonly("name", &units::Unit::name,
                             "A read-only string describing the "
                             "type of unit.")
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self)
      .def("__pow__", [](const units::Unit &self,
                         const int64_t power) { return pow(self, power); })
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__rmul",
           [](const units::Unit &self, double scalar) { return scalar * self; })
      .def("__rmul", [](const units::Unit &self,
                        int64_t scalar) { return scalar * self; })
      .def("__rmul", &doMultScalarUnit)
      .def("__rtruediv",
           [](const units::Unit &self, double scalar) { return scalar / self; })
      .def("__rtruediv", [](const units::Unit &self,
                            int64_t scalar) { return scalar / self; })
      .def("__rtruediv", &doDivScalarUnit);

  auto units = m.def_submodule("units");
  units.attr("angstrom") = units::angstrom;
  units.attr("counts") = units::counts;
  units.attr("deg") = units::deg;
  units.attr("dimensionless") = units::dimensionless;
  units.attr("kg") = units::kg;
  units.attr("K") = units::K;
  units.attr("meV") = units::meV;
  units.attr("m") = units::m;
  units.attr("one") = units::one;
  units.attr("rad") = units::rad;
  units.attr("s") = units::s;
  units.attr("us") = units::us;
  units.attr("ns") = units::ns;
  units.attr("mm") = units::mm;
}
