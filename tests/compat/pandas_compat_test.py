import pandas
import pytest

import scipp as sc
from scipp.compat import from_pandas
from scipp.compat.pandas_compat import parse_bracket_head
from scipp.testing import assert_identical


def _make_reference_da(row_name, row_coords, values, dtype="int64"):
    return sc.DataArray(
        data=sc.Variable(dims=[row_name], values=values, dtype=dtype),
        coords={row_name: sc.Variable(dims=[row_name], values=row_coords, dtype=dtype)},
        name=row_name,
    )


def _make_1d_reference_ds(row_name, data_name, values, coords, dtype="int64"):
    return sc.Dataset(
        data={data_name: sc.Variable(dims=[row_name], values=values, dtype=dtype)},
        coords={row_name: sc.Variable(dims=[row_name], values=coords, dtype=dtype)},
    )


def _make_nd_reference_ds(row_name, row_coords, data, dtype="int64"):
    return sc.Dataset(
        data={
            key: sc.Variable(dims=[row_name], values=value, dtype=dtype)
            for key, value in data.items()
        },
        coords={
            row_name: sc.Variable(dims=[row_name], values=row_coords),
        },
    )


def test_series():
    pd_df = pandas.Series(data=[1, 2, 3])

    sc_ds = from_pandas(pd_df)

    reference_da = _make_reference_da("row", [0, 1, 2], [1, 2, 3])

    assert sc.identical(sc_ds, reference_da)


def test_series_with_named_axis():
    pd_df = pandas.Series(data=[1, 2, 3])
    pd_df.rename_axis("row-name", inplace=True)

    sc_ds = from_pandas(pd_df)

    reference_da = _make_reference_da("row-name", [0, 1, 2], [1, 2, 3])

    assert sc.identical(sc_ds, reference_da)


def test_series_with_named_axis_non_str():
    pd_df = pandas.Series(data=[1, 2, 3])
    pd_df.rename_axis(987, inplace=True)

    sc_ds = from_pandas(pd_df)

    reference_da = _make_reference_da("987", [0, 1, 2], [1, 2, 3])

    assert sc.identical(sc_ds, reference_da)


def test_series_with_named_series():
    pd_df = pandas.Series(data=[1, 2, 3])
    pd_df.name = "the name"

    sc_ds = from_pandas(pd_df)

    reference_da = _make_reference_da("row", [0, 1, 2], [1, 2, 3])
    reference_da.name = "the name"

    assert sc.identical(sc_ds, reference_da)


def test_series_with_named_series_no_str():
    pd_df = pandas.Series(data=[1, 2, 3])
    pd_df.name = 8461

    sc_ds = from_pandas(pd_df)

    reference_da = _make_reference_da("row", [0, 1, 2], [1, 2, 3])
    reference_da.name = "8461"

    assert sc.identical(sc_ds, reference_da)


def test_series_with_named_series_and_named_axis():
    pd_df = pandas.Series(data=[1, 2, 3])
    pd_df.rename_axis("axis-name", inplace=True)
    pd_df.name = "series-name"

    sc_ds = from_pandas(pd_df)

    reference_da = _make_reference_da("axis-name", [0, 1, 2], [1, 2, 3])
    reference_da.name = "series-name"

    assert sc.identical(sc_ds, reference_da)


def test_series_without_index_coord():
    pd_df = pandas.Series(data=[1, 2, 3])

    sc_ds = from_pandas(pd_df, include_index=False)

    reference_da = _make_reference_da("row", [0, 1, 2], [1, 2, 3])
    reference_da.coords.clear()

    assert sc.identical(sc_ds, reference_da)


def test_1d_dataframe():
    pd_df = pandas.DataFrame(data=[1, 2, 3])

    sc_ds = from_pandas(pd_df)

    reference_ds = _make_1d_reference_ds("row", "0", [1, 2, 3], [0, 1, 2])

    assert sc.identical(sc_ds, reference_ds)


def test_1d_dataframe_with_named_axis():
    pd_df = pandas.DataFrame(data={"my-column": [1, 2, 3]})
    pd_df.rename_axis("1d_df", inplace=True)

    sc_ds = from_pandas(pd_df)

    reference_ds = _make_1d_reference_ds("1d_df", "my-column", [1, 2, 3], [0, 1, 2])

    assert sc.identical(sc_ds, reference_ds)


def test_1d_dataframe_without_index_coord():
    pd_df = pandas.DataFrame(data=[1, 2, 3])

    sc_ds = from_pandas(pd_df, include_index=False)

    reference_ds = _make_1d_reference_ds("row", "0", [1, 2, 3], [0, 1, 2])
    reference_ds.coords.clear()

    assert sc.identical(sc_ds, reference_ds)


def test_2d_dataframe():
    pd_df = pandas.DataFrame(data={"col1": (2, 3), "col2": (5, 6)})

    sc_ds = from_pandas(pd_df)

    reference_ds = _make_nd_reference_ds(
        "row", [0, 1], data={"col1": (2, 3), "col2": (5, 6)}
    )

    assert sc.identical(sc_ds, reference_ds)


def test_2d_dataframe_with_named_axes():
    pd_df = pandas.DataFrame(data={"col1": (2, 3), "col2": (5, 6)})
    pd_df.rename_axis("my-name-for-rows", inplace=True)

    sc_ds = from_pandas(pd_df)

    reference_ds = _make_nd_reference_ds(
        "my-name-for-rows", [0, 1], data={"col1": (2, 3), "col2": (5, 6)}
    )

    assert sc.identical(sc_ds, reference_ds)


def test_dataframe_select_single_data():
    pd_df = pandas.DataFrame(data={"col1": (1, 2), "col2": (6, 3), "col3": (4, 0)})

    sc_ds = from_pandas(pd_df, data_columns="col2")
    reference_ds = _make_nd_reference_ds(
        "row", [0, 1], data={"col1": (1, 2), "col2": (6, 3), "col3": (4, 0)}
    )
    reference_ds.coords["col1"] = reference_ds.pop("col1").data
    reference_ds.coords["col3"] = reference_ds.pop("col3").data
    assert_identical(sc_ds, reference_ds)

    sc_ds = from_pandas(pd_df, data_columns=["col1"])
    reference_ds = _make_nd_reference_ds(
        "row", [0, 1], data={"col1": (1, 2), "col2": (6, 3), "col3": (4, 0)}
    )
    reference_ds.coords["col2"] = reference_ds.pop("col2").data
    reference_ds.coords["col3"] = reference_ds.pop("col3").data
    assert_identical(sc_ds, reference_ds)


def test_dataframe_select_multiple_data():
    pd_df = pandas.DataFrame(data={"col1": (1, 2), "col2": (6, 3), "col3": (4, 0)})

    sc_ds = from_pandas(pd_df, data_columns=["col3", "col1"])
    reference_ds = _make_nd_reference_ds(
        "row", [0, 1], data={"col1": (1, 2), "col2": (6, 3), "col3": (4, 0)}
    )
    reference_ds.coords["col2"] = reference_ds.pop("col2").data
    assert_identical(sc_ds, reference_ds)


def test_dataframe_select_no_data():
    pd_df = pandas.DataFrame(data={"col1": (1, 2), "col2": (6, 3), "col3": (4, 0)})

    sc_ds = from_pandas(pd_df, data_columns=[])
    reference_ds = _make_nd_reference_ds(
        "row", [0, 1], data={"col1": (1, 2), "col2": (6, 3), "col3": (4, 0)}
    )
    reference_ds.coords["col1"] = reference_ds.pop("col1").data
    reference_ds.coords["col2"] = reference_ds.pop("col2").data
    reference_ds.coords["col3"] = reference_ds.pop("col3").data
    assert_identical(sc_ds, reference_ds)


def test_dataframe_select_undefined_raises():
    pd_df = pandas.DataFrame(data={"col1": (1, 2), "col2": (6, 3), "col3": (4, 0)})

    with pytest.raises(KeyError):
        _ = from_pandas(pd_df, data_columns=["unknown"])


def test_2d_dataframe_does_not_parse_units_by_default():
    pd_df = pandas.DataFrame(data={"col1 [m]": (1, 2), "col2 [one]": (6, 3)})

    sc_ds = from_pandas(pd_df)

    reference_ds = _make_nd_reference_ds(
        "row", [0, 1], data={"col1 [m]": (1, 2), "col2 [one]": (6, 3)}
    )

    assert_identical(sc_ds, reference_ds)


def test_2d_dataframe_parse_units_brackets():
    pd_df = pandas.DataFrame(data={"col1 [m]": (1, 2), "col2 [one]": (6, 3)})

    sc_ds = from_pandas(pd_df, head_parser="bracket")

    reference_ds = _make_nd_reference_ds(
        "row", [0, 1], data={"col1": (1, 2), "col2": (6, 3)}
    )
    reference_ds["col1"].unit = "m"
    reference_ds["col2"].unit = "one"

    assert_identical(sc_ds, reference_ds)


def test_2d_dataframe_parse_units_brackets_string_dtype():
    pd_df = pandas.DataFrame(
        data={"col1 [m]": ("a", "b"), "col2": ("c", "d")}, dtype="string"
    )

    sc_ds = from_pandas(pd_df, head_parser="bracket")

    reference_ds = _make_nd_reference_ds(
        "row",
        [0, 1],
        data={"col1": ("a", "b"), "col2": ("c", "d")},
        dtype="str",
    )
    reference_ds["col1"].unit = "m"
    reference_ds["col2"].unit = None

    assert_identical(sc_ds, reference_ds)


def test_parse_bracket_head_whitespace():
    name, unit = parse_bracket_head("")
    assert name == ""
    assert unit == sc.units.default_unit

    name, unit = parse_bracket_head(" ")
    assert name == " "
    assert unit == sc.units.default_unit


def test_parse_bracket_head_only_name():
    name, unit = parse_bracket_head("a name 123")
    assert name == "a name 123"
    assert unit == sc.units.default_unit

    name, unit = parse_bracket_head(" padded name  ")
    assert name == " padded name  "
    assert unit == sc.units.default_unit


def test_parse_bracket_head_only_unit():
    name, unit = parse_bracket_head("[m]")
    assert name == ""
    assert unit == "m"

    name, unit = parse_bracket_head(" [kg]")
    assert name == ""
    assert unit == "kg"


def test_parse_bracket_head_name_and_unit():
    name, unit = parse_bracket_head("the name [s]")
    assert name == "the name"
    assert unit == "s"

    name, unit = parse_bracket_head("title[A]")
    assert name == "title"
    assert unit == "A"


def test_parse_bracket_head_empty_unit():
    name, unit = parse_bracket_head("name []")
    assert name == "name"
    assert unit == sc.units.default_unit


def test_parse_bracket_head_dimensionless():
    name, unit = parse_bracket_head("name [one]")
    assert name == "name"
    assert unit == "one"

    name, unit = parse_bracket_head("name [dimensionless]")
    assert name == "name"
    assert unit == "one"


def test_parse_bracket_head_complex_unit():
    name, unit = parse_bracket_head("name [m / s**2]")
    assert name == "name"
    assert unit == "m/s**2"


def test_parse_bracket_head_bad_string():
    with pytest.raises(ValueError):
        parse_bracket_head("too [many] [brackets]")
