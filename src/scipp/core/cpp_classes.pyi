############################################
#               ATTENTION                  #
# This file was generated by tools/stubgen #
# Do not edit!                             #
############################################
# flake8: noqa

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Literal, Mapping, \
    Optional, Sequence, Tuple, Union, overload

from .._binding import _NoDefault
from ..coords.graph import GraphDict
from ..typing import Dims, VariableLike, VariableLikeType
from ..units import default_unit
from .bins import Bins

__all__ = [
    "BinEdgeError",
    "BinnedDataError",
    "CoordError",
    "Coords",
    "Coords_items_view",
    "Coords_keys_view",
    "Coords_values_view",
    "DType",
    "DTypeError",
    "DataArray",
    "DataArrayError",
    "Dataset",
    "DatasetError",
    "Dataset_items_view",
    "Dataset_keys_view",
    "Dataset_values_view",
    "DefaultUnit",
    "DimensionError",
    "GroupByDataArray",
    "GroupByDataset",
    "Masks",
    "Masks_items_view",
    "Masks_keys_view",
    "Masks_values_view",
    "Slice",
    "Unit",
    "UnitError",
    "Variable",
    "VariableError",
    "VariancesError"
]

class BinEdgeError(RuntimeError):
    ...

class BinnedDataError(RuntimeError):
    ...

class CoordError(RuntimeError):
    ...

class Coords:

    def __contains__(self, arg0: str) -> bool:
        ...

    def __delitem__(self, arg0: str) -> None:
        ...

    def __eq__(self, arg0: object) -> bool:  # type: ignore[override]
        ...

    def __getitem__(self, arg0: str) -> Variable:
        ...

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...

    def __ne__(self, arg0: object) -> bool:  # type: ignore[override]
        ...

    def __repr__(self) -> str:
        ...

    def __setitem__(self, arg0: str, arg1: Variable) -> None:
        ...

    def __str__(self) -> str:
        ...

    def _ipython_key_completions_(self) -> list:
        ...

    def _pop(self, k: str) -> Any:
        ...

    def get(self, key, default=None):
        ...

    def is_edges(self, key: str, dim: Optional[str]=None) -> bool:
        ...

    def items(self) -> Coords_items_view:
        ...

    def keys(self) -> Coords_keys_view:
        ...

    def pop(self, key, default=_NoDefault):
        ...

    def update(self, other: Any=None, /, **kwargs) -> None:
        ...

    def values(self) -> Coords_values_view:
        ...

class Coords_items_view:

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

class Coords_keys_view:

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

class Coords_values_view:

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

class DType:
    DataArray: DType = ...
    DataArrayView: DType = ...
    Dataset: DType = ...
    DatasetView: DType = ...
    PyObject: DType = ...
    Variable: DType = ...
    VariableView: DType = ...

    def __eq__(self, arg0: object) -> bool:  # type: ignore[override]
        ...

    def __init__(self, arg0: Any) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...
    affine_transform3: DType = ...
    bool: DType = ...
    datetime64: DType = ...
    float32: DType = ...
    float64: DType = ...
    int32: DType = ...
    int64: DType = ...
    linear_transform3: DType = ...
    rotation3: DType = ...
    string: DType = ...
    translation3: DType = ...
    vector3: DType = ...

class DTypeError(TypeError):
    ...

class DataArray:

    def __abs__(self) -> DataArray:
        ...

    @overload
    def __add__(self, arg0: Dataset) -> Dataset:
        ...

    @overload
    def __add__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __add__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __add__(self, arg0: float) -> DataArray:
        ...

    @overload
    def __and__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __and__(self, arg0: Variable) -> DataArray:
        ...

    def __bool__(self) -> None:
        ...

    def __copy__(self) -> DataArray:
        ...

    def __deepcopy__(self, arg0: dict) -> DataArray:
        ...

    def __eq__(self, arg0: object) -> DataArray:  # type: ignore[override]
        ...

    @overload
    def __floordiv__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __floordiv__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __floordiv__(self, arg0: float) -> DataArray:
        ...

    @overload
    def __ge__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __ge__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __getitem__(self, arg0: int) -> DataArray:
        ...

    @overload
    def __getitem__(self, arg0: slice) -> DataArray:
        ...

    @overload
    def __getitem__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __getitem__(self, arg0: Tuple[str, Variable]) -> DataArray:
        ...

    @overload
    def __getitem__(self, arg0: Tuple[str, int]) -> DataArray:
        ...

    @overload
    def __getitem__(self, arg0: Tuple[str, slice]) -> DataArray:
        ...

    @overload
    def __getitem__(self, arg0: ellipsis) -> DataArray:
        ...

    @overload
    def __getitem__(self, arg0: List[int]) -> DataArray:
        ...

    @overload
    def __getitem__(self, arg0: Tuple[str, List[int]]) -> DataArray:
        ...

    @overload
    def __gt__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __gt__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __iadd__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __iadd__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __iadd__(self, arg0: float) -> DataArray:
        ...

    @overload
    def __iand__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __iand__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __imod__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __imod__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __imod__(self, arg0: float) -> DataArray:
        ...

    @overload
    def __imul__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __imul__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __imul__(self, arg0: float) -> DataArray:
        ...

    def __init__(self, data: Variable, coords: Union[Mapping[str, Variable], Iterable[Tuple[str, Variable]]]={}, masks: Union[Mapping[str, Variable], Iterable[Tuple[str, Variable]]]={}, attrs: Union[Mapping[str, Variable], Iterable[Tuple[str, Variable]]]={}, name: str='') -> None:
        ...

    def __invert__(self) -> DataArray:
        ...

    @overload
    def __ior__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __ior__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __isub__(self, arg0: DataArray) -> Any:
        ...

    @overload
    def __isub__(self, arg0: Variable) -> Any:
        ...

    @overload
    def __isub__(self, arg0: float) -> Any:
        ...

    @overload
    def __itruediv__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __itruediv__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __itruediv__(self, arg0: float) -> DataArray:
        ...

    @overload
    def __ixor__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __ixor__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __le__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __le__(self, arg0: Variable) -> DataArray:
        ...

    def __len__(self) -> int:
        ...

    @overload
    def __lt__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __lt__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __mod__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __mod__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __mod__(self, arg0: float) -> DataArray:
        ...

    @overload
    def __mul__(self, arg0: Dataset) -> Dataset:
        ...

    @overload
    def __mul__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __mul__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __mul__(self, arg0: float) -> DataArray:
        ...

    def __ne__(self, arg0: object) -> DataArray:  # type: ignore[override]
        ...

    def __neg__(self) -> DataArray:
        ...

    @overload
    def __or__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __or__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __pow__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __pow__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __pow__(self, arg0: float) -> DataArray:
        ...

    def __radd__(self, arg0: float) -> DataArray:
        ...

    def __repr__(self) -> str:
        ...

    def __rfloordiv__(self, arg0: float) -> DataArray:
        ...

    def __rmod__(self, arg0: float) -> DataArray:
        ...

    def __rmul__(self, arg0: float) -> DataArray:
        ...

    def __rpow__(self, arg0: float) -> DataArray:
        ...

    def __rsub__(self, arg0: float) -> DataArray:
        ...

    def __rtruediv__(self, arg0: float) -> DataArray:
        ...

    @overload
    def __setitem__(self, arg0: Tuple[str, Variable], arg1: Variable) -> None:
        ...

    @overload
    def __setitem__(self, arg0: Tuple[str, Variable], arg1: DataArray) -> None:
        ...

    @overload
    def __setitem__(self, arg0: int, arg1: Any) -> None:
        ...

    @overload
    def __setitem__(self, arg0: slice, arg1: Any) -> None:
        ...

    @overload
    def __setitem__(self, arg0: Tuple[str, int], arg1: Any) -> None:
        ...

    @overload
    def __setitem__(self, arg0: Tuple[str, slice], arg1: Any) -> None:
        ...

    @overload
    def __setitem__(self, arg0: ellipsis, arg1: Any) -> None:
        ...

    def __sizeof__(self) -> int:
        ...

    @overload
    def __sub__(self, arg0: Dataset) -> Dataset:
        ...

    @overload
    def __sub__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __sub__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __sub__(self, arg0: float) -> DataArray:
        ...

    @overload
    def __truediv__(self, arg0: Dataset) -> Dataset:
        ...

    @overload
    def __truediv__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __truediv__(self, arg0: Variable) -> DataArray:
        ...

    @overload
    def __truediv__(self, arg0: float) -> DataArray:
        ...

    @overload
    def __xor__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __xor__(self, arg0: Variable) -> DataArray:
        ...

    def _ipython_key_completions_(self) -> list:
        ...

    def _rename_dims(self, arg0: Dict[str, str]) -> DataArray:
        ...

    def _repr_html_(self) -> str:
        ...

    def all(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def any(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def astype(self, type: Any, *, copy: bool=True) -> DataArray:
        ...

    @property
    def attrs(self) -> Coords:
        ...

    def bin(self, arg_dict=None, /, **kwargs):
        ...

    @property
    def bins(self):
        ...

    @bins.setter
    def bins(self, bins: Bins):
        ...

    def broadcast(self, dims: Optional[Union[List[str], Tuple[str, ...]]]=None, shape: Optional[Sequence[int]]=None, sizes: Optional[Dict[str, int]]=None) -> VariableLikeType:
        ...

    def ceil(self, *, out: Optional[VariableLike]=None) -> VariableLike:
        ...

    @property
    def coords(self) -> Coords:
        ...

    def copy(self, deep: bool=True) -> DataArray:
        ...

    @property
    def data(self) -> Variable:
        ...

    @data.setter
    def data(self, arg1: Variable) -> None:
        ...

    @property
    def dim(self) -> str:
        ...

    @property
    def dims(self) -> tuple:
        ...

    @overload
    def drop_attrs(self, arg0: str) -> DataArray:
        ...

    @overload
    def drop_attrs(self, arg0: List[str]) -> DataArray:
        ...

    @overload
    def drop_coords(self, arg0: str) -> DataArray:
        ...

    @overload
    def drop_coords(self, arg0: List[str]) -> DataArray:
        ...

    @overload
    def drop_masks(self, arg0: str) -> DataArray:
        ...

    @overload
    def drop_masks(self, arg0: List[str]) -> DataArray:
        ...

    @property
    def dtype(self) -> DType:
        ...

    def flatten(self, dims: Optional[Union[List[str], Tuple[str, ...]]]=None, to: Optional[str]=None) -> VariableLikeType:
        ...

    def floor(self, *, out: Optional[VariableLike]=None) -> VariableLike:
        ...

    def fold(self, dim: str, sizes: Optional[Dict[str, int]]=None, dims: Optional[Union[List[str], Tuple[str, ...]]]=None, shape: Optional[Sequence[int]]=None) -> VariableLikeType:
        ...

    def group(self, /, *args: Union[str, Variable]):
        ...

    def groupby(self, /, group: Union[Variable, str], *, bins: Optional[Variable]=None) -> Union[GroupByDataArray, GroupByDataset]:
        ...

    def hist(self, arg_dict=None, /, **kwargs):
        ...

    @property
    def masks(self) -> Masks:
        ...

    def max(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def mean(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    @property
    def meta(self) -> Coords:
        ...

    def min(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    @property
    def name(self) -> str:
        ...

    @name.setter
    def name(self, arg1: str) -> None:
        ...

    def nanhist(self, arg_dict=None, /, **kwargs):
        ...

    def nanmax(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def nanmean(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def nanmin(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def nansum(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    @property
    def ndim(self) -> int:
        ...

    def plot(*args, **kwargs):
        ...

    def rebin(self, arg_dict=None, deprecated=None, /, **kwargs):
        ...

    def rename(self, dims_dict: Dict[str, str]=None, /, **names: str) -> DataArray:
        ...

    def rename_dims(self, dims_dict: Optional[Dict[str, str]]=None, /, **names: str) -> VariableLikeType:
        ...

    def round(self, *, out: Optional[VariableLike]=None) -> VariableLike:
        ...

    @property
    def shape(self) -> tuple:
        ...

    @property
    def sizes(self) -> dict:
        ...

    def squeeze(self, dim: Optional[Union[str, List[str], Tuple[str, ...]]]=None) -> VariableLikeType:
        ...

    def sum(self, dim: Dims=None) -> VariableLikeType:
        ...

    def to(self, *, unit: Optional[Union[Unit, str]]=None, dtype: Optional[Any]=None, copy: bool=True) -> VariableLikeType:
        ...

    def to_hdf5(self, filename: Union[str, Path]):
        ...

    def transform_coords(self, targets: Optional[Union[str, Iterable[str]]]=None, /, graph: Optional[GraphDict]=None, *, rename_dims: bool=True, keep_aliases: bool=True, keep_intermediate: bool=True, keep_inputs: bool=True, quiet: bool=False, **kwargs: Callable) -> Union[DataArray, Dataset]:
        ...

    def transpose(self, dims: Optional[Union[List[str], Tuple[str, ...]]]=None) -> VariableLikeType:
        ...

    def underlying_size(self) -> int:
        ...

    @property
    def unit(self) -> Optional[Unit]:
        ...

    @unit.setter
    def unit(self, arg1: Union[str, Unit, None, DefaultUnit]) -> None:
        ...

    @property
    def value(self) -> Any:
        ...

    @value.setter
    def value(self, arg1: Any) -> None:
        ...

    @property
    def values(self) -> Any:
        ...

    @values.setter
    def values(self, arg1: Any) -> None:
        ...

    @property
    def variance(self) -> Any:
        ...

    @variance.setter
    def variance(self, arg1: Any) -> None:
        ...

    @property
    def variances(self) -> Any:
        ...

    @variances.setter
    def variances(self, arg1: Any) -> None:
        ...

class DataArrayError(RuntimeError):
    ...

class Dataset:

    def __abs__(self) -> Dataset:
        ...

    @overload
    def __add__(self, arg0: Dataset) -> Dataset:
        ...

    @overload
    def __add__(self, arg0: DataArray) -> Dataset:
        ...

    @overload
    def __add__(self, arg0: Variable) -> Dataset:
        ...

    def __bool__(self) -> None:
        ...

    def __contains__(self, arg0: str) -> bool:
        ...

    def __copy__(self) -> Dataset:
        ...

    def __deepcopy__(self, arg0: dict) -> Dataset:
        ...

    def __delitem__(self, arg0: str) -> None:
        ...

    @overload
    def __getitem__(self, arg0: str) -> DataArray:
        ...

    @overload
    def __getitem__(self, arg0: int) -> Dataset:
        ...

    @overload
    def __getitem__(self, arg0: slice) -> Dataset:
        ...

    @overload
    def __getitem__(self, arg0: Variable) -> Dataset:
        ...

    @overload
    def __getitem__(self, arg0: Tuple[str, Variable]) -> Dataset:
        ...

    @overload
    def __getitem__(self, arg0: Tuple[str, int]) -> Dataset:
        ...

    @overload
    def __getitem__(self, arg0: Tuple[str, slice]) -> Dataset:
        ...

    @overload
    def __getitem__(self, arg0: ellipsis) -> Dataset:
        ...

    @overload
    def __getitem__(self, arg0: List[int]) -> Dataset:
        ...

    @overload
    def __getitem__(self, arg0: Tuple[str, List[int]]) -> Dataset:
        ...

    @overload
    def __iadd__(self, arg0: Dataset) -> Dataset:
        ...

    @overload
    def __iadd__(self, arg0: DataArray) -> Dataset:
        ...

    @overload
    def __iadd__(self, arg0: Variable) -> Dataset:
        ...

    @overload
    def __iadd__(self, arg0: float) -> Dataset:
        ...

    @overload
    def __imul__(self, arg0: Dataset) -> Dataset:
        ...

    @overload
    def __imul__(self, arg0: DataArray) -> Dataset:
        ...

    @overload
    def __imul__(self, arg0: Variable) -> Dataset:
        ...

    @overload
    def __imul__(self, arg0: float) -> Dataset:
        ...

    def __init__(self, data: Union[Mapping[str, Union[Variable, DataArray]], Iterable[Tuple[str, Union[Variable, DataArray]]]]={}, coords: Union[Mapping[str, Variable], Iterable[Tuple[str, Variable]]]={}) -> None:
        ...

    @overload
    def __isub__(self, arg0: Dataset) -> Any:
        ...

    @overload
    def __isub__(self, arg0: DataArray) -> Any:
        ...

    @overload
    def __isub__(self, arg0: Variable) -> Any:
        ...

    @overload
    def __isub__(self, arg0: float) -> Any:
        ...

    def __iter__(self) -> Iterator:
        ...

    @overload
    def __itruediv__(self, arg0: Dataset) -> Dataset:
        ...

    @overload
    def __itruediv__(self, arg0: DataArray) -> Dataset:
        ...

    @overload
    def __itruediv__(self, arg0: Variable) -> Dataset:
        ...

    @overload
    def __itruediv__(self, arg0: float) -> Dataset:
        ...

    def __len__(self) -> int:
        ...

    @overload
    def __mul__(self, arg0: Dataset) -> Dataset:
        ...

    @overload
    def __mul__(self, arg0: DataArray) -> Dataset:
        ...

    @overload
    def __mul__(self, arg0: Variable) -> Dataset:
        ...

    def __repr__(self) -> str:
        ...

    @overload
    def __setitem__(self, arg0: str, arg1: Variable) -> None:
        ...

    @overload
    def __setitem__(self, arg0: str, arg1: DataArray) -> None:
        ...

    @overload
    def __setitem__(self, arg0: Tuple[str, Variable], arg1: Dataset) -> None:
        ...

    @overload
    def __setitem__(self, arg0: int, arg1: Any) -> None:
        ...

    @overload
    def __setitem__(self, arg0: slice, arg1: Any) -> None:
        ...

    @overload
    def __setitem__(self, arg0: Tuple[str, int], arg1: Any) -> None:
        ...

    @overload
    def __setitem__(self, arg0: Tuple[str, slice], arg1: Any) -> None:
        ...

    @overload
    def __setitem__(self, arg0: ellipsis, arg1: Any) -> None:
        ...

    def __sizeof__(self) -> int:
        ...

    @overload
    def __sub__(self, arg0: Dataset) -> Dataset:
        ...

    @overload
    def __sub__(self, arg0: DataArray) -> Dataset:
        ...

    @overload
    def __sub__(self, arg0: Variable) -> Dataset:
        ...

    @overload
    def __truediv__(self, arg0: Dataset) -> Dataset:
        ...

    @overload
    def __truediv__(self, arg0: DataArray) -> Dataset:
        ...

    @overload
    def __truediv__(self, arg0: Variable) -> Dataset:
        ...

    def _ipython_key_completions_(self) -> list:
        ...

    def _pop(self, k: str) -> Any:
        ...

    def _rename_dims(self, arg0: Dict[str, str]) -> Dataset:
        ...

    def _repr_html_(self) -> str:
        ...

    def all(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def any(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    @property
    def bins(self):
        ...

    @bins.setter
    def bins(self, bins: Bins):
        ...

    def clear(self) -> None:
        ...

    @property
    def coords(self) -> Coords:
        ...

    def copy(self, deep: bool=True) -> Dataset:
        ...

    @property
    def dim(self) -> str:
        ...

    @property
    def dims(self) -> tuple:
        ...

    @overload
    def drop_coords(self, arg0: str) -> Dataset:
        ...

    @overload
    def drop_coords(self, arg0: List[str]) -> Dataset:
        ...

    def get(self, key, default=None):
        ...

    def groupby(self, /, group: Union[Variable, str], *, bins: Optional[Variable]=None) -> Union[GroupByDataArray, GroupByDataset]:
        ...

    def hist(self, arg_dict=None, /, **kwargs):
        ...

    def items(self) -> Dataset_items_view:
        ...

    def keys(self) -> Dataset_keys_view:
        ...

    def max(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def mean(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    @property
    def meta(self) -> Coords:
        ...

    def min(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def nanmax(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def nanmean(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def nanmin(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def nansum(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    @property
    def ndim(self) -> int:
        ...

    def plot(*args, **kwargs):
        ...

    def pop(self, key, default=_NoDefault):
        ...

    def rebin(self, arg_dict=None, deprecated=None, /, **kwargs):
        ...

    def rename(self, dims_dict: Dict[str, str]=None, /, **names: str) -> Dataset:
        ...

    def rename_dims(self, dims_dict: Optional[Dict[str, str]]=None, /, **names: str) -> VariableLikeType:
        ...

    @property
    def shape(self) -> tuple:
        ...

    @property
    def sizes(self) -> dict:
        ...

    def squeeze(self, dim: Optional[Union[str, List[str], Tuple[str, ...]]]=None) -> VariableLikeType:
        ...

    def sum(self, dim: Dims=None) -> VariableLikeType:
        ...

    def to_hdf5(self, filename: Union[str, Path]):
        ...

    def transform_coords(self, targets: Optional[Union[str, Iterable[str]]]=None, /, graph: Optional[GraphDict]=None, *, rename_dims: bool=True, keep_aliases: bool=True, keep_intermediate: bool=True, keep_inputs: bool=True, quiet: bool=False, **kwargs: Callable) -> Union[DataArray, Dataset]:
        ...

    def underlying_size(self) -> int:
        ...

    def update(self, other: Any=None, /, **kwargs) -> None:
        ...

    def values(self) -> Dataset_values_view:
        ...

class DatasetError(RuntimeError):
    ...

class Dataset_items_view:

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

class Dataset_keys_view:

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

class Dataset_values_view:

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

class DefaultUnit:

    def __repr__(self) -> str:
        ...

class DimensionError(RuntimeError):
    ...

class GroupByDataArray:

    def all(self, dim: str) -> DataArray:
        ...

    def any(self, dim: str) -> DataArray:
        ...

    @property
    def bins(self):
        ...

    def concat(self, dim: str) -> DataArray:
        ...

    def max(self, dim: str) -> DataArray:
        ...

    def mean(self, dim: str) -> DataArray:
        ...

    def min(self, dim: str) -> DataArray:
        ...

    def nanmax(self, dim: str) -> DataArray:
        ...

    def nanmin(self, dim: str) -> DataArray:
        ...

    def nansum(self, dim: str) -> DataArray:
        ...

    def sum(self, dim: str) -> DataArray:
        ...

class GroupByDataset:

    def all(self, dim: str) -> Dataset:
        ...

    def any(self, dim: str) -> Dataset:
        ...

    @property
    def bins(self):
        ...

    def concat(self, dim: str) -> Dataset:
        ...

    def max(self, dim: str) -> Dataset:
        ...

    def mean(self, dim: str) -> Dataset:
        ...

    def min(self, dim: str) -> Dataset:
        ...

    def nanmax(self, dim: str) -> Dataset:
        ...

    def nanmin(self, dim: str) -> Dataset:
        ...

    def nansum(self, dim: str) -> Dataset:
        ...

    def sum(self, dim: str) -> Dataset:
        ...

class Masks:

    def __contains__(self, arg0: str) -> bool:
        ...

    def __delitem__(self, arg0: str) -> None:
        ...

    def __eq__(self, arg0: object) -> bool:  # type: ignore[override]
        ...

    def __getitem__(self, arg0: str) -> Variable:
        ...

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...

    def __ne__(self, arg0: object) -> bool:  # type: ignore[override]
        ...

    def __repr__(self) -> str:
        ...

    def __setitem__(self, arg0: str, arg1: Variable) -> None:
        ...

    def __str__(self) -> str:
        ...

    def _ipython_key_completions_(self) -> list:
        ...

    def _pop(self, k: str) -> Any:
        ...

    def get(self, key, default=None):
        ...

    def is_edges(self, key: str, dim: Optional[str]=None) -> bool:
        ...

    def items(self) -> Masks_items_view:
        ...

    def keys(self) -> Masks_keys_view:
        ...

    def pop(self, key, default=_NoDefault):
        ...

    def update(self, other: Any=None, /, **kwargs) -> None:
        ...

    def values(self) -> Masks_values_view:
        ...

class Masks_items_view:

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

class Masks_keys_view:

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

class Masks_values_view:

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

class Slice:
    ...

class Unit:

    def __abs__(self) -> Unit:
        ...

    def __add__(self, arg0: Unit) -> Unit:
        ...

    def __eq__(self, arg0: object) -> bool:  # type: ignore[override]
        ...

    def __hash__(self) -> int:
        ...

    def __init__(self, arg0: str) -> None:
        ...

    def __mul__(self, arg0: Unit) -> Unit:
        ...

    def __ne__(self, arg0: object) -> bool:  # type: ignore[override]
        ...

    def __pow__(self, arg0: int) -> Unit:
        ...

    def __repr__(self) -> str:
        ...

    def __rmul__(self, value):
        ...

    def __rtruediv__(self, value):
        ...

    def __str__(self) -> str:
        ...

    def __sub__(self, arg0: Unit) -> Unit:
        ...

    def __truediv__(self, arg0: Unit) -> Unit:
        ...

    def _repr_html_(self) -> str:
        ...

    def _repr_pretty_(self, arg0: Any, arg1: bool) -> None:
        ...

    def from_dict(self) -> Unit:
        ...

    @property
    def name(self) -> str:
        ...

    def to_dict(self) -> dict:
        ...

class UnitError(RuntimeError):
    ...

class Variable:

    def __abs__(self) -> Variable:
        ...

    @overload
    def __add__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __add__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __add__(self, arg0: float) -> Variable:
        ...

    def __and__(self, arg0: Variable) -> Variable:
        ...

    def __bool__(self) -> bool:
        ...

    def __copy__(self) -> Variable:
        ...

    def __deepcopy__(self, arg0: dict) -> Variable:
        ...

    def __eq__(self, arg0: object) -> Variable:  # type: ignore[override]
        ...

    def __float__(self) -> float:
        ...

    @overload
    def __floordiv__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __floordiv__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __floordiv__(self, arg0: float) -> Variable:
        ...

    def __format__(self, format_spec: str) -> str:
        ...

    @overload
    def __ge__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __ge__(self, arg0: float) -> Variable:
        ...

    @overload
    def __getitem__(self, arg0: int) -> Variable:
        ...

    @overload
    def __getitem__(self, arg0: slice) -> Variable:
        ...

    @overload
    def __getitem__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __getitem__(self, arg0: Tuple[str, int]) -> Variable:
        ...

    @overload
    def __getitem__(self, arg0: Tuple[str, slice]) -> Variable:
        ...

    @overload
    def __getitem__(self, arg0: ellipsis) -> Variable:
        ...

    @overload
    def __getitem__(self, arg0: List[int]) -> Variable:
        ...

    @overload
    def __getitem__(self, arg0: Tuple[str, List[int]]) -> Variable:
        ...

    @overload
    def __gt__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __gt__(self, arg0: float) -> Variable:
        ...

    @overload
    def __iadd__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __iadd__(self, arg0: float) -> Variable:
        ...

    def __iand__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __imod__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __imod__(self, arg0: float) -> Variable:
        ...

    @overload
    def __imul__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __imul__(self, arg0: float) -> Variable:
        ...

    def __init__(self, *, dims: Any, values: Any=None, variances: Any=None, unit: Union[str, Unit, None, DefaultUnit]=default_unit, dtype: Any=None) -> None:
        ...

    def __int__(self) -> int:
        ...

    def __invert__(self) -> Variable:
        ...

    def __ior__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __ipow__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __ipow__(self, arg0: float) -> Variable:
        ...

    @overload
    def __isub__(self, arg0: Variable) -> Any:
        ...

    @overload
    def __isub__(self, arg0: float) -> Any:
        ...

    @overload
    def __itruediv__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __itruediv__(self, arg0: float) -> Variable:
        ...

    def __ixor__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __le__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __le__(self, arg0: float) -> Variable:
        ...

    def __len__(self) -> int:
        ...

    @overload
    def __lt__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __lt__(self, arg0: float) -> Variable:
        ...

    @overload
    def __mod__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __mod__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __mod__(self, arg0: float) -> Variable:
        ...

    @overload
    def __mul__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __mul__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __mul__(self, arg0: float) -> Variable:
        ...

    def __ne__(self, arg0: object) -> Variable:  # type: ignore[override]
        ...

    def __neg__(self) -> Variable:
        ...

    def __or__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __pow__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __pow__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __pow__(self, arg0: float) -> Variable:
        ...

    def __radd__(self, arg0: float) -> Variable:
        ...

    def __repr__(self) -> str:
        ...

    def __rfloordiv__(self, arg0: float) -> Variable:
        ...

    def __rmod__(self, arg0: float) -> Variable:
        ...

    def __rmul__(self, arg0: float) -> Variable:
        ...

    def __rpow__(self, arg0: float) -> Variable:
        ...

    def __rsub__(self, arg0: float) -> Variable:
        ...

    def __rtruediv__(self, arg0: float) -> Variable:
        ...

    @overload
    def __setitem__(self, arg0: int, arg1: Any) -> None:
        ...

    @overload
    def __setitem__(self, arg0: slice, arg1: Any) -> None:
        ...

    @overload
    def __setitem__(self, arg0: Tuple[str, int], arg1: Any) -> None:
        ...

    @overload
    def __setitem__(self, arg0: Tuple[str, slice], arg1: Any) -> None:
        ...

    @overload
    def __setitem__(self, arg0: ellipsis, arg1: Any) -> None:
        ...

    def __sizeof__(self) -> int:
        ...

    @overload
    def __sub__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __sub__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __sub__(self, arg0: float) -> Variable:
        ...

    @overload
    def __truediv__(self, arg0: Variable) -> Variable:
        ...

    @overload
    def __truediv__(self, arg0: DataArray) -> DataArray:
        ...

    @overload
    def __truediv__(self, arg0: float) -> Variable:
        ...

    def __xor__(self, arg0: Variable) -> Variable:
        ...

    def _ipython_key_completions_(self) -> list:
        ...

    def _rename_dims(self, arg0: Dict[str, str]) -> Variable:
        ...

    def _repr_html_(self) -> str:
        ...

    def all(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def any(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def astype(self, type: Any, *, copy: bool=True) -> Variable:
        ...

    def bin(self, arg_dict=None, /, **kwargs):
        ...

    @property
    def bins(self):
        ...

    @bins.setter
    def bins(self, bins: Bins):
        ...

    def broadcast(self, dims: Optional[Union[List[str], Tuple[str, ...]]]=None, shape: Optional[Sequence[int]]=None, sizes: Optional[Dict[str, int]]=None) -> VariableLikeType:
        ...

    def ceil(self, *, out: Optional[VariableLike]=None) -> VariableLike:
        ...

    def copy(self, deep: bool=True) -> Variable:
        ...

    def cumsum(self, dim: Optional[str]=None, mode: Literal['exclusive', 'inclusive']='inclusive') -> VariableLikeType:
        ...

    @property
    def dim(self) -> str:
        ...

    @property
    def dims(self) -> tuple:
        ...

    @property
    def dtype(self) -> DType:
        ...

    @property
    def fields(self):
        ...

    def flatten(self, dims: Optional[Union[List[str], Tuple[str, ...]]]=None, to: Optional[str]=None) -> VariableLikeType:
        ...

    def floor(self, *, out: Optional[VariableLike]=None) -> VariableLike:
        ...

    def fold(self, dim: str, sizes: Optional[Dict[str, int]]=None, dims: Optional[Union[List[str], Tuple[str, ...]]]=None, shape: Optional[Sequence[int]]=None) -> VariableLikeType:
        ...

    def hist(self, arg_dict=None, /, **kwargs):
        ...

    def max(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def mean(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def min(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def nanhist(self, arg_dict=None, /, **kwargs):
        ...

    def nanmax(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def nanmean(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def nanmin(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    def nansum(self, dim: Optional[str]=None) -> VariableLikeType:
        ...

    @property
    def ndim(self) -> int:
        ...

    def plot(*args, **kwargs):
        ...

    def rename(self, dims_dict: Dict[str, str]=None, /, **names: str) -> Variable:
        ...

    def rename_dims(self, dims_dict: Optional[Dict[str, str]]=None, /, **names: str) -> VariableLikeType:
        ...

    def round(self, *, out: Optional[VariableLike]=None) -> VariableLike:
        ...

    @property
    def shape(self) -> tuple:
        ...

    @property
    def sizes(self) -> dict:
        ...

    def squeeze(self, dim: Optional[Union[str, List[str], Tuple[str, ...]]]=None) -> VariableLikeType:
        ...

    def sum(self, dim: Dims=None) -> VariableLikeType:
        ...

    def to(self, *, unit: Optional[Union[Unit, str]]=None, dtype: Optional[Any]=None, copy: bool=True) -> VariableLikeType:
        ...

    def to_hdf5(self, filename: Union[str, Path]):
        ...

    def transpose(self, dims: Optional[Union[List[str], Tuple[str, ...]]]=None) -> VariableLikeType:
        ...

    def underlying_size(self) -> int:
        ...

    @property
    def unit(self) -> Optional[Unit]:
        ...

    @unit.setter
    def unit(self, arg1: Union[str, Unit, None, DefaultUnit]) -> None:
        ...

    @property
    def value(self) -> Any:
        ...

    @value.setter
    def value(self, arg1: Any) -> None:
        ...

    @property
    def values(self) -> Any:
        ...

    @values.setter
    def values(self, arg1: Any) -> None:
        ...

    @property
    def variance(self) -> Any:
        ...

    @variance.setter
    def variance(self, arg1: Any) -> None:
        ...

    @property
    def variances(self) -> Any:
        ...

    @variances.setter
    def variances(self, arg1: Any) -> None:
        ...

class VariableError(RuntimeError):
    ...

class VariancesError(RuntimeError):
    ...