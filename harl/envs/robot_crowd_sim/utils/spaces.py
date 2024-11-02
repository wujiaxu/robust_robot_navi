import numpy as np

class NaviObsSpace:

  def __init__(self, shapes:dict, dtype, name: str):
    """Initializes a new `Array` spec.

    Args:
      shape: An iterable specifying the array shape.
      dtype: numpy dtype or string specifying the array dtype.
      name: Optional string containing a semantic name for the corresponding
        array. Defaults to `None`.

    Raises:
      TypeError: If `shape` is not an iterable of elements convertible to int,
      or if `dtype` is not convertible to a numpy dtype.
    """
    self._shape_total = 0
    self._shapes = shapes
    self._shape_dims = {}
    for compo_name in shapes.keys():
        dim = 1
        shape = shapes[compo_name]
        for d in shape:
            dim*=d
        self._shape_total+=dim
        self._shape_dims[compo_name] = (shape,dim)
    self._dtype = np.dtype(dtype)
    self._name = name

  def get_shape(self,name):
      return self._shapes[name]

  @property
  def shape_dict(self):
    """Returns a `tuple` specifying the array shape."""
    return self._shapes
  @property
  def shape(self):
    """Returns a `tuple` specifying the array shape."""
    return (self._shape_total,)

  @property
  def dtype(self):
    """Returns a numpy dtype specifying the array dtype."""
    return self._dtype

  @property
  def name(self):
    """Returns the name of the Array."""
    return self._name
  