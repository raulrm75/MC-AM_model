import numpy as np
from cells import AbstractCell, CubicalCell
from algebra import ZModule, ChainComplex, Morphism
from cellcomplexes import CellComplex
from itertools import product
from cubparser import cubes_iter


def cell_models(dimension):
    models = {d: [] for d in range(dimension + 1)}
    for cell in product(range(2), repeat=dimension):
        models[sum(cell)].append(cell)
    return models


class CubicalComplex(CellComplex):
    _cells_array = None

    def __init__(self, shape=None):
        super(CubicalComplex, self).__init__()
        self._cells_array = None
        self._dim_masks = {}
        self.shape = shape

    def _create_dim_masks(self):
        models = cell_models(self.dim)
        indices = {dimension: [
            tuple(
                slice(0, s, 2) if m == 0 else slice(1, s, 2)
                for (m, s) in zip(model, self.shape))
            for model in models[dimension]] for dimension in self.dimensions}
        result = {dim: np.zeros_like(self._cells_array) for
            dim in self.dimensions}
        for dim in result:
            for slice_index in indices[dim]:
                result[dim][slice_index] = 1
        self._dim_masks = result

    @property
    def shape(self):
        if self._cells_array is None:
            return ()
        else:
            return self._cells_array.shape

    @shape.setter
    def shape(self, shape):
        assert all(i % 2 == 1 for i in shape)
        if self._cells_array is not None:
            if shape is None:
                self._cells_array = None
            else:
                self._cells_array.shape = shape
        elif shape is not None:
            self._cells_array = np.zeros(shape, int)

        if self.dim:
            self._create_dim_masks()

    @property
    def cells_array(self):
        return self._cells_array

    def add_cell(self, cell, add_faces=False):
        if not isinstance(cell, CubicalCell):
            raise TypeError(
                'Only CubicalCell can be added to a CubicalComplex,'
                ' rather than {}'.format(type(cell)))
        else:
            if self.shape is None:
                raise ValueError('The CubicalComplex has no initial shape.')
            else:
                self._cells_array[cell.cell] = 1

    def __contains__(self, cell):
        if not isinstance(cell, CubicalCell):
            return False
        try:
            return bool(self._cells_array[cell.cell])
        except IndexError:
            return False

    @property
    def dim(self):
        try:
            return len(self.shape)
        except TypeError:
            return 0

    def __call__(self, dimension):
        if dimension in self.dimensions:
            for idx in zip(*np.where(
                    self._cells_array * self._dim_masks[dimension])):
                yield CubicalCell(idx)
        else:
            return ()
    
    @classmethod
    def from_file(self, cub_file):
        #maximal_faces = list(cubes_iter(cub_file))
        indexes = [c.cell for c in cubes_iter(cub_file)]
        print(indexes)
        


if __name__ == '__main__':
    CubicalComplex.from_file('tests/kleinbot.cub')
