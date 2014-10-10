from cells import AbstractCell
from algebra import ZModule, ChainComplex, Morphism


class CellComplex(object):
    _cells = {}

    def __init__(self):
        # Cells grouped by dimension.
        # Every element in self._cells[d] must be an AbstractCell instance
        self._cells = {}

    def __contains__(self, cell):
        return (cell.dim in self.dimensions) and (cell in self._cells[cell.dim])

    def star(self, cell):
        for dim in (d for d in self.dimensions if d > cell.dim):
            for facet in self(dim):
                if cell <= facet and cell != facet:
                    yield facet
                else:
                    continue

    def facets(self, cell):
        if (cell.dim + 1) in self.dimensions:
            return (c for c in self(cell.dim + 1) if cell < c)
        else:
            return ()

    def add_cell(self, cell, add_faces=False):
        if isinstance(cell, AbstractCell):
            d = cell.dim
            if d in self._cells:
                if cell not in self._cells[d]:
                    self._cells[d].append(cell)
            else:
                self._cells[d] = [cell]
            if add_faces:
                for face in cell.boundary:
                    self.add_cell(face, add_faces)
        else:
            raise SyntaxError(
                'A valid cell must be provided, instead of {}'.format(
                    type(cell)))

    def __call__(self, dimension):
        if dimension in self.dimensions:
            return (cell for cell in self._cells[dimension])
        else:
            return ()

    def __iter__(self):
        for d in self.dimensions:
            for cell in self(d):
                yield cell

    @property
    def dim(self):
        return max(self._cells.keys())

    @property
    def chain_complex(self):
        modules = {}
        differentials = {}
        for q in range(self.dim + 1):
            base = tuple(self(q))
            if base:
                modules[q] = ZModule(base)
            else:
                modules[q] = ZModule()
        modules[-1] = ZModule()

        for q in range(self.dim + 1):
            differentials[q] = Morphism(modules[q], modules[q - 1])
            for cell in modules[q].base:
                differentials[q][cell] = cell.differential

        return ChainComplex(modules, differentials)

    @property
    def dimensions(self):
        return range(self.dim + 1)
