from itertools import product
from functools import total_ordering
import numpy as np
from itemproperties import itemproperty
import CHomP


class Chain(object):
    _coeff = {}

    def __init__(self, arg=None):
        from cells import AbstractCell
        self._coeff = {}
        if arg is None:
            self._coeff = {}
        elif isinstance(arg, AbstractCell):
            self._coeff = {arg: 1}
        elif isinstance(arg, dict):
            for cell in arg:
                if arg[cell] != 0:
                    self._coeff[cell] = arg[cell]
        elif isinstance(self, Chain):
            self._coeff.update(arg._coeff)
        else:
            raise SyntaxError('A valid cellular object, dictionary or Chain' +
                'must be provided.')

    def __hash__(self):
        return hash(tuple(hash(cell) for cell in self.support))

    def __iter__(self):
        for cell in self._coeff:
            yield cell

    def __getitem__(self, cell):
        if cell in self._coeff:
            return self._coeff[cell]
        else:
            return 0

    def __setitem__(self, cell, value):
        self._coeff[cell] = value

    def __add__(self, other):
        if isinstance(other, Chain):
            otherChain = other
        else:
            otherChain = Chain(other)

        if otherChain == 0:
            return self
        else:
            result = Chain()
            result._coeff.update(self._coeff)

            for cell in otherChain._coeff:
                if cell in result._coeff:
                    result._coeff[cell] += otherChain._coeff[cell]
                else:
                    result._coeff[cell] = otherChain._coeff[cell]

            nullCells = [cell for cell, coeff in result._coeff.items()
                if coeff == 0]
            for cell in nullCells:
                del result._coeff[cell]
            return result

    def __sub__(self, other):
        if isinstance(other, Chain):
            otherChain = other
        else:
            otherChain = Chain(other)

        if otherChain == 0:
            return self
        else:
            result = Chain()
            result._coeff.update(self._coeff)
            for cell in otherChain._coeff:
                if cell in result._coeff:
                    result._coeff[cell] -= otherChain._coeff[cell]
                else:
                    result._coeff[cell] = - otherChain._coeff[cell]

            nullCells = [cell for cell, coeff in result._coeff.items()
                if coeff == 0]
            for cell in nullCells:
                del result._coeff[cell]
            return result

    def __mul__(self, coeff):
        if isinstance(coeff, int):
            if self == 0:
                return self
            else:
                result = Chain()
                if coeff == 0:
                    return result
                else:
                    result._coeff.update(self._coeff)
                    for cell in result._coeff:
                        result._coeff[cell] *= coeff
                    return result
        else:
            raise NotImplemented('Only integers and chains can be multiplied.')

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        result = Chain()
        for cell, coeff in self._coeff.items():
            result._coeff[cell] = - coeff
        return result

    def __str__(self):
        if self == 0:
            return '0'
        else:
            data = []
            for cell in self._coeff.keys():
                coeff = self._coeff[cell]
                sign = '+' if coeff >= 0 else '-'
                data.append((sign, coeff, cell))
            terms = ['{} {}{}'.format(
                sign,
                '' if abs(coeff) == 1 else '{} · '.format(abs(coeff)), cell)
                for sign, coeff, cell in data]
            result = ' '.join(terms)
            if result[0] == '+':
                return result[1:]
            else:
                return result

    def __len__(self):
        return len(self._coeff)

    def __delitem__(self, cell):
        if cell in self._coeff:
            del self._coeff[cell]

    @property
    def dim(self):
        if self._coeff:
            return next(iter(self._coeff)).dim

    @property
    def support(self):
        return (cell for cell in self._coeff)

    @property
    def differential(self):
        result = Chain()
        for cell in self._coeff:
            coeff = self._coeff[cell]
            diff = cell.differential
            result += coeff * diff
        return result

    def __eq__(self, other):
        if isinstance(other, int):
            return other == 0 and not self._coeff
        elif isinstance(other, Chain):
            return self._coeff == other._coeff
        else:
            return False

    def __ne__(self, other):
        return not (self == other)

    def dot(self, other):
        if isinstance(other, Chain):
            return sum(self[c[0]] * other[c[1]] for c in
                product(self.support, other.support))
        else:
            raise NotImplemented('<·, ·> is only implemented for chains.')

    @property
    def coeffs(self):
        return self._coeff.values()

    def draw(self, axes, **kwargs):
        for cell in self.support:
            try:
                cell.draw(axes, **kwargs)
            except AttributeError:
                print('The cells supporting this chain cannot be drawn.')

    def findbest(self):
        len_ = len(self)
        if len_ == 0:
            return None
        elif len_ == 1:
            return tuple(self._coeff.keys())[0]
        else:
            candidates = list(self._coeff.items())
            return sorted(candidates, key=lambda x: abs(x[1]))[0]


@total_ordering
class ZModule(object):
    _base = ()

    def __init__(self, base=()):
        self._base = base

    def __eq__(self, other):
        if isinstance(other, int):
            return (other == 0) and self._base == ()
        elif isinstance(other, ZModule):
            return self._base == other._base

    def __lt__(self, other):
        if self.dim < other.dim:
            return other.base[:self.dim] == self.base
        else:
            return False

    def __str__(self):
        if self._base:
            return 'Z[{}]'.format(', '.join(map(str, self._base)))
        else:
            return '0'

    def __call__(self, chain):
        if isinstance(chain, np.ndarray):
            result = Chain()
            non_zero = np.nonzero(chain)[0]
            if np.any(non_zero):
                for i in np.nditer(np.nonzero(chain)[0]):
                    result = result + Chain(self._base[i]) * int(chain[i, 0])
            return result

        if not isinstance(chain, Chain):
            chain = chain.chain

        result = np.zeros((self.dim, 1), int)
        for cell in chain.support:
            result[self._base.index(cell)] = chain[cell]
        return result

    def __contains__(self, chain):
        if not isinstance(chain, Chain):
            chain = chain.chain

        result = True
        for cell in chain.support:
            result = result and cell in self._base
            if not result:
                break
        return result

    @property
    def base(self):
        if self._base:
            return self._base
        else:
            return (0,)

    @property
    def dim(self):
        d = len(self._base)
        if d == 0:
            return 1
        else:
            return d


class Morphism(object):
    _src = ZModule()
    _dst = ZModule()
    _matrix = np.zeros((1, 1), dtype=int)

    def __init__(self, src=ZModule(), dst=ZModule(), matrix=None):
        self._src = src
        self._dst = dst
        self._matrix = np.zeros((self._dst.dim, self._src.dim), int)
        if not matrix is None:
            self._matrix = matrix

    def __str__(self):
        return 'Morhism from \n {} \n to \n {} \n with matrix \n {}'.format(
            self._src, self._dst, self.matrix)

    def __call__(self, chain):
        if not isinstance(chain, Chain):
            chain = chain.chain

        #product = self._matrix.dot(self._src(chain))
        #result = Chain()
        #for i, v in np.ndenumerate(product):
            #result += v * self._dst_base[i[0]].chain
        #return result
        return self._src(self._matrix.dot(self._src(chain)))

    def __setitem__(self, index, value):
        if isinstance(index, int):
            idx = index
        if isinstance(index, tuple) and isinstance(value, int):
            self._matrix[index] = value
            return
        else:
            idx = self._src.base.index(index)

        if isinstance(value, Chain):
            self._matrix[:, idx] = self._dst(value)[:, 0]
        elif isinstance(value, int):
            self._matrix[:, idx] = value
        elif isinstance(value, np.ndarray):
            self._matrix[:, idx] = value[:, 0]

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._matrix[:, index]
        elif isinstance(index, tuple):
            return self._matrix[index]
        else:
            idx = self._src.base.index(index)
            return self._matrix[:, idx]

    def __eq__(self, other):
        return (
            self._src == other._src and
            self._dst == other._dst and
            np.all(self._matrix == other._matrix))

    def __add__(self, other):
        return Morphism(self._src, self._dst, self._matrix + other._matrix)

    def __sub__(self, other):
        return Morphism(self._src, self._dst, self._matrix - other._matrix)

    def __neg__(self):
        return Morphism(self._src, self._dst, - self._matrix)

    def __mul__(self, other):
        return Morphism(other._src, self._dst, self.matrix.dot(other.matrix))

    @property
    def matrix(self):
        return self._matrix

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst

    def del_src_base_element(self, element):
        if element in self._src._base:
            del_col = self._src._base.index(element)
            self._src._base = (self._src._base[:del_col] +
                self._src._base[del_col + 1:])
            self._matrix = np.delete(self._matrix, [del_col], 1)
        else:
            raise IndexError('{} is not in src''s base'.format(element))

    def del_dst_base_element(self, element):
        if element in self._dst._base:
            del_row = self._dst._base.index(element)
            self._dst._base = (self._dst._base[:del_row] +
                self._dst._base[del_row + 1:])
            self._matrix = np.delete(self._matrix, del_row, 0)
        else:
            raise IndexError('{} is not in dst''s base'.format(element))

    def add_src_base_element(self, element):
        if not self._src._base:
            self._src._base = (element,)
            self._matrix = np.zeros((self._dst.dim, 1), int)
        else:
            if element not in self._src._base:
                self._src._base = self._src._base + (element,)
                new_col = np.zeros((self._matrix.shape[0], 1), int)
                self._matrix = np.append(self._matrix, new_col, 1)

    def add_dst_base_element(self, element):
        if not self._dst._base:
            self._dst._base = (element,)
            self._matrix = np.zeros((1, self._src.dim), int)
        else:
            if element not in self._dst._base:
                self._dst._base = self._dst._base + (element,)
                new_row = np.zeros((1, self._matrix.shape[1]), int)
                self._matrix = np.append(self._matrix, new_row, 0)

    def restriction(self, new_src):
        if not isinstance(new_src, ZModule):
            new_src = ZModule(new_src)
        if not new_src <= self.src:
            raise SyntaxError('new_src must be an ordered sub-base of ' +
                'self.base.')
        else:
            new_matrix = self.matrix[:, :new_src.dim].copy()
            return Morphism(new_src, self.dst, new_matrix)

    @classmethod
    def identity(src, dst=None):
        if dst is None:
            dst = src
        I = np.eye(src.dim, dst.dim, dtype=int)
        return Morphism(src, dst, I)


@total_ordering
class ChainComplex(object):
    _dim = 0
    _modules = {}
    _differential = {}
    _D = {}
    _A = {}
    _G = {}
    # self.D[q] is SNF of self.d[q]
    # self.A[q] == np.linalg.inv(self.G[q])
    # self.G[q - 1].dot(self.D[q]).dot(self.A[q]) == self.d[q]
    # self.A[q - 1].dot(self.d[q]).dot(self.G[q]) == self.D[q]
    _computedSNF = False

    def __init__(self, modules={}, differential={}):
        self._modules.update(modules)
        self._dim = max(self._modules.keys())

        self._differential = differential
        self._D = {}
        self._A = {}
        self._G = {}
        self._computedSNF = False

    def __getitem__(self, index):
        if index in self._modules:
            return self._modules[index]
        else:
            return ZModule()

    def __eq__(self, other):
        result = self.dim == other.dim
        if not result:
            return False

        for q in range(self.dim + 1):
            result = result and self[q] == other[q]
            if not result:
                return False

        for q in range(self.dim + 1):
            result = result and np.all(self.d[q] == other.d[q])
            if not result:
                return False

        return result

    def __lt__(self, other):
        result = True
        for q in range(max(self.dim, other.dim) + 1):
            result = result and self[q] < other[q]
            if not result:
                return False
        return result

    @itemproperty
    def d(self, q):
        if q < 0 or q > self.dim:
            return Morphism()
        else:
            return self._differential[q]

    @property
    def dimensions(self):
        return range(self._dim + 1)

    @property
    def dim(self):
        return self._dim

    @itemproperty
    def D(self, q):
        if q in self._D:
            return self._D[q]
        else:
            return np.zeros((0, 0), int)

    @itemproperty
    def A(self, q):
        if q in self._A:
            return self._A[q]
        else:
            return np.zeros((0, 0), int)

    @itemproperty
    def G(self, q):
        if q in self._G:
            return self._G[q]
        else:
            return np.zeros((0, 0), int)

    def computeSNF(self):
        if not self._computedSNF:
            MSNF = CHomP.MatrixSNF()
            for q in range(self.dim + 1):
                MSNF.matrices[q] = CHomP.MMatrix(self.d[q].matrix)
            MSNF.computeSNF()
            for q in range(self.dim + 1):
                self._D[q] = MSNF.matrices[q]._data
                self._A[q] = MSNF.chgBasisA[q]._data
                self._G[q] = MSNF.chgBasisG[q]._data
            self._computedSNF = True


class ChainMap(object):
    _src_complex = None
    _dst_complex = None
    _degree = 0
    _morphisms = {}

    def __init__(self, src_complex, dst_complex=None, degree=0, morphisms={}):
        self._src_complex = src_complex
        if dst_complex is None:
            self._dst_complex = src_complex
        else:
            self._dst_complex = dst_complex
        self._degree = degree
        dim = max(self._src_complex.dim, self._dst_complex.dim)
        for q in range(dim + 1):
            if q in morphisms:
                self._morphisms[q] = morphisms[q]
            else:
                self._morphisms[q] = Morphism(
                    self._src_complex[q], self._dst_complex[q + self._degree])

    def __call__(self, chain):
        dim = chain.dim
        if not dim is None:
            return self._morphisms[dim](chain)

    def __add__(self, other):
        if (self.src == other.src and self.dst == other.dst and
            self.degree == other.degree):
            new_morphisms = {}
            new_morphisms.update(self.morphisms)
            for q in other.morphism:
                if q in new_morphisms:
                    new_morphisms[q] += other[q]
                else:
                    new_morphisms[q] = other[q]
            return ChainMap(self.src, self.dst, new_morphisms)
        else:
            raise SyntaxError('Incompatible chain maps.')

    def __getitem__(self, index):
        if index in self._morphisms:
            return self._morphisms[q]

    def __setitem__(self, index, value):
        self._morphisms[index] = value

    @property
    def src(self):
        return self._src_complex

    @property
    def dst(self):
        return self._dst_complex

    @property
    def degree(self):
        return self._degree

    def __getitem__(self, index):
        if index in self._morphisms:
            return self._morphisms[index]
        else:
            return Morphism()

    @itemproperty
    def matrix(self, q):
        return self[q].matrix

    @matrix.setter
    def matrix(self, q, value):
        self[q].matrix = value

    def __str__(self):
        return '<Chain map: \n{}>'.format('\n'.join(
            ('{} --> {}'.format(q, str(m)) for (q, m) in
            self._morphisms.items())))

class ChainContraction(object):
    _src = None
    _dst = None
    _projection = None
    _inclusion = None
    _integral = None

    def __init__(self, src, dst, projection=None, inclusion=None,
        integral=None):
        self._src = src
        self._dst = dst

        if not projection is None:
            self._projection = projection
        else:
            self._projection = ChainMap(self._src, self._dst)

        if not inclusion is None:
            self._inclusion = inclusion
        else:
            self._inclusion = ChainMap(self._dst, self._src)

        if not integral is None:
            self._integral = integral
        else:
            self._integral = ChainMap(self._src, self._dst, 1)

    def __mul__(self, other):
        if other.dst <= self.src:
            return ChainContraction(
                other.src,
                self.dst,
                self.projection * other.projection,
                other.inclusion * self.inclusion,
                other.integral +
                    other.inclusion * self.integral * other.projection)
        else:
            raise NotImplemented('Composition of ChainContractions require ' +
                'that other.dst <= self.src')


if __name__ == '__main__':
    def print_array(array, desc='', full=False):
        print('{}: Shape = {}'.format(desc if desc else 'Array',
            'x'.join(map(str, array.shape))))
        print('-' * 80)
        if not full:
            for idx in zip(*np.where(array != 0)):
                print('{} --> {}'.format(idx, array[idx]))
        else:
            print(array)

    from Examples import KleinBottle
    K = KleinBottle()
    C = K.chain_complex
    C.computeSNF()
    for q in range(3):
        #print_array(C.D[q], 'D[{}]'.format(q))
        #print_array(C.A[q], 'A[{}]'.format(q), True)
        #print_array(C.G[q], 'G[{}]'.format(q), True)
        #print_array(C.A[q].dot(C.G[q]), full=True)
        if q == 0:
            print(C.d[q].matrix.dot(
                np.linalg.inv(C.A[q]).astype(int)) - C.D[q])
        if q > 0:
            print(C.A[q - 1].dot(C.d[q].matrix).dot(
                np.linalg.inv(C.A[q]).astype(int)) - C.D[q])
