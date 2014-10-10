from abc import ABC, abstractmethod
from functools import total_ordering
from algebra import Chain
from collections import Iterable
import numpy as np


class AbstractCell(ABC):

    @abstractmethod
    def __str__(self):
        pass

    @property
    @abstractmethod
    def dim(self):
        pass

    @property
    @abstractmethod
    def differential(self):
        pass

    @property
    @abstractmethod
    def boundary(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __lt__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass


@total_ordering
class Cell(AbstractCell):
    _differential = Chain()
    _id = ''
    _dim = 0

    def __init__(self, id, dim, differential=None):
        self._id = id
        self._dim = dim
        if differential is None:
            self._differential = Chain()
        else:
            self._differential = differential

    def __str__(self):
        return '<{}>'.format(self._id)

    @property
    def dim(self):
        return self._dim

    @property
    def differential(self):
        return self._differential

    @property
    def boundary(self):
        return self.differential.support

    def __eq__(self, other):
        return (self._id == other._id and self.dim == other.dim and
            self.differential == other.differential)

    def __lt__(self, other):
        if self.dim < other.dim:
            if other.dim - self.dim == 1:
                return self in other.boundary
            else:
                result = False
                for face in other.boundary:
                    result = result or other <= face
                    if result:
                        return True
                return result
        else:
            return False

    def __hash__(self):
        return hash(self._id)


@total_ordering
class Simplex(AbstractCell):
    _vertices = ()

    def __init__(self, vertices):
        L = list(vertices)
        L.sort
        self._vertices = tuple(L)

    def __str__(self):
        return '<{}>'.format(', '.join(str(v) for v in self._vertices))

    @property
    def dim(self):
        return len(self._vertices) - 1

    @property
    def differential(self):
        result = Chain()
        if self.dim > 0:
            for i, s in enumerate(self._vertices):
                face = Simplex(self._vertices[:i] + self._vertices[i + 1:])
                result._coeff[face] = (-1) ** i
        return result

    @property
    def boundary(self):
        faces = []
        for i, s in enumerate(self._vertices):
            faces.append(self._vertices[:i] + self._vertices[i + 1:])
        faces.sort()
        return (Simplex(face) for face in faces)

    def __eq__(self, other):
        return self._vertices == other._vertices

    def __lt__(self, other):
        return set(self._vertices) < set(other._vertices)

    def __hash__(self):
        return hash(self._vertices)

int_types = (int, np.int8, np.int16, np.int32, np.int64, np.uint8,
    np.uint16, np.uint32, np.uint64)


def isCell(iterable):
    '''
        Test whether an iterable is a cell, i.e., it is a list of integers.
    '''
    return (isinstance(iterable, Iterable) and
        all(isinstance(i, int_types) for i in iterable))


def isIntervalList(iterable):
    '''
        Test whether an iterable is an interval list, i.e., it is a list
        of pairs of consecutive integers.
    '''
    return (isinstance(iterable, Iterable) and
        all(isinstance(I, tuple) and 1 <= len(I) <= 2 and
                isinstance(max(I), int) and isinstance(min(I), int_types) and
                0 <= max(I) - min(I) <= 1 for I in iterable))


class CubicalCell(AbstractCell):
    _intervals = None
    _cell = None
    _emb = None
    _dim = None
    _boundary = None
    _differential = None
    _center = None
    _index = None

    def __init__(self, arg=None):
        if isIntervalList(arg):
            self._emb = len(arg)
            self._intervals = tuple((min(I), max(I)) for I in arg)
            self._cell = tuple(I[0] + I[1] for I in self._intervals)
            self._dim = sum(i % 2 for i in self.cell)
            self._boundary = self.boundary
            self._differential = self.differential
            self._center = tuple((min(I) + max(I)) / 2 for I in             
                self._intervals)
            self._index = None
        elif isCell(arg):
            self._emb = len(arg)
            self._cell = arg
            self._intervals = tuple(
                (i // 2, i // 2) if i % 2 == 0 else 
                ((i - 1) // 2, (i + 1) // 2)
                for i in self.cell)
            self._dim = sum(i % 2 for i in self.cell)
            self._boundary = self.boundary
            self._differential = self.differential
            self._center = tuple((min(I) + max(I)) / 2 for I in 
                self._intervals)
            self._index = None
        else:
            raise TypeError('An appropiate argument must be provided.')

    def __str__(self):
        if self._intervals:
            intervalStringList = []
            for interval in self._intervals:
                if min(interval) == max(interval):
                    intervalStringList.append('({})'.format(min(interval)))
                else:
                    intervalStringList.append('({},{})'.format(min(interval),
                        max(interval)))
            return "x".join(intervalStringList)
        else:
            return "()"

    @property
    def dim(self):
        return self._dim

    @property
    def differential(self):
        if self._differential is None:
            bdry = {}
            nonDegeneratedIdxs = tuple(
                p for p, I in enumerate(self._intervals)
                if I[0] != I[1]
            )

            for j, p in enumerate(nonDegeneratedIdxs):
                pre = tuple(
                    I[0] + I[1] for I in self._intervals[:p]
                )
                curr = self._intervals[p]
                post = tuple(
                    I[0] + I[1] for I in self._intervals[p + 1:]
                )
                pos_bdry = pre + (2 * curr[1],) + post
                neg_bdry = pre + (2 * curr[0],) + post

                bdry[pos_bdry] = int((-1) ** j)
                bdry[neg_bdry] = int((-1) ** (j + 1))
            return Chain(bdry)
        else:
            return self._differential

    @property
    def boundary(self):
        return (CubicalCell(cell) for cell in self.differential.support)

    def __eq__(self, other):
        if isCell(other):
            return self._cell == other
        elif isIntervalList(other):
            return self._intervals == other
        elif isinstance(other, CubicalCell):
            return self._cell == other._cell
        else:
            return False

    def __le__(self, other):
        if isCell(other) or isIntervalList(other):
            if len(other) == self._emb:
                return self <= CubicalCell(other)
            else:
                raise TypeError('Uncomparable cubic cells because of its '
                    'different embedding numbers.')
        elif isinstance(other, CubicalCell):
            if self._emb == other._emb:
                return all(set(I1) <= set(I2) for (I1, I2) in
                    zip(self._intervals, other._intervals))
            else:
                raise TypeError('Uncomparable cubic cells because of its '
                    'different embedding numbers.')

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        if isCell(other) or isIntervalList(other):
            if len(other) == self._emb:
                return self <= CubicalCell(other)
            else:
                raise TypeError('Uncomparable cubic cells because of its '
                    'different embedding numbers.')
        elif isinstance(other, CubicalCell):
            if self._emb == other._emb:
                print(list(map(set, self._intervals)))
                print(list(map(set, other._intervals)))
                return all(set(I1) >= set(I2) for (I1, I2) in
                    zip(self._intervals, other._intervals))

    def __gt__(self, other):
        return self >= other and self != other

    def __hash__(self):
        return hash(self.cell)

    @property
    def intervals(self):
        return self._intervals

    @property
    def cell(self):
        return self._cell
    
    def faces(self):
        result = set(())
        if self.dim > 0:
            for face in self.boundary:
                result = result.union(set((face,)))
                if face.dim > 0:
                    result = result.union(face.faces())
        return result
