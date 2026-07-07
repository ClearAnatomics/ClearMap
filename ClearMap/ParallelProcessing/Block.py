# -*- coding: utf-8 -*-
"""
Block
=====

The Block module provides a :mod:`~ClearMap.IO.Source` class used in parallel 
processing of very large arrays in 
:mod:`ClearMap.ParallelProcessing.BlockProcessing`.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'https://idisco.info'
__download__  = 'https://www.github.com/ChristophKirst/ClearMap2'

from typing import List

import numpy as np

import ClearMap.IO.IO as io
import ClearMap.IO.Slice as slc


###############################################################################
### Block source
###############################################################################

class Block(slc.Slice):
    """Block source

    A Block is a Slice with a sub-slice indicating the valid region to use when
    combining the results after block processing.

    Each block has an index tuple that specifies its position in the grid of
    blocks in which the source was split into.

    Each block also can carry a reference to its neighbouring blocks.

    See also
    --------
    :mod:`ClearMap.ParallelProcessing.BlockProcessing`
    """

    def __init__(self, source=None, slicing=None, valid=None, valid_slicing=None,
                 offsets=None, index=None, iteration=None, blocks_shape=None, neighbours=None, name=None):
        """Constructor"""
        super(Block, self).__init__(source=source, slicing=slicing, name=name)

        if valid is None:
            if valid_slicing is None:
                if offsets is None:
                    valid_slicing = slice(None)
                else:
                    valid_slicing = _offsets_to_slicing(offsets)
            valid = slc.Slice(source=self, slicing=valid_slicing)
        if not isinstance(valid, slc.Slice):
            raise ValueError('The valid slice of the block is not specified correctly!')

        self._valid = valid
        self._index = index
        self._iteration = iteration
        self._blocks_shape = blocks_shape
        self._neighbours = neighbours

    @property
    def name(self):
        """The name of this source.

        Returns
        -------
        name : str
            Name of this source.
        """
        return 'Block-' + self.source.name

    @property
    def valid(self):
        """
        Return the slice of the valid region of this block.

        Returns
        -------
        valid : Slice
            The valid slice of this block.
        """
        return self._valid

    @property
    def index(self):
        """
        Return the grid index of the block.

        Returns
        -------
        index : tuple of ints
            The multi index of this block in a grid of blocks.
        """
        return self._index

    @property
    def blocks_shape(self):
        """
        Return the shape of the block grid this block belongs too.

        Returns
        -------
        shape : tuple of ints
            The shape of the grid of blocks this block is part of.
        """
        return self._blocks_shape

    @property
    def iteration(self):
        """
        Return the index of this block in the list of all blocks to process.

        Returns
        -------
        index : tuple of ints
            The multi index of this block in a grid of blocks.
        """

        if self._iteration is None:
            if self.index is not None and self.blocks_shape is not None:
                return np.ravel_multi_index(self.index, self.blocks_shape)

        return self._iteration

    @iteration.setter
    def iteration(self, iteration):
        self._iteration = iteration

    @property
    def n_iterations(self):
        """Returns the number of blocks in the grid to which this block belongs.

        Returns
        -------
        n_iterations : int or None
            The number of blocks in the block grid.
        """
        if self.blocks_shape is not None:
            return np.prod(self.blocks_shape)
        else:
            return None  # just explicit for IDE

    @property
    def neighbours(self):
        """
        Returns the neighbours of this block.

        Returns
        -------
        neighbours : dict or None
            The neighbours of this block in the form {index : block,...} or None
        """
        return self._neighbours

    @neighbours.setter
    def neighbours(self, neighbours):
        self._neighbours = neighbours

    def as_virtual(self):
        return Block(source=self.source.as_virtual(), slicing=self.slicing, valid_slicing=self.valid.slicing)


    def as_real(self):
       return Block(source=self.source.as_real(), slicing=self.slicing, valid_slicing=self.valid.slicing)

    def as_memory_block(self):
        source = io.as_source(self.as_memory())
        return Block(source=source, slicing=slice(None), valid_slicing=self.valid.slicing,
                     index=self.index, neighbours=self.neighbours)

    def iteration_info(self):
        """
        Return info string about the iteration of this block in the gird of blocks.

        Returns
        -------
        info : str
            Info string.
        """
        info = ''
        iteration = self.iteration
        if iteration is not None:
            info += f'{self.iteration:d}/{self.n_iterations:d}'

        index = self.index
        if index is not None:
            if info != '':
                info += '<'
            info += f'{index!r}'
            blocks_shape = self.blocks_shape
            if blocks_shape is not None:
                info += f'/{blocks_shape!r}'
            if info != '':
                info += '>'

        return info

    def info(self, short: bool = True):
        """
        Return info string about this block within the grid of blocks.

        Returns
        -------
        info : str
            Info string.
        """
        info = self.iteration_info()
        if info != '':
            info += ' '
        if short:
            slc_fmt = slc._slicing_to_str(self.slicing, self.ndim)
            info += f'{self.shape!r}@{self.source.shape!r}[{slc_fmt}]'
        else:
            info += f'{self!r} @ {self.source!r}[{self.slicing!r}]'
        return info


###############################################################################
### Helpers
###############################################################################

def _offsets_to_slicing(offsets, ndim: int | None = None, shape: List[int] | None = None):
    """Parses offsets into standard form ((low0, high0),(low1, high1),...)."""
    if shape is not None:
        ndim = len(shape)

    if not isinstance(offsets, (list, tuple)):
        offsets = [offsets] * (ndim or 1)

    if ndim is not None:
        if len(offsets) != ndim:
            raise ValueError(f'Offset dimension {len(offsets):d} does not match data dimensions {ndim:d}')
    else:
        ndim = len(offsets)  # FIXME: unused

    new_offsets = []
    for d, o in enumerate(offsets):
        if not isinstance(o, (list, tuple)):
            o = [o]
        o = list(o)
        if len(o) == 1:
            o = (o[0], o[0])
        if len(o) != 2:
            raise ValueError(f'Offset {o!r} in dimension {d:d} not valid!')
        if o[0] == 0:
            o[0] = None
        if o[1] == 0:
            o[1] = None
        elif isinstance(o[1], int):
            o[1] = -o[1]
        if shape is not None:
            if o[0] is not None:
                if o[0] >= shape[d] or -o[0] > shape[d]:  # REFACTOR: merge w/ below
                    raise ValueError(f'Offset {o!r} out of range {shape[d]:d} in dimension {d:d}!')
            if o[1] is not None:
                if o[1] >= shape[d] or -o[1] > shape[d]:
                    raise ValueError(f'Offset {o!r} out of range {shape[d]:d} in dimension {d:d}!')
        new_offsets.append(o)

    new_offsets = tuple(slice(o[0], o[1]) for o in new_offsets)

    return new_offsets


###############################################################################
### Tests
###############################################################################

def _test():
    import numpy as np   #analysis:ok
    import ClearMap.ParallelProcessing.Block as blk

    import ClearMap.IO.IO as io
    source = io.open_ro(np.asarray(np.random.rand(50,100,200), order='F'))

    block = blk.Block(source=source, index=(1,2,3), blocks_shape=(10,20,30))

    print(block.n_iterations, block.iteration)
    print(block.iteration_info())
