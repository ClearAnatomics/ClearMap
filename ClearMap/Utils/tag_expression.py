# -*- coding: utf-8 -*-
"""
tag_expression
==============

Module providing routines to check and convert between tag expressions.
This simplfies the handling of list of files from using regular expressions
to tag expression. 

A tag is a simple placeholder for a string or number in a file name.
An expression is a filename with any number of tags.

A tag has the format <Name,Type,Width>. 
  * Name : str that specifies the tag name
  * Type : 'I' or 'S' for integer or string, optional and defaults to 'I'
  * Width : int, if given indicates a fixed width and missing digits or chars
    are replaced by trailing zeros or spaces.

The module also provides functions to infer the tag expressions from a list of
file names via the function :func:`~ClearMap.Utils.tag_expression.detect`.

Expressions can also be converted to glob or regular expressions.


Example
-------
An example tag expression would be 'file_<X,2>_<Y,3>.npy' and would for example
match the filename 'file_01_013.npy'

>>> import ClearMap.Utils.tag_expression as te
>>> e = te.Expression('file_<X,2>_<Y,3>.npy')
>>> e.tag_names()
e.tag_names()

>>> e.glob_pattern()
'file_[0-9][0-9]_[0-9][0-9].npy'

>>> e.indices('file_01_013.npy')
[1, 13]

>>> e.re()
'file_(?P<X>\\d{2})_(?P<Y>\\d{3})\\.npy'

>>> e.string({'X':1, 'Y':10})
'file_01_010.npy'

See also
--------
:mod:`~ClearMap.Utils.RegularExpression`
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


import copy
import glob

import re
from pathlib import Path

import natsort

TAG_START = '<'
TAG_END = '>'
TAG_SEPARATOR = ','
TAG_INT = 'I'
TAG_STR = 'S'


def ttype_to_dtype(ttype):
    if ttype == TAG_INT or ttype is None:
        return int
    elif ttype == TAG_STR:
        return str
    else:
        raise ValueError(f'The specified tag type {ttype} is not valid!')


def default_tag_name(index=None):
    tag = 'Tag'
    if index is not None:
        tag += f'{index}'
    return tag


class Tag:
    """
    A tag is a simple placeholder for a string or number in a file name.

    A tag has the format <Name,Type,Width>.
    * Name : str that specifies the tag name
    * Type : 'I' or 'S' for integer or string, optional and defaults to 'I'
    * Width : int, if given indicates a fixed width and missing digits or chars
    are replaced by trailing zeros or spaces.

    Attributes
    ----------
    name: str
        The name of the tag
    ttype: str
        The type of the tag, either 'I' for integer or 'S' for string.
        defined from the constants TAG_INT and TAG_STR
    width: int
        The width of the tag, i.e. the number of digits or characters
    """
    def __init__(self, tag=None, name=None, ttype=TAG_INT, width=None, reference=None, trange=None):
        if tag is not None:
            self.parse(tag)
        else:
            self.name = name
            self.ttype = ttype
            self.width = width
            self.reference = reference
            self.trange = trange

    def label(self, index=None):
        if self.name is not None:
            return self.name
        else:
            return default_tag_name(index=index)

    def dtype(self):
        """
        Return the data type (python) of the tag.

        Returns
        -------
        type
            The data type of the tag.
        """
        return ttype_to_dtype(self.ttype)

    def tag(self):
        t = TAG_START
        if self.name is not None:
            t += self.name + TAG_SEPARATOR
        if self.ttype is not None:
            t += self.ttype + TAG_SEPARATOR
        if self.width is not None:
            t += str(self.width) + TAG_SEPARATOR
        if len(t) > len(TAG_START):
            t = t[:-len(TAG_SEPARATOR)]
        t += TAG_END
        return t

    def glob_pattern(self):
        e = ''
        if self.width is not None:
            if self.ttype == TAG_INT or self.ttype is None:
                s = '[0-9]'
            elif self.ttype == TAG_STR:
                s = '?'
            e += ''.join([s] * self.width)
        else:
            e += '*'
        return e

    def re(self, name=None):
        if name is None:
            name = self.name
        e = ''
        if self.reference:
            if name is not None:
                e += f'(?P={name})'
            else:
                raise ValueError('Name needs to be given for a referenced tag!')
        else:
            if name is not None:
                e += f'(?P<{name}>'
            else:
                e += '('
            if self.ttype == TAG_INT or self.ttype is None:
                e += '\d'
            else:  # self.ttype == TAG_STR:
                e += '.'
            if self.width is not None:
                e += f'{{{self.width}}}'
            else:
                e += '*?'
            e += ')'
        return e

    def string(self, value=None):
        if value is None:
            return self.tag()
        elif value == '?':
            return '?' * self.width

        if self.width is None:
            if self.ttype == TAG_INT or self.ttype is None:
                frmt = '%d'
            else:
                frmt = '%s'
        else:
            if self.ttype == TAG_INT or self.ttype is None:
                frmt = f'%0{self.width}d'
            else:
                frmt = f'%{self.width}s'
        return frmt % value

    def string_from_index(self, index=None):
        if index is not None:
            if self.trange is not None:
                value = self.trange[index]
            else:
                value = index
        else:
            value = None
        return self.string(value=value)

    def value(self, string):
        return self.dtype()(string)

    def index(self, value):
        if self.trange is None:
            if isinstance(value, str):
                raise IndexError(f'No range to determine index for tag {self} and value {value}!')
            else:
                return value
        else:
            for i, r in enumerate(self.trange):
                if value == r:
                    return i
            raise IndexError(f'Value {value} not in tag range {self.trange}!')

    def parse(self, tag):
        """
        Parse a tag from a string. The string should be in the format of the tag.
        i.e. f'{TAG_START}{Name}{TAG_SEPARATOR}{Type}{TAG_SEPARATOR}{Width}{TAG_END}'
        tag_type and tag_width are optional but the tag name is required.
        If any of the optional values are not given then the tag pattern is shortened
        accordingly.
        If tag_type is not given it defaults to 'I' for integer. If tag_width is not
        given the tag is considered to have variable width.
        This function will initialize the tag with the parsed values.

        Parameters
        ----------
        tag: str
            The tag string to parse.

        Returns
        -------
        None
        """
        if len(tag) < len(TAG_START) + len(TAG_END):
            raise ValueError(f'The string {tag} is not a valid tag!')
        if not tag.startswith(TAG_START):
            raise ValueError(f'Expecting the tag to start with {TAG_START} found {tag[:len(TAG_START)]}!')
        if not tag.endswith(TAG_END):
            raise ValueError(f'Expecting the tag to end with {TAG_END} found {tag[-len(TAG_END):]}!')

        tag = tag[len(TAG_START):-len(TAG_END)]
        if not tag:
            self.__init__()
            return

        tag_attributes = {
            'name': None,
            'ttype': None,
            'width': None
        }

        values = tag.split(TAG_SEPARATOR)
        if len(values) > 3:
            raise ValueError(f'Found {len(values)} > 3 tag attributes !')
        elif len(values) == 3:
            tag_attributes['name'] = values[0]
            tag_attributes['ttype'] = values[1]
            tag_attributes['width'] = int(values[2])
        elif len(values) == 2:
            tag_attributes['name'] = values[0]
            tmp = values[1]
            try:
                tag_attributes['width'] = int(tmp)
            except ValueError:
                tag_attributes['ttype'] = tmp

        for k, v in tag_attributes.items():
            if v is not None and not v:
                raise ValueError(f'Empty value for tag attribute {k} in {tag}! This is not allowed!')

        self.__init__(**tag_attributes)

    def __str__(self):
        return self.tag()

    def __repr__(self):
        return self.__str__()


class Expression:
    tag_regexp = re.compile(f'{TAG_START}.*?{TAG_END}')
    def __init__(self, pattern=None):
        if isinstance(pattern, Expression):
            self.pattern = copy.copy(pattern.pattern)
            self.tags = copy.copy(pattern.tags)
        elif isinstance(pattern, (str, Path)):
            self.parse(str(pattern))
        else:
            if pattern is None:
                pattern = []  # FIXME: no plural in pattern what list is it? it seems tags
            self.pattern = pattern
            self.tags = [p for p in pattern if isinstance(p, Tag) and not p.reference]

    def tag(self):  # FIXME: confusing name. Does this "tag" the expression?
        e = ''
        for p in self.pattern:
            if isinstance(p, Tag):
                e += p.tag()
            else:
                e += p
        return e

    def tag_max(self, tag_name):
        """
        Return the maximum value (i.e. range[-1]) of a tag in the expression.

        Parameters
        ----------
        tag_name: str
            The name of the tag

        Returns
        -------
        int
            The maximum value of the tag
        """
        return self.tag_range(tag_name)[1]

    def tag_min(self, tag_name):
        """
        Return the minimum value (i.e. range[0]) of a tag in the expression.

        Parameters
        ----------
        tag_name: str
            The name of the tag

        Returns
        -------
        int
            The minimum value of the tag
        """
        return self.tag_range(tag_name)[0]

    def tag_range(self, tag_name):
        """
        Return the range of values of a tag in the expression.

        Parameters
        ----------
        tag_name: str
            The name of the tag

        Returns
        -------
        tuple[int]
            The minimum and maximum value of the tag
        """
        file_list = self.glob(sort=True)
        values = [self.values(f)[tag_name] for f in file_list]
        return min(values), max(values)

    def re(self):
        """
        Convert the expression to a regular expression string.

        Examples
        --------
        >>> e.re()
        'file_(?P<X>\\d{2})_(?P<Y>\\d{3})\\.npy'

        Returns
        -------
        str
            The regular expression string.
        """
        e = ''
        n_tag = 0
        for p in self.pattern:
            if isinstance(p, Tag):
                e += p.re(name=p.label(n_tag))
                n_tag += 1
            else:
                e += re.escape(p)
        return e

    def glob_pattern(self, values=None):
        """
        Convert the expression to a glob pattern string.
        >>> exp = Expression('file_<X,2>_<Y,3>.npy')

        >>> exp.glob_pattern()
        'file_[0-9][0-9]_[0-9][0-9].npy'

        Parameters
        ----------
        values: dict
            A dictionary with values for the tags

        Returns
        -------
        str
            The glob pattern string
        """
        e = ''
        n_tag = 0
        for p in self.pattern:
            if isinstance(p, Tag):
                if values is None:
                    e += p.glob_pattern()
                else:
                    lab = p.label(n_tag)
                    if lab in values.keys():
                        e += escape_glob(p.string(value=values[lab]))
                    else:
                        e += p.glob_pattern()
                n_tag += 1
            else:
                e += escape_glob(p)
        return e

    def glob(self, sort=False):
        """
        Execute the glob pattern.

        Parameters
        ----------
        sort: bool
            If True, the paths are sorted using natsort.

        Returns
        -------
        list[str]
            A list of paths that match the glob pattern.
        """
        paths = glob.glob(self.glob_pattern())
        if sort:
            paths = natsort.natsorted(paths)
        return paths

    def string(self, values=None):
        """
        Convert a value to a string according to the tag specification.

        Examples
        --------
        >>> exp = Expression('file_<X,2>_<Y,3>.npy')
        >>> exp.string({'X':1, 'Y':10})
        'file_01_010.npy'

        Parameters
        ----------
        values: dict
            A dictionary with values for the tags. Each key is the tag name.

        Returns
        -------
        str
            The string representation of the value.
        """
        e = ''
        n_tag = 0
        for p in self.pattern:
            if isinstance(p, Tag):
                if values is not None:
                    v = values.get(p.label(n_tag), None)  # FIXME: do not replace if not in values (use kwarg)
                else:
                    v = None
                e += p.string(value=v)
                n_tag += 1
            else:
                e += p
        return e

    def char_index(self, tag_name, with_markups=False):
        """
        Return the character index range of the tag in the expression

        Parameters
        ----------
        tag_name: str
            The name of the tag
        with_markups: bool
            If True, the markup characters (<,width,type>) are included in the index range

        Returns
        -------
        tuple[int]
            The character index range of the tag in the expression
        """
        n_tag = 0
        start = 0  # FIXME: exclude markups
        for p in self.pattern:
            if isinstance(p, Tag):
                if p.label(n_tag) == tag_name:
                    if with_markups:
                        return start, start + len(p.string())
                    else:
                        return start, start + p.width
                else:
                    if with_markups:
                        start += len(p.string())
                    else:
                        start += p.width
                    n_tag += 1
            else:
                start += len(p)

    def values(self, string):
        """
        Extract the values from a string according to the tag specification.

        Parameters
        ----------
        string: str
            The expression string to extract the values from.

        Returns
        -------
        dict
            A dictionary with the values for the tags. Keys are the tag names.
        """
        tags = self.tags
        search = re.compile(self.re()).search
        match = search(string)
        if match is None:
            return {}
        else:
            d = match.groupdict()
            for k, v in d.items():
                for i, t in enumerate(tags):
                    if k == t.label(i):
                        v = t.dtype()(v)
                        d[k] = v
                        break
            return d

    def string_from_index(self, indices):
        """
        Convert a dictionary or list of tag indices to a string according to the tag specification.

        Examples
        --------
        >>> exp = Expression('file_<X,2>_<Y,3>.npy')
        >>> exp.string_from_index([1, 13])
        'file_01_013.npy'

        Parameters
        ----------
        indices: list[int] | dict
            A list of indices for the tags. If a dictionary is given, the keys are the tag names.

        Returns
        -------

        """
        if isinstance(indices, int):
            indices = [indices]
        if not isinstance(indices, dict):
            indices = {name: idx for idx, name in zip(indices, self.tag_names())}
        out = ''
        n_tags = 0
        for p in self.pattern:
            if isinstance(p, Tag):
                out += p.string_from_index(index=indices[p.label(n_tags)])
                n_tags += 1
            else:
                out += p
        return out

    def indices(self, string):
        """
        Infer the indices from a string.

        Examples
        --------
        >>> e.indices('file_01_013.npy')
        [1, 13]

        Parameters
        ----------
        string: str
            The path string to infer the indices from

        Returns
        -------
        list[int]
            The indices of the tags
        """
        tags = self.tags
        search = re.compile(self.re()).search
        match = search(string)
        if match is None:
            raise ValueError('Cannot infer indices from string!')
        else:
            d = match.groupdict()
            for k, v in d.items():
                for i, t in enumerate(tags):
                    if k == t.label(i):
                        v = t.index(t.dtype()(v))
                        d[k] = v
                        break
        indices = [d[t.label(i)] for t in tags]
        return indices

    def tag_names(self):
        return [t.label(i) for i, t in enumerate(self.tags)]

    def n_tags(self):
        return len(self.tags)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.tags[i]
        else:
            for j, n in enumerate(self.tag_names()):
                if n == i:
                    return self.tags[j]
            raise IndexError(f'No tag with name "{i}"!')

    def parse(self, expression):
        pattern = []
        tags = []
        start = 0
        for match in Expression.tag_regexp.finditer(expression):
            if match.start() > start:
                pattern.append(expression[start:match.start()])
            tag = Tag(tag=match.group())
            pattern.append(tag)
            tags.append(tag)
            start = match.end()
        if start < len(expression):
            pattern.append(expression[start:])

        # check for references
        for i, t in enumerate(tags):
            if t.name is not None and t.reference is not True:
                refs = [r for r in tags[i+1:] if r.name == t.name]
                for r in refs:
                    r.reference = True
                    if r.ttype is None:
                        r.ttype = t.ttype
                    elif r.ttype != t.ttype:
                        raise ValueError(f'The reference {r} has not the same type as the tag {t}!')
                    if r.width is None:
                        r.width = t.width

        self.pattern = pattern
        self.tags = [q for q in pattern if isinstance(q, Tag) and not q.reference]

    def detect(self, strings, names=None, max_check=None, with_trange=False):
        if not isinstance(strings, list):
            strings = [strings]

        if not all([len(s) == len(strings[0]) for s in strings]):
            raise ValueError('Cannot infer tag expression from strings of different length!')

        if max_check is None:
            max_check = len(strings)

        if names is None:
            names = []

        # detect differences in filenames
        s0 = strings[0]
        tags = []
        tag_start = -1
        tag_end = -1
        for i, c in enumerate(s0):
            same = True
            for s in strings[1:]:
                if s[i] != c:
                    if i == tag_end:
                        tag_end += 1
                    else:
                        tag_start = i
                        tag_end = i + 1
                    break
            if same and tag_start != -1:
                tags.append((tag_start, tag_end))
                tag_start = -1
                tag_end = -1

        # detect trailing zeros
        tags_full = []
        for t in tags:
            s, e = t
            while s > 0 and s0[s-1] == '0':
                s -= 1
            tags_full.append((s, e))
        tags = tags_full

        # infer pattern
        pattern = []
        p = 0
        for i, t in enumerate(tags):
            s, e = t
            if s-p > 0:
                pattern.append(s0[p:s])
                p = e

            ttype = TAG_INT
            values = []
            for s in strings[:max_check]:
                v = s[t[0]:t[1]]
                try:
                    v = int(v)
                except ValueError:
                    ttype = TAG_STR
                    if not with_trange:
                        break
                values.append(v)
            if len(names) > 0:
                name = names.pop(0)
            else:
                name = default_tag_name(i)
            trange = values if with_trange else None
            pattern.append(Tag(name=name, ttype=ttype, width=(t[1] - t[0]), trange=trange))
        if p < len(s0) > 0:
            pattern.append(s0[p:])

        self.pattern = pattern
        self.tags = [q for q in pattern if isinstance(q, Tag) and not q.reference]

    def __str__(self):
        return f'{self.tag()}'

    def __repr__(self):
        return f'tag_expression({self})'


def parse(expression):
    e = Expression()
    e.parse(expression=expression)
    return e


def detect(strings, names=None, max_check=None, with_trange=False):
    e = Expression()
    e.detect(strings=strings, names=names, max_check=max_check, with_trange=with_trange)
    return e


def escape_glob(string):
    e = ''
    for c in string:
        if c in '?[]':
            e += f'[{c}]'
        else:
            e += c
    return e


def _test():
    """Tests"""
    import ClearMap.Utils.tag_expression as te
    # reload(te)

    # values and strings
    t = te.parse('/test/test<X,I,4>_<Y,I,3>_<X>.tif')

    s = '/test/test0010_013_0010.tif'
    v = t.values(s)
    s2 = t.string(v)
    s == s2

    t.string({'X': 111})

    # indices
    t.indices(s)
    t = te.parse('/test/test<X,I,4>_<Y,S>.tif')
    t['X']
    t['Y'].trange = list('abcd')
    t.string_from_index([0, 2])

    # glob
    import ClearMap.Tests.Files as tf

    s = tf.io.join(tf.tif_sequence, 'sequence<I,4>.tif')
    t = te.parse(s)
    f = t.glob()

    # detection
    te.detect(f, names=['X'])

    t = te.detect(f, names=['X'], with_trange=True)
    t['X'].trange
  
