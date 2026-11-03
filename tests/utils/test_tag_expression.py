"""
The regular expressions that :mod:`ClearMap.Utils.tag_expression` builds from tags, and the source of the module itself:
a backslash in a normal string literal that is not a valid escape (``'\\d'``) is a ``SyntaxWarning`` since Python 3.12
(a ``DeprecationWarning`` before) and will be a ``SyntaxError``, so the module must not contain one.
"""
import re
import warnings
from pathlib import Path

from ClearMap.Utils import tag_expression
from ClearMap.Utils.tag_expression import Tag, TAG_INT, TAG_STR, parse


def test_the_module_compiles_without_invalid_escape_sequences():
    source_path = Path(tag_expression.__file__)

    with warnings.catch_warnings():
        warnings.simplefilter('error', category=SyntaxWarning)
        warnings.simplefilter('error', category=DeprecationWarning)
        compile(source_path.read_text(encoding='utf-8'), str(source_path), 'exec')


def test_an_integer_tag_matches_digits_in_the_regular_expression():
    assert Tag(name='X', ttype=TAG_INT, width=2).re() == r'(?P<X>\d{2})'
    assert Tag(name='X', ttype=TAG_INT).re() == r'(?P<X>\d*?)'


def test_a_string_tag_matches_any_character_in_the_regular_expression():
    assert Tag(name='n', ttype=TAG_STR, width=3).re() == '(?P<n>.{3})'


def test_the_regular_expression_of_an_expression_matches_the_names_it_describes():
    expression = parse('tile_<X,I,2>_<Y,I,2>.tif')

    match = re.fullmatch(expression.re(), 'tile_03_12.tif')

    assert match.groupdict() == {'X': '03', 'Y': '12'}
    assert re.fullmatch(expression.re(), 'tile_ab_12.tif') is None
