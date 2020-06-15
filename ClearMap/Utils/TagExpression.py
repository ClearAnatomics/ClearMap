# -*- coding: utf-8 -*-
"""
TagExpression
=============

Module providing routines to check and convert between tag expressions.
This simplfies the handling of list of files from using regular expressions
to tag expression. 

A tag is a simple placeholder for a string or number in a file name.
An expression is a filename with any number of tags.

A tag has the format <Name,Type,Width>. 
  * Name : str that specifies the tag name
  * Type : 'I' or 'S' for integer or string, optional and defaults to 'I'
  * Width : int, if given indicates a fixed width  and missing digist or chars
    are replaced by trailing zeros or spaces.

The module also provides functions to infer the tag expressions from a list of
file names via the function :func:`~ClearMap.Utils.TagExpression.detect`.

Expressions can also be converted to glob or regular expressions.


Example
-------
An example tag expression would be 'file_<X,2>_<Y,3>.npy' and would for example
match the filename 'file_01_013.npy'

>>> import ClearMap.Utils.TagExpression as te
>>> e = te.Expression('file_<X,2>_<Y,3>.npy')
>>> e.tag_names()
e.tag_names()

>>> e.glob()
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
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import copy

import re

TAG_START = '<';
TAG_END   = '>';
TAG_SEPARATOR = ',';
TAG_INT = 'I'; 
TAG_STR = 'S';

def ttype_to_dtype(ttype):
  if ttype == TAG_INT or ttype is None:
    return int
  elif ttype == TAG_STR:
    return str;
  else:
    raise ValueError('The specified tag type %r is not valid!' % ttype);

def default_tag_name(index = None):
  tag = 'Tag';
  if index is not None:
    tag += '%d' % index;
  return tag;


class Tag(object):
  def __init__(self, tag = None, name = None, ttype = TAG_INT, width = None, reference = None, trange = None):
    if tag is not None:
      self.parse(tag);
    else:
      self.name = name;
      self.ttype = ttype;
      self.width = width;
      self.reference = reference;
      self.trange = trange; 
  
  def label(self, index = None):
    if self.name is not None:
      return self.name;
    else:
      return default_tag_name(index = index)
  
  def dtype(self):
    return ttype_to_dtype(self.ttype);

  def tag(self):
    t = TAG_START;
    if self.name is not None:
      t += self.name + TAG_SEPARATOR;
    if self.ttype is not None:
      t += self.ttype + TAG_SEPARATOR;
    if self.width is not None:
      t += str(self.width) + TAG_SEPARATOR;
    if len(t) > len(TAG_START):
      t = t[:-len(TAG_SEPARATOR)];
    t += TAG_END;
    return t;
  
  def glob(self):
    e = '';
    if self.width != None:
      if self.ttype == TAG_INT or self.ttype is None:
        s = '[0-9]'
      elif self.ttype == TAG_STR:
        s = '?'
      e += ''.join([s] * self.width);
    else:
      e += '*';
    return e;

  def re(self, name = None):
    if name is None:
      name = self.name;
    e = '';
    if self.reference:
      if name is not None:
        e += '(?P=' + name + ')';
      else:
        raise ValueError('Name needs to be given for a referenced tag!');
    else:
      if name is not None:
        e += '(?P<' + name + '>';
      else:
        e += '('
      if self.ttype == TAG_INT or self.ttype is None:
        e += '\d';
      else: # self.ttype == TAG_STR:
        e += '.';
      if self.width is not None:
        e += '{' + str(self.width) + '}';
      else:
        e += '*?';
      e += ')'
    return e;
    
  def string(self, value = None):
    if value is None:
      return self.tag();
    
    if self.width is None:
      if self.ttype == TAG_INT or self.ttype is None:
        frmt = '%d';
      else:
        frmt = '%s';
    else:
      if self.ttype == TAG_INT or self.ttype is None:
        frmt = '%0' + str(self.width) + 'd';
      else:
        frmt = '%' + str(self.width) + 's';
    return frmt % value;
  
  
  def string_from_index(self, index = None):
    if index is not None:
      if self.trange is not None:
        value = self.trange[index];
      else:
        value = index;
    else:
      value = None;
    return self.string(value=value);
  
  
  def value(self, string):
    return self.dtype()(string);
  
    
  def index(self, value):
    if self.trange is None:
      if isinstance(value, str):
        raise IndexError('No range to determine index for tag %r and value %r!' % (self, value));
      else:
        return value;
    else:
      for i,r in enumerate(self.trange):
        if value == r:
          return i;
      raise IndexError('Value %r not in tag range %r!' % (value, self.trange));
  
  def parse(self, tag):
    if len(tag) < len(TAG_START) + len(TAG_END):
      raise ValueError('The string %s is not a valid tag!' % tag)
    if tag[:len(TAG_START)] != TAG_START:
      raise ValueError('Expecting the tag to start with %s found %s!' % (TAG_START, tag[:len(TAG_START)]));
    if tag[-len(TAG_END):] != TAG_END:
      raise ValueError('Expecting the tag to end with %s found %s!' % (TAG_END, tag[-len(TAG_END):]));
    tag = tag[len(TAG_START):-len(TAG_END)];
    if len(tag) == 0:
      self.__init__();
      return;
    values = tag.split(TAG_SEPARATOR);
    if len(values) > 3:
      raise ValueError('Found %d > %d tag specifications!' % (len(values), 3));
    
    name = [];
    ttype = [];
    width = [];
    for i,v in enumerate(values):  
      try:
        if i == 0: #dont expect a width specification in the first entry
          raise Exception;
        width.append(int(v));
      except:
        if v in [TAG_INT, TAG_STR]:
          ttype.append(v);
        elif len(v) == 0:
          raise ValueError('Found two tag separators without a value in between!');
        else:
          name.append(v);
    
    if len(name) > 1:
      raise ValueError('More than one name found in tag: %r' % name);
    if len(ttype) > 1:
      raise ValueError('More than one type found in tag: %r' % ttype);
    if len(width) > 1:
      raise ValueError('More than one widht found in tag: %r' % width);
    
    name = name[0] if len(name) == 1 else None;
    ttype = ttype[0] if len(ttype) == 1 else None;
    width = width[0] if len(width) == 1 else None;
    
    self.__init__(name=name, ttype=ttype, width=width);

  def __str__(self):
    return self.tag();
  
  def __repr__(self):
    return self.__str__();


class Expression(object):
  def __init__(self, pattern = None):
    if isinstance(pattern, Expression):
      self.pattern = copy.copy(pattern.pattern);
      self.tags = copy.copy(pattern.tags);
    elif isinstance(pattern, str):
      self.parse(pattern);
    else:
      if pattern is None:
        pattern = [];
      self.pattern = pattern;
      self.tags = [p for p in pattern if isinstance(p, Tag) and not p.reference];
   
  def tag(self):
    e = '';
    for p in self.pattern:
      if isinstance(p, Tag):
        e += p.tag();
      else:
        e += p;
    return e;
  
  def re(self):
    e = '';
    n_tag = 0;
    for p in self.pattern:
      if isinstance(p, Tag):
        e += p.re(name = p.label(n_tag));
        n_tag +=1;
      else:
        e += re.escape(p);
    return e;  
    
  def glob(self, values = None):
    e = '';
    n_tag = 0;
    for p in self.pattern:
      if isinstance(p, Tag):
        if values is None:
          e += p.glob();
        else:
          lab = p.label(n_tag);
          if lab in values.keys():
            e += escape_glob(p.string(value = values[lab]));
          else:
            e += p.glob();
        n_tag += 1;
      else:
        e += escape_glob(p);
    return e;
    
  def string(self, values = None):
    e = '';
    n_tag = 0;
    for p in self.pattern:
      if isinstance(p, Tag):
        if values is not None:
          v =  values.get(p.label(n_tag), None);
        else:
          v = None;
        e += p.string(value = v);
        n_tag += 1;
      else:
        e += p;
    return e;
    
  def values(self, string):
    tags = self.tags;
    search = re.compile(self.re()).search;
    match = search(string);
    if match is None:
      return {};
    else:
      d = match.groupdict();
      for k,v in d.items():
        for i,t in enumerate(tags):
          if k == t.label(i):
            v = t.dtype()(v);
            d[k] = v;
            break;
      return d;
    
  def string_from_index(self, indices):
    if not isinstance(indices, dict):
      indices = {t.label(i) : indices[i] for i,t in enumerate(self.tags)};
    e = '';
    n_tag = 0;
    for p in self.pattern:
      if isinstance(p, Tag):
        e += p.string_from_index(index = indices[p.label(n_tag)]);
        n_tag += 1;
      else:
        e += p;
    return e;
    
  def indices(self, string):
    tags = self.tags;
    search = re.compile(self.re()).search;
    match = search(string);
    if match is None:
      raise ValueError('Cannot infer indices from string!')
    else:
      d = match.groupdict();
      for k,v in d.items():
        for i,t in enumerate(tags):
          if k == t.label(i):
            v = t.index(t.dtype()(v));
            d[k] = v;
            break;
    indices = [d[t.label(i)] for t in tags];
    return indices;
  
  def tag_names(self):
    return [t.label(i) for i,t in enumerate(self.tags)];
  
  def ntags(self):
    return len(self.tags); 
  
  def __getitem__(self, i):
    if isinstance(i, int):
      return self.tags[i];
    else:
      for j,n in enumerate(self.tag_names()):
        if n == i:
          return self.tags[j];
      raise IndexError('No tag with name %r!' % i);
  
  def parse(self, expression):
    p = re.compile(TAG_START + '.*?' + TAG_END);
    pattern = [];
    tags = [];
    start = 0;
    for match in p.finditer(expression):
      if match.start() > start:
        pattern.append(expression[start:match.start()]);
      tag = Tag(tag = match.group())
      pattern.append(tag);
      tags.append(tag);
      start = match.end();
    if start < len(expression):
      pattern.append(expression[start:]);
    
    #check for references
    for i,t in enumerate(tags):
      if t.name is not None and t.reference is not True:
        refs = [r for r in tags[i+1:] if r.name == t.name];
        for r in refs:
          r.reference = True;
          if r.ttype is None:
            r.ttype = t.ttype;
          elif r.ttype != t.ttype:
            raise ValueError('The reference %r has not the same type as the tag %r!' % (r,t));
          if r.width is None:
            r.width = t.width;
    
    self.pattern = pattern;
    self.tags = [q for q in pattern if isinstance(q, Tag) and not q.reference];
  
  
  def detect(self, strings, names = None, max_check = None, with_trange = False):
    if not isinstance(strings,list):
      strings = [strings];
    
    ls = [len(s) for s in strings];
    for l in ls:
      if l != ls[0]:
        raise ValueError('Cannot infer tag expression from strings of different length!');
      
    if max_check is None:
      max_check = len(strings);         
      
    if names is None:
      names = [];
    
    #detect differences in filenames
    s0 = strings[0];
    tags = [];
    tag_start = -1;
    tag_end = -1;
    for i, c in enumerate(s0):
      same = True;
      for s in strings[1:]:
        if s[i] != c:
          if i == tag_end:
            tag_end += 1;
          else:
            tag_start = i;
            tag_end   = i + 1;
          break;
      if same and tag_start != -1:
        tags.append((tag_start, tag_end));
        tag_start = -1; tag_end = -1;
  
    #detect trailing zeros  
    tags_full = [];
    for t in tags:
      s,e = t;
      while s > 0 and s0[s-1] == '0':
        s -= 1;
      tags_full.append((s,e));
    tags = tags_full;  
    
    #infer pattern
    pattern = [];
    p = 0;
    for i,t in enumerate(tags):
      s,e = t; 
      if s-p > 0:
        pattern.append(s0[p:s]);
        p = e;
      
      ttype = TAG_INT;
      values = [];
      for s in strings[:max_check]:
        v = s[t[0]:t[1]];
        try:
          v = int(v);
        except:
          ttype = TAG_STR;
          if not with_trange:
            break;
        values.append(v);
      if len(names) > 0:
        name = names.pop(0);
      else:
        name = default_tag_name(i);
      trange = values if with_trange else None;
      pattern.append(Tag(name = name, ttype = ttype, width = t[1] - t[0], trange = trange));
    if p < len(s0) > 0:
        pattern.append(s0[p:]); 
    
    self.pattern = pattern;
    self.tags = [q for q in pattern if isinstance(q, Tag) and not q.reference];
  
  
  def __str__(self):
    return 'TagExpression(' + self.tag() + ')';
    
  def __repr__(self):
    return self.__str__();
      

def parse(expression):
  e = Expression()
  e.parse(expression=expression);
  return e;


def detect(strings, names = None, max_check = None, with_trange = False):
  e = Expression();
  e.detect(strings=strings, names=names, max_check=max_check, with_trange=with_trange);
  return e;


def escape_glob(string):
  e = '';
  for c in string:
    if c in '?[]':
      e += '[' + c + ']';
    else:
      e += c;
  return e;




def _test():
  """Tests"""
  import ClearMap.Utils.TagExpression as te
  #reload(te);  
  
  #values and strings
  t = te.parse('/test/test<X,I,4>_<Y,I,3>_<X>.tif')
  
  s = '/test/test0010_013_0010.tif';
  v = t.values(s);
  s2 = t.string(v)
  s == s2
  
  t.string({'X' : 111})
  
  #indices
  t.indices(s)
  t = te.parse('/test/test<X,I,4>_<Y,S>.tif');
  t['X']
  t['Y'].trange = list('abcd')
  t.string_from_index([0,2])  
  
  #glob
  import ClearMap.Tests.Files as tf
  import glob
  
  s = tf.io.join(tf.tif_sequence, 'sequence<I,4>.tif')
  t = te.parse(s)
  f = glob.glob(t.glob())
  
  #detection
  te.detect(f, names = ['X'])
  
  t = te.detect(f, names = ['X'], with_trange = True);
  t['X'].trange

  
  