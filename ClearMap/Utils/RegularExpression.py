# -*- coding: utf-8 -*-
"""
RegularExpression
=================

Module providing routines to check and convert between regular and file 
expressions. 

Utility module fused by :mod:`ClearMap.Utils.TagExpression`.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__docformat__ = 'rest'

import numbers
import re
import sre_parse as sre
import numpy as np


from ClearMap.Utils.Formatting import as_type

###############################################################################
### Regular expressions
###############################################################################


def format_expression(expression, ignore = None):
  """Inserts escapes infront of certain regular expression symbols.
  
  Arguments
  ---------
  expression : str
    The regulsr expresion.
  ignore : list of chars
    A list of characters to ignore as regular expressions commands.

  Returns
  -------
  expression : str
    The regular expression with escaped characters that are ignored.
  """
  if ignore is None or len(ignore) == 0:
    return expression;
  
  new = [];
  for i,c in enumerate(expression):
    if c in ignore:
      if i < len(expression) - 1 and expression[i+1] == '*':
        new.append(c);
      elif i > 0 and expression[i-1] != "\\":
        new.append("\\");
        new.append(c)
      else:
        new.append(c);
    else:
      new.append(c);
  
  return ''.join(new);


def is_expression(expression, group_names = None, n_patterns = None, ignore = None, exclude = None, verbose = False):
  """Checks if the regular expression fullfills certain criteria
  
  Arguments
  ---------
  expression : str
    The regular expression to check.
  group_names :  list of str or None
    List of group names expected to be present in the expression.
  n_patterns : int or None
    Number of group patterns to expect. If negative, the expression is expted to have at least this number of groups. 
  ignore : list of chars or None
    Optional list of chars that should not be regarded as a regular expression command. 
    Useful for filenames setting ignore = ['.'].
  exclude : list of str or None
    Exculde these tokens when counting groups.
  verbose : bool
    If True, print reason for expression to not fullfil desired criteria.
    
  Returns
  -------
  is_expression : bool
    Returns True if the expression fullfills the desired criteria.
  """
  if not isinstance(expression, str):
    if verbose:
      print('Expression is not a string!');
    return False;
    
  if group_names is None and n_patterns is None:
    return True;
  
  #ignore certain commands
  expression = format_expression(expression, ignore = ignore);
  
  #parse regular expression 
  pattern = sre.parse(expression);

  #group patterns
  gd = pattern.pattern.groupdict
    
  if group_names is None:
    group_names = [];
  elif group_names is all:
    group_names = gd.keys();
  
  for gn in gd.keys():
    if gn not in group_names:
      if verbose:
        print('Expression does contain a non required group %s!' % gn);
      return False;
  
  for gn in group_names:
    if gn not in gd.keys():
      if verbose:
        print('Expression does not contain required group %s!' % gn);
      return False;

  if exclude is None:
    exclude = [];
  
  placeholders = [l[0] for l in pattern if  l[0] != sre.LITERAL and l[0] not in exclude];
  n = len(placeholders);
    
  if n_patterns is not None:
    if n_patterns > 0 and n_patterns != n:
      if verbose:
        print('Expression has more than %d regular expression patterns: %r!' % (n_patterns, placeholders));
      return False;
    
    elif n_patterns < 0 and n <= -n_patterns:
      if verbose:
         print('Expression has less than %d regular expression patterns: %r!' % -n_patterns, placeholders);
      return False;
  
  if n == 0:
    if verbose:
       print('Expression has no regular expression patterns!');
    return False;
  
  return True;


###############################################################################
### Groups
###############################################################################

def subpatterns_to_groups(expression, ignore = None, exclude = None, group_names = None):
  """Replaces subpatterns with groups in a regular expression.
  
  Arguments
  ---------
  expression : str
    The regular expression to check.
  ignore : list of chars or None
    Optional list of chars that should not be regarded as a regular expression command. 
    Useful for filenames setting ignore = ['.'].
  exclude : list of str or None
    Exculde these tokens when counting groups.
  group_names : list of str
    The group names to use for the new groups.
    
  Returns
  -------
  expression : str
    The regular expression with subpatterns replaced as groups.
  """
  pattern = expression_to_pattern(expression, ignore=ignore);
  #print pattern.pattern.groupdict
  
  
  if exclude is None:
    exclude = [];
  exclude.extend([sre.LITERAL, sre.GROUPREF, sre.AT])
  
  if group_names is None:
    group_names = [];
  
  gid = pattern.pattern.groups;
  p_new = [];
  gd_new = {};
  for p in pattern:
    if p[0] == sre.SUBPATTERN:
      if p[1][0] not in pattern.pattern.groupdict.values():
        if len(group_names) > 0:
          name = group_names.pop(0);
          gd_new[name] = p[1][0];
      p_new.append(p);
    elif p[0] in exclude:
      p_new.append(p);
    else:
      gid += 1;
      p_new.append((sre.SUBPATTERN, (gid, [p])));
      if len(group_names) > 0:
        name = group_names.pop(0);
        gd_new[name] = gid;
  pattern.data = p_new;  
  pattern.pattern.groupdict.update(gd_new);
  
  expression = pattern_to_expression(pattern);
  return expression;  


def n_groups(expression):
  """Returns the number of groups in the expression.
  
  Arguments
  ---------
  expression : str
    The regular expression.
  
  Returns
  -------
  n : int
   The number of groups in the epxression.
  """
  p = expression_to_pattern(expression);
  return p.pattern.groups - 1;
  

def group_names(expression):
  """Returns the names of groups in the regular expression
  
  Arguments
  ---------
  expression : str
    The regular expression.
  
  Returns
  -------
  names : list of str
      The group names in the regular expression sorted according to appearance.
  """
  
  #parse regular expression for name expressions
  p = sre.parse(expression);
  gd = p.pattern.groupdict;
  names = list(gd.keys());
  
  #sort according to appearance
  order = np.argsort(list(gd.values()));
  names = [names[o] for o in order];
  
  return names;
    

def group_dict(expression, value, as_types = [int, float]):
  """Returns a dictionary with the values of the groups in the regular expression that match the value string.
  
  Arguments
  ---------
  expression : string
    The regular expression.
  value : string
    The text to match and extract group values from.
  as_types : list of types
    List of types to try to convert the extracted group value to.
  
  Returns
  -------
  values : dict
    The values for each group item.
  """
  s = re.compile(expression).search;
  m = s(value);
  
  if m is None:
    return {};
  else:
    gd = m.groupdict();
    if as_types is None:
      return gd;
    
    gd = { k : as_type(v, types=as_types) for k,v in gd.items()};
    return gd;

    
def insert_group_names(expression, groups = None, ignore = None):
  """Inserts group names into a regular expression for spcified groups.
  
  Arguments
  ---------
  expression : str
    The regular expression.
  groups : dict or None
    A dictionary specifying the group names as {groupid : groupname}.
    
  Returns
  -------
  expression : str
    The regular expression with named groups.
  """
  if not isinstance(groups, dict) or len(groups) == 0:
    return expression;
  
  expression = format_expression(expression, ignore=ignore);
  pattern = sre.parse(expression);
  
  #group patterns
  gd = pattern.pattern.groupdict
  for i,n in groups.items():
    gd[n] = i;
  pattern.pattern.groupdict = gd;
  
  return pattern_to_expression(pattern);


###############################################################################
### Regular expressions and patterns
###############################################################################

TEXT = 'text';

class PatternInverter():
  # inverted categories
  in_categories =    { 
      sre.CATEGORY_DIGIT :  r"\d", 
      sre.CATEGORY_NOT_DIGIT: r"\D", 
      sre.CATEGORY_SPACE : r"\s",
      sre.CATEGORY_NOT_SPACE : r"\S",
      sre.CATEGORY_WORD : r"\w",
      sre.CATEGORY_NOT_WORD : r"\W"
      };
  
  at_categories = {
      sre.AT_BEGINNING_STRING : r"\A", 
      sre.AT_BOUNDARY : r"\b",
      sre.AT_NON_BOUNDARY : r"\B", 
      sre.AT_END_STRING : r"\Z",
      sre.AT_BEGINNING : "^",
      sre.AT_END : "$"
      };
  
  escapes = {
      ord("\a"): r"\a",
      ord("\b"): r"\b",
      ord("\f"): r"\f",
      ord("\n"): r"\n",
      ord("\r"): r"\r",
      ord("\t"): r"\t",
      ord("\v"): r"\v",
      ord("\\"): "\\"
      }
  
  def __init__(self, groups = None):
    if groups is None:
      groups = {};
    self.groups = groups;
  
  def generate_any(self, pattern):
    return '.';
  
  def generate_literal(self, pattern):
    _, val = pattern; 
    c = self.escapes.get(val);
    if c is None:
      return chr(val);
    else:
      return c; 
      
  def generate_not_literal(self, pattern):
    _, val = pattern; 
    c = self.escapes.get(val);
    if c is None:
      c = chr(val);    
    return "[^" + c + "]";
  
  def generate_max_repeat(self, pattern):
    rmin = pattern[1][0];
    rmax = pattern[1][1];
    rep = pattern[1][2];
    
    if rmin == 0 and rmax == sre.MAXREPEAT:
      return self.generate_re(rep) + '*';
    if rmin == rmax:
      return self.generate_re(rep) + '{' + str(rmin) + '}';
    return self.generate_re(rep) + '{' + str(rmin) + ',' + str(rmax) + '}';
    
  def generate_min_repeat(self, pattern):
    rep = pattern[1][2];
    return self.generate_re(rep) + '*?';
    
  def generate_group(self, pattern, group_name):
    return '(?P<' + group_name + '>' + self.generate_re(pattern) + ')';
    
  def generate_group_ref(self, pattern):
    gid = pattern[1];
    return '(?P=' + self.groups[gid] + ')';
   
  def generate_negate(self, pattern):
    return '^';
  
  def generate_range(self, pattern):
    r = pattern[1];
    return chr(r[0]) + '-' + chr(r[1]);
  
  def generate_in(self, pattern):
    p = pattern[1];
    if len(p) == 1:
      tp = p[0];
      if tp[0] == sre.CATEGORY:
        return self.in_categories[tp[1]];
      elif tp[0] == sre.AT:
        return self.at_categories[tp[1]];
    return '[' + self.generate_re(p) + ']';
  
  def generate_branch(self, pattern):
    return '|'.join([self.generate_re(p) for p in pattern[1][1]]);
  
  def generate_subpattern(self, pattern):    
    sid = pattern[1][0];
    if sid in self.groups.keys():
      group_name = self.groups[sid];
      return self.generate_group(pattern[1][1], group_name);
    
    if sid is not None:
      return '(' + self.generate_re(pattern[1][1]) + ')';
    
    sp = pattern[1][1];
    if sp[0][0] == sre.GROUPREF_EXISTS:
      gr = sp[0];
      gid = gr[1][0];
      gyes = gr[1][1];
      gno = gr[1][2];
      if self.groups is not None and gid in self.groups.keys():
        group_name = self.groups[gid];
      else:
        group_name = str(gid);
      if gno is None:
        return '(?(' + group_name + ')' + self.generate_re(gyes) + ')';
      else:
        return '(?(' + group_name + ')' + self.generate_re(gyes) + '|' + self.generate_re(gno) + ')';
    else:
      return '(?:' + self.generate_re(pattern[1][1]) + ')';
  
  def generate_assert(self, pattern):    
    return '(?=' + self.generate_re(pattern[1][1]) + ')';
    
  def generate_assert_not(self, pattern):    
    return '(?!' + self.generate_re(pattern[1][1]) + ')';
   
  def generate_at(self, pattern):
    return self.at_categories[pattern[1]];
  
  def generate_re_type(self, pattern):
    ptype = pattern[0];
    if ptype == sre.LITERAL:
      return self.generate_literal(pattern);
    if ptype == sre.NOT_LITERAL:
      return self.generate_not_literal(pattern);
    elif ptype == sre.ANY:
      return self.generate_any(pattern);
    elif ptype == sre.AT:
      return self.generate_at(pattern);
    elif ptype == sre.ASSERT:
      return self.generate_assert(pattern);
    elif ptype == sre.ASSERT_NOT:
      return self.generate_assert_not(pattern);
    elif ptype == sre.SUBPATTERN:
      return self.generate_subpattern(pattern);
    elif ptype == sre.BRANCH:
      return self.generate_branch(pattern);
    elif ptype == sre.IN:
      return self.generate_in(pattern);   
    elif ptype == sre.GROUPREF:
      return self.generate_group_ref(pattern);  
    elif ptype == sre.MIN_REPEAT:
      return self.generate_min_repeat(pattern);
    elif ptype == sre.MAX_REPEAT:
      return self.generate_max_repeat(pattern);
    elif ptype == sre.RANGE:
      return self.generate_range(pattern);
    elif ptype == sre.NEGATE:
      return self.generate_negate(pattern); 
    elif ptype == TEXT:
      return pattern[1];
    else:
      raise RuntimeError("Inversion of pattern of type '%s' not implemented!" % ptype);    
  
  def generate_re(self, pattern):
    r = '';    
    for p in pattern:
      r += self.generate_re_type(p);
    return r;    
 
 

def expression_to_pattern(expression, ignore = None):
  """Convert a regular expression to a parsed pattern for manipulation
  
  Arguments
  ---------
  expression : str
    The regular expression to convert.
    
  Returns
  -------
  pattern : list
    The parsed pattern of the regular expression.
  """
  expression = format_expression(expression, ignore = ignore);
  return sre.parse(expression);


def pattern_to_expression(pattern):
  """Convert a pattern to regular expression
  
  Arguments
  ---------
  pattern : list
    The regular expression in pattern form.
      
  Returns
  -------
  expression : str
    The regular expression.
  """  
  if isinstance(pattern, sre.SubPattern) or isinstance(pattern, sre.Pattern):
    gd = pattern.pattern.groupdict;
  else:
    gd = {};
  
  gdi = {y:x for x,y in gd.items()};
  pi = PatternInverter(groups = gdi);
  return pi.generate_re(pattern);


def replace(expression, replace = None, ignore = None):
  """Replaces patterns in a regular expression with given strings
  
  Arguments
  ---------
  expression : str
    The regular expression.
  replace : dict
    The replacements to do in the regular expression given as {pos : str} or {groupname : str}.
  ignore : list or chars
    Ignore certain regular expression commands.
  
  Returns
  -------
  replaced : str
    The regular expression with replacements.
  """
  if not isinstance(replace, dict): 
    return expression;

  #ignore certain chars
  expression = format_expression(expression, ignore=ignore);

  #parse regular expression 
  pattern = sre.parse(expression);

  #group patterns
  gd = pattern.pattern.groupdict
  gdi = {y:x for x,y in gd.items()};
  
  gkeys = gdi.keys();
  rkeys = replace.keys();
  
  newpattern = [];
  
  replace_group = {};  
  
  for p in pattern:
    if p[0] == sre.SUBPATTERN:
      rid = sid = p[1][0];
      if sid is not None:
        if sid in gkeys:
          sid = gdi[sid];      
        if sid in rkeys:
          r = replace[sid];
          if isinstance(r, str):
            rr = r;
          elif isinstance(r, numbers.Integral):
            #digit format
            dp = p[1][1];
            if isinstance(dp, sre.SubPattern):
              if dp[0][0] == sre.MAX_REPEAT:
                mr = dp[0][1][0];
                if not dp[0][1][2].data == [(sre.IN, [(sre.CATEGORY, sre.CATEGORY_DIGIT)])]:
                  rr = '%r' % r;
                else:
                  rr = ('%s0%dd' % ('%', mr)) % r;
              elif dp.data == [(sre.IN, [(sre.CATEGORY, sre.CATEGORY_DIGIT)])]:
                  rr = '%d' % r;
          else:
            rr = '%r' % r;
          newpattern += [(TEXT, rr)];
          replace_group[rid] = rr;
        else:
          newpattern.append(p);
      else:
        newpattern.append(p);
    elif p[0] == sre.GROUPREF:
      rid = p[1];
      newpattern += [(TEXT, replace_group[rid])];
    else:
      newpattern.append(p);
  
  pattern.data = newpattern;
  return pattern_to_expression(pattern);
      

###############################################################################
### Regular expressions and glob
###############################################################################

def expression_to_glob(expression, replace = None, default = '*', ignore = '.[]'):
  """Converts a regular expression to a glob expression, e.g. to search for files
  
  Arguments
  ---------
  expression : str
    The regular expression.
  replace : dict, all or None
    A dictionary specifying how to replace specific groups. If all or None, all groups are replaced with the default.
  ignore : list of chars
    Ignore these special chars in the regular expression.
    
  Returns
  -------
  expression : str
    The regular expression in glob form.
  """
  expression = format_expression(expression, ignore=ignore);
  
  if not isinstance(replace, dict):
    replace = {};
  
  #parse regular expression 
  pattern = sre.parse(expression);

  #group patterns
  gd = pattern.pattern.groupdict
  gdi = {y:x for x,y in gd.items()};
  
  gkeys = gdi.keys();
  rkeys = replace.keys();
  
  defp = (TEXT, default);
  
  newpattern = [];
  for p in pattern:
    if p[0] == sre.SUBPATTERN:
      sid = p[1][0];
      if sid is not None:
        if sid in rkeys:
          newpattern.append((TEXT, replace[sid]));
        elif sid in gkeys:
          gn = gdi[sid];
          if gn in rkeys:
            newpattern.append((TEXT, replace[gn]));
          else:
            newpattern.append(defp);
        else:
          newpattern.append(defp);
      else:
        newpattern.append(defp);
    else:
      newpattern.append(p);
  
  pattern.data = newpattern;
  g = pattern_to_expression(pattern);
  
  #replace '[' ']' with '[[]' and '[]]'
  gn = ''.join(['[' + c + ']' if c in '[]' else c for c in g]);
  
  return gn;
  

def glob_to_expression(expression, groups = None, to_group = '*'):
  """Converts a glob expression to a regular expression
  
  Arguments
  ---------
  expression : str
    A glob expression.
  groups : dict or None
    A dicitonary specifying how to name groups in the form {id : name}
  to_group : list of chars or None
    Glob placeholders to convert to a group.
  
  Returns
  -------
  expression : str
    The regular expression.
  """
  #replace '[[]' and '[]]'
  expression = expression.replace('[[]', '[');
  expression = expression.replace('[]]', ']');
  
  if to_group is None:
    to_group = '';
  if to_group is all:
    to_group = '*?[';
  
  #convert to re
  i, n = 0, len(expression)
  res = ''
  while i < n:
    c = expression[i]
    i += 1
    if c == '*':
      if c in to_group:
        res = res + '(.*)';
      else:
        res = res + '.*';
    elif c == '?':
      if c in to_group:
        res = res + '(.)';
      else:
        res = res + '.'
    elif c == '[':
      j = i
      if j < n and expression[j] == '!':
          j = j+1
      if j < n and expression[j] == ']':
          j = j+1
      while j < n and expression[j] != ']':
          j = j+1
      if j >= n:
          res = res + '\\['
      else:
        inner = expression[i:j];
        i = j+1
        if inner[0] == '!':
            inner = '^' + inner[1:]
        inner = '[' + inner + ']';
        if c in to_group:
          inner = '(' + inner + ')';
        res = res + inner
    elif c == '.':
      res = res + re.escape(c);
    else:
      res = res + c;
  expression = res;

  #replace groups  
  expression =  insert_group_names(expression, groups = groups);

  return expression;


def _test():
  """Test RegularExpression module"""
  import ClearMap.Utils.RegularExpression as cre
  #reload(cre);
  expression = '/path/to/files/image_file_(?P<row>\d{2})_(?P<col>\d{2})_(?P<z>\d{4}).tif'
  filename = '/path/to/files/image_file_05_01_0012.tif'
  
  cre.group_names(expression)
  
  cre.group_dict(expression, filename)
  
  cre.is_expression(expression, groups = ['row', 'col', 'z'], exclude = ['any'])
  
  cre.is_expression(expression, groups = ['row', 'col', 'z'], ignore = ['.'])  

  cre.is_expression(expression, groups=all, n_patterns=3, verbose=True)
  
  cre.is_expression(expression, groups = all, n_patterns=3, verbose = True,  ignore = ['.'])
  
  
  #reload(cre)  
  expression = '/test/test_(?P<row>\d{4})_(?P<col>\d{3}).tif';
  pattern = cre.expression_to_pattern(expression);
  expression_new = cre.pattern_to_expression(pattern);
  print(pattern)
  print(expression_new)
  print(expression_new == expression) 
  
  cre.group_names(expression)

  #reload(cre)  
  expression = '/test/test_(?:\d)_(?P<col>\d{3})_.(?P=col)tif[00 x 01]$';
  pattern = cre.expression_to_pattern(expression, ignore = '.[]');
  expression_new = cre.pattern_to_expression(pattern)  
  expression == expression_new
  
  #reload(cre)
  cre.insert_group_names(r'test/([7-8]).tif', groups = {1 : 'z'})
  
  #reload(cre)
  source = r'/test/test_(?:\d)_(?P<col>\d{3})_[7-9]\.(?P=col)tif$';
  print(cre.replace(source, {'col' : 4}))

  
  #reload(cre)
  cre.expression_to_glob('test/([7-8]).tif', ignore = '.[]')

  #reload(cre)  
  cre.glob_to_expression('test/*.tif', groups = {1 : 'first'} )
  
  
  #reload(cre)
  cre.expression_to_pattern('/home/test(?P<z>\d{3}).tif', ignore = '.[]')
  
  #reload(cre);
  cre.subpatterns_to_groups('/home/test\d{3}.tif', ignore = '.')