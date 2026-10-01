import sys, types
m = types.ModuleType('pyximport'); m.install = lambda *a, **k: None; m.pyximport = m; m.get_distutils_extension = lambda *a, **k: (None, None); sys.modules['pyximport'] = m
CALLS = []
class Code(types.ModuleType):
    def __init__(self, name, missing=()):
        super().__init__(name); self._missing = set(missing)
    def __getattr__(self, n):
        if n.startswith('__') or n in self._missing: raise AttributeError(n)
        def f(*a, **k):
            CALLS.append((n, a, k))
            return None
        return f
def stub(modname, missing=()):
    sys.modules[modname] = Code(modname, missing)
    return sys.modules[modname]

import importlib.abc, importlib.machinery
from unittest import mock
_REAL_MISSING = set()
class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import any missing third-party module as a MagicMock-backed module."""
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in ('ClearMap', 'Cython', 'cython', 'pytest', '_pytest', 'hypothesis'):
            return None
        for f in sys.meta_path:
            if f is self: continue
            try:
                spec = f.find_spec(name, path, target)
            except Exception:
                spec = None
            if spec is not None:
                return None
        _REAL_MISSING.add(name)
        return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = mock.MagicMock(name=spec.name); m.__path__ = []; m.__spec__ = spec; m.__name__ = spec.name
        return m
    def exec_module(self, module): pass
sys.meta_path.append(_Finder())
