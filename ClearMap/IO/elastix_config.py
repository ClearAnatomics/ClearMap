from collections.abc import Iterable


class ElastixEntry:
    def __init__(self, ln=None, key=None, values=None):
        self.ln = ln
        if ln is not None:
            self.key = None
            self.values = None
            self.parse()
        elif key is not None and values is not None:
            self.key = key
            if not isinstance(values, Iterable):
                values = [values]
            self.values = values
            self.build_ln()
        else:
            raise ValueError('Missing arguments, at least ln or both key and values required')

    def parse(self):
        if self.type == 'entry':
            elements = self.ln[1:-2].split(' ')  # Remove leading "(" and trailing ")\n"
            self.key = elements[0]
            self.values = elements[1:]

    @property
    def type(self):
        if self.ln.startswith('//'):
            return 'comment'
        elif self.ln.startswith('('):
            return 'entry'
        elif self.ln.startswith('\n'):
            return 'blank'
        else:
            raise NotImplementedError(f'Unknown entry type for "{self.ln}"')

    def __str__(self):
        return self.ln

    def build_ln(self):
        values_str = ' '.join(self.__format_value(v) for v in self.values)
        self.ln = f'({self.key} {values_str})\n'

    def __format_value(self, val):
        return f'"{val}"' if isinstance(val, str) else str(val)


class ElastixParser:

    def __init__(self, src_path=None):
        self.data = []
        self.keys = []
        if src_path is not None:
            self.path = src_path
            self.parse()

    def parse(self):
        with open(self.path, 'r') as cfg:
            self.data = [ElastixEntry(ln=ln) for ln in cfg.readlines()]
        self.__update_keys()

    def __update_keys(self):
        self.keys = [d.key for d in self.data if d.key is not None]

    def write(self):
        with open(self.path, 'w') as cfg:
            cfg.writelines([str(e) for e in self.data])

    def __getitem__(self, item):
        if item not in self.keys:
            raise KeyError(f'Key {item} missing')
        else:
            for d in self.data:
                if d.key == item:
                    return d.values

    def __setitem__(self, key, value):
        for d in self.data:
            if d.key == key:
                d.values = value
                d.build_ln()
                break
        else:
            self.data.append(ElastixEntry(key=key, values=value))
            self.__update_keys()
