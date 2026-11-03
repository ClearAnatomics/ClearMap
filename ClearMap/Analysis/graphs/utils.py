import sys
import mmap
import struct

import graph_tool as gt
import numpy as np

from ClearMap.External import pickle_python_3 as pickle



# ---- Type tables for properties scanner -----------------------------------------------------------
# Known value-type byte codes used by graph-tool. These have been stable for years.
# If you hit a ValueError with an unknown code, add it here (and in TYPE_SIZES).
TYPE_NAMES = {
    0x00: "bool",            # stored as 1 byte (uchar)
    0x01: "int16_t",
    0x02: "int32_t",
    0x03: "int64_t",
    0x04: "double",
    0x05: "long double",     # 16 bytes on x86_64 builds; treat as 16 here
    0x06: "string",
    0x07: "vector<bool>",
    0x08: "vector<int16_t>",
    0x09: "vector<int32_t>",
    0x0A: "vector<int64_t>",
    0x0B: "vector<double>",
    0x0C: "vector<long double>",
    0x0D: "vector<string>",
    0x0E: "python::object",  # pickled bytes (length-prefixed string)
    # The following two are present in recent versions:
    0x0F: "int8_t",          # signed 8-bit (rare but supported)
    0x10: "uint64_t",        # unsigned long (common for IDs/counters)
}

# Fixed-width sizes for scalar codes (others are length-prefixed)
TYPE_SIZES = {
    0x00: 1,   # bool (uchar)
    0x0F: 1,   # int8_t
    0x01: 2,   # int16_t
    0x02: 4,   # int32_t
    0x03: 8,   # int64_t
    0x04: 8,   # double
    0x05: 16,  # long double (graph-tool writes 16 bytes)
    0x10: 8,   # uint64_t
}

# vector element sizes
VEC_ELEM_SIZE = {
    0x07: 1,    # vector<bool> (as bytes)
    0x08: 2,    # vector<int16_t>
    0x09: 4,    # vector<int32_t>
    0x0A: 8,    # vector<int64_t>
    0x0B: 8,    # vector<double>
    0x0C: 16,   # vector<long double)
}

SCOPE = {0x00: "graph", 0x01: "vertex", 0x02: "edge"}

# ---- Helpers for properties scanner ---------------------------------------------------------------
def u8(mm, off):
    return mm[off], off + 1

def u64(mm, off, endian):
    return struct.unpack(endian + 'Q', mm[off:off+8])[0], off + 8

def read_str(mm, off, endian):
    L, off = u64(mm, off, endian)
    s = mm[off:off+L].decode('utf-8', errors='strict')
    return s, off + L

def skip_str(mm, off, endian):
    L, off = u64(mm, off, endian)
    return off + L

def skip_vec_scalar(mm, off, endian, elem_size):
    L, off = u64(mm, off, endian)
    return off + L * elem_size

def skip_vec_string(mm, off, endian):
    L, off = u64(mm, off, endian)
    for _ in range(L):
        off = skip_str(mm, off, endian)
    return off

def idx_width_for_N(N):
    if N <= 0xFF: return 1
    if N <= 0xFFFF: return 2
    if N <= 0xFFFFFFFF: return 4
    return 8

# -------------- properties scanner ----------------------------------------------------------------

def scan_gt_props(path):
    """Scan a .gt file and return a list of (scope, name, type-name) for each property."""
    out = []
    with open(path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        off = 0

        # Magic "\xe2\x9b\xbe gt" (the teacup), version, endianness
        if len(mm) < 8 or mm[:6] != b'\xe2\x9b\xbe gt':
            raise ValueError("Not a .gt file (bad magic)")
        off = 6
        version, off = u8(mm, off)
        endian_flag, off = u8(mm, off)
        endian = '<' if endian_flag == 0x00 else '>'

        # Comment (length-prefixed UTF-8). Your example shows stats in here.
        comment, off = read_str(mm, off, endian)
        # print("comment:", comment)  # optional

        # Graph structure
        directed_flag, off = u8(mm, off)
        N, off = u64(mm, off, endian)

        idx_sz = idx_width_for_N(N)
        E = 0
        for _ in range(N):
            d_i, off = u64(mm, off, endian)
            E += d_i
            off += d_i * idx_sz  # skip neighbors

        # If undirected, E is the number of stored adjacency entries (one per edge).
        # That is exactly the number of edge-property values that follow.

        # Properties
        total_props, off = u64(mm, off, endian)

        for _ in range(total_props):
            scope_code, off = u8(mm, off)
            name, off = read_str(mm, off, endian)
            vtype, off = u8(mm, off)

            scope = SCOPE.get(scope_code, f"unknown({scope_code})")
            vname = TYPE_NAMES.get(vtype, None)
            if vname is None:
                raise ValueError(f"Unknown value-type code 0x{vtype:02x} at file offset {off-1}. "
                                 f"Please extend TYPE_NAMES/TYPE_SIZES/VEC_ELEM_SIZE as needed.")

            out.append((scope, name, vname))

            # Decide how many values to skip
            count = 1 if scope_code == 0x00 else (N if scope_code == 0x01 else E)

            # Skip payload
            if vtype in TYPE_SIZES:               # fixed-width scalar (incl. uint64 and long double)
                off += TYPE_SIZES[vtype] * count
            elif vtype == 0x06:                   # string
                for _ in range(count):
                    off = skip_str(mm, off, endian)
            elif vtype in VEC_ELEM_SIZE:          # vector<T> where T is scalar
                for _ in range(count):
                    off = skip_vec_scalar(mm, off, endian, VEC_ELEM_SIZE[vtype])
            elif vtype == 0x0D:                   # vector<string>
                for _ in range(count):
                    off = skip_vec_string(mm, off, endian)
            elif vtype == 0x0E:                   # python::object (bytes, length-prefixed)
                for _ in range(count):
                    off = skip_str(mm, off, endian)
            else:
                # Should be unreachable due to earlier guard
                raise ValueError(f"Unhandled type 0x{vtype:02x} at {off}")
    return out
# --------------------------------------------------------------------------------------------------

def pickler(stream, obj):
    sstream = gt.gt_io.BytesIO()
    pickle.dump(obj, sstream)  # ,  gt.gt_io.GT_PICKLE_PROTOCOL)
    stream.write(sstream.getvalue())


def unpickler(stream):
    data = stream.read(buflen=2**31)
    # print('unpickler loaded %d' % len(data))
    sstream = gt.gt_io.BytesIO(data)
    if sys.version_info < (3,):
        return pickle.load(sstream)
    return pickle.load(sstream, encoding="bytes")


def edges_to_vertices(edges):
    return np.array([[int(e.source()), int(e.target())] for e in edges])
    # return np.array([[e.source(), e.target()] for e in edges], dtype=int)  # TODO: see which is better cpu/RAM wise
