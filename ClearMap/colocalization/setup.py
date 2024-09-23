from setuptools import setup
from Cython.Build import cythonize


def make_ext(modname, pyxfilename):
    import numpy as np
    from distutils.extension import Extension

    ext = Extension(
        modname,
        sources=[pyxfilename],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
        language="c++",
    )

    return ext


ext = make_ext("ClearMap.colocalization.bounding_boxes", "bbox.pyx")

setup(ext_modules=cythonize([ext]))
