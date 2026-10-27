import os
import re
import sys
import subprocess
import platform

from multiprocessing import cpu_count
from pathlib import Path

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# Nothing is built when this file is imported (the tests do), only when it is run (see main): numpy and Cython are
# imported where they are used.

FORCE = '--force' in sys.argv   # python setup.py build_ext --inplace --force

# N_PROCS must be defined before parallel_build_ext is used.
N_PROCS = max(1, cpu_count() - 2)

os_name = platform.system().lower()

# The CPU the compiled code is built for: CLEARMAP_TARGET_ARCHITECTURE is the -march of the compiler.
# * native (the default): the CPU of the machine that builds. Fastest, but the code only runs on CPUs at least as
#   recent.
# * a level that every machine that will run the code has, to build once for several machines (a module for a
#   cluster, a wheel): x86-64-v2, x86-64-v3 (AVX2, FMA)... These names need GCC 11 or later (clang 12 or later);
#   /lib64/ld-linux-x86-64.so.2 --help lists the levels a machine supports.
# * compiler-default: no -march, the baseline of the compiler. Slowest, runs everywhere.
# OPTIMISE_CLEARMAP_COMPILATION_FOR_TARGET (false: compiler-default) is still understood when the above is not set.
TARGET_ARCHITECTURE_ENV_VAR = 'CLEARMAP_TARGET_ARCHITECTURE'
LEGACY_OPTIMISE_FOR_TARGET_ENV_VAR = 'OPTIMISE_CLEARMAP_COMPILATION_FOR_TARGET'
# -march=native: every instruction of the CPU that compiles, so only the same or newer CPUs can run the code
TARGET_THE_CPU_THAT_BUILDS = 'native'
# No -march: only the baseline instructions of the architecture (SSE2 on x86-64), so any CPU of it can run the code
TARGET_ANY_CPU = 'compiler-default'
FALSE_SPELLINGS = ('false', '0', 'no', 'off')
ARCHITECTURE_NAME = re.compile(r'[a-z0-9][a-z0-9._+-]*')  # The value goes to the compiler: nothing else

LARGE_SOURCE_FILES_OPTION = '-flarge-source-files'
HELP_ARGUMENT = '--help=common'  # GCC: lists the options of the compiler. Other compilers may not know it
PROBE_TIMEOUT_SECONDS = 30

requirements = [  # pip
    'lxml'
]

excluded_pyx = ['_Old', '_Todo', 'StatisticsPointListCode', 'flow', 'OrientationCode']

data_dirs = [
    'ClearMap/External/elastix',
    'ClearMap/External/geodesic_distance',
    'ClearMap/Resources',
    'ClearMap/ImageProcessing/Skeletonization',   # .npy and .npy.zip
    'ClearMap/ImageProcessing/Binary',  # .npy and .npy.zip
    'ClearMap/ImageProcessing/machine_learning/vessel_filling/resources'  # .pth
]


###############################################################################
# Parallel C/C++ compilation
###############################################################################

class parallel_build_ext(_build_ext):
    def initialize_options(self):
        super().initialize_options()
        self.parallel = N_PROCS


###############################################################################
# Platform helpers
###############################################################################

def _to_msvc_flags(flags):
    """Convert GCC-style flags to MSVC equivalents where possible."""
    out = []
    for arg in flags:
        if arg == '-fopenmp':
            out.append('/openmp')
        elif arg.startswith('-m'):
            # -march=native, -mtune=native → no MSVC equivalent, skip
            continue
        elif arg == '-O3':
            out.append('/O2')  # MSVC max optimisation
        elif arg == '-w':
            out.append('/w')   # disable warnings
        elif arg.startswith('-f'):
            # -flarge-source-files etc → no MSVC equivalent, skip
            continue
        else:
            out.append(arg)
    return out


def use_the_source_install_dir_for_data():
    """Modify the data install dir to match the source install dir (from https://stackoverflow.com/a/3042436)"""
    from distutils.command.install import INSTALL_SCHEMES  # The distutils of setuptools: it is imported above
    for scheme in INSTALL_SCHEMES.values():
        scheme['data'] = scheme['purelib']


def target_compile_arguments(environment):
    """
    The arguments that choose the CPU the code is built for (not for MSVC, which has no ``-march``).

    Parameters
    ----------
    environment : Mapping[str, str]
        The environment variables (``os.environ``).

    Returns
    -------
    list[str]
        ``-march`` (and ``-mtune`` for the CPU of the machine that builds), nothing for the compiler default.

    Raises
    ------
    ValueError
        If ``CLEARMAP_TARGET_ARCHITECTURE`` cannot be the name of an architecture.
    """
    architecture = environment.get(TARGET_ARCHITECTURE_ENV_VAR, '').strip().lower()
    if not architecture:
        optimisation_is_off = environment.get(LEGACY_OPTIMISE_FOR_TARGET_ENV_VAR, '').strip().lower() in FALSE_SPELLINGS
        architecture = TARGET_ANY_CPU if optimisation_is_off else TARGET_THE_CPU_THAT_BUILDS
    if not ARCHITECTURE_NAME.fullmatch(architecture):
        raise ValueError(f'{TARGET_ARCHITECTURE_ENV_VAR}={architecture!r} is not the name of an architecture '
                         f'(e.g. "{TARGET_THE_CPU_THAT_BUILDS}", "x86-64-v3" or "{TARGET_ANY_CPU}")')
    if architecture == TARGET_ANY_CPU:
        return []
    if architecture == TARGET_THE_CPU_THAT_BUILDS:
        return [f'-march={architecture}', '-mtune=native']
    return [f'-march={architecture}']


def build_compiler_command():
    """
    The command of the compiler that distutils builds the extensions with (``CC``, else the one of the
    interpreter), with its leading arguments. Empty if distutils does not give one (MSVC).
    """
    from distutils.ccompiler import new_compiler  # The distutils of setuptools: it is imported above
    from distutils.sysconfig import customize_compiler
    compiler = new_compiler()
    customize_compiler(compiler)
    return list(getattr(compiler, 'compiler_so', []))


def compiler_lists_option(compiler_command, option):
    """
    Whether the compiler lists an option in its help, to use it only where it is known (GCC only options).

    Parameters
    ----------
    compiler_command : Sequence[str]
        The compiler and its leading arguments (see :func:`build_compiler_command`).
    option : str
        The option, e.g. ``-flarge-source-files``.

    Returns
    -------
    bool
        False if it is not listed, and whenever the compiler cannot say: it is missing, does not understand
        :data:`HELP_ARGUMENT` (clang), fails or takes too long. Never raises: an option not seen is left out.
    """
    if not compiler_command:
        return False
    try:
        completed = subprocess.run([*compiler_command, HELP_ARGUMENT], capture_output=True, text=True,
                                   errors='replace', check=False, timeout=PROBE_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and option in completed.stdout


def default_compile_arguments():
    arguments = []
    if not os_name.startswith('windows'):
        arguments += ['-w', '-O3']

    if os_name.startswith('linux'):
        # Add only if supported by the compiler that builds (GCC only, and not necessarily named gcc)
        if compiler_lists_option(build_compiler_command(), LARGE_SOURCE_FILES_OPTION):
            arguments += [LARGE_SOURCE_FILES_OPTION]

    if not os_name.startswith('windows'):  # MSVC has no -march
        arguments += target_compile_arguments(os.environ)
    return arguments


def default_libraries():
    return [] if os_name.startswith('windows') else ['m']


def openmp_is_used():
    """Whether to compile with OpenMP (``--no-openmp`` turns it off, and is removed from the arguments)"""
    if '--no-openmp' in sys.argv:
        sys.argv.remove('--no-openmp')
        return False
    if os_name.startswith('linux') or os_name.startswith('windows'):
        return True
    if os_name.startswith('darwin'):
        cpp_compiler_version = subprocess.check_output(['c++', '--version']).decode('ascii').split('\n')[0]
        return 'g++' in cpp_compiler_version
    raise ValueError(f'Unknown OS {os_name}')


def compile_and_link_arguments(use_openmp):
    """The (compile, link) arguments of every extension"""
    extra_compile_args = default_compile_arguments()
    extra_link_args = []

    # OpenMP flags (platform-conditional — NOT in file headers)
    if use_openmp:
        extra_compile_args += ['-fopenmp']
        extra_link_args += ['-fopenmp']

    if os_name.startswith('windows'):
        extra_compile_args = _to_msvc_flags(extra_compile_args)
        extra_link_args = _to_msvc_flags(extra_link_args)
    return extra_compile_args, extra_link_args


def module_path_to_dotted(ext_path):
    return os.path.splitext(ext_path)[0].replace(os.sep, '.')


def find_data_files(src_dir):
    out = {}
    for root, subdirs, files in os.walk(src_dir):
        if root not in out:
            out[root.replace(os.sep, '/')] = []
        for f in files:
            if os.path.splitext(f)[-1] not in ('.py', '.pyc'):
                out[root.replace(os.sep, '/')].append(f"{root}/{f}".replace(os.sep, '/'))
    return out


def get_data_files():
    data_files = [('licenses', ['LICENSE.txt', 'LICENSE'])]
    for p in data_dirs:
        data_files.extend([(k, v) for k, v in find_data_files(p).items()])
    data_files.extend([('', ['start_gui.sh'])])
    return data_files


def make_extensions(extra_compile_args, extra_link_args, libraries):
    import numpy as np
    extension_paths = [str(p) for p in Path('ClearMap').rglob('*.pyx')
                       if not any(excl in str(p) for excl in excluded_pyx)]
    extensions = []
    for ext_path in extension_paths:
        extension = Extension(
            name=module_path_to_dotted(ext_path),
            sources=[ext_path],
            libraries=libraries,
            # language= intentionally omitted — read from #distutils: header
            include_dirs=[np.get_include(), os.path.dirname(os.path.abspath(ext_path))],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
        extensions.append(extension)
    return extensions


def read_clearmap_version():
    """The version written in ClearMap/_version.py (read, not imported: ClearMap is not built yet)"""
    namespace = {}
    exec((Path(__file__).resolve().parent / 'ClearMap' / '_version.py').read_text(encoding='utf-8'), namespace)
    return namespace['__version__']


def main():
    from Cython.Build import cythonize

    use_the_source_install_dir_for_data()
    use_openmp = openmp_is_used()
    libraries = default_libraries()
    extra_compile_args, extra_link_args = compile_and_link_arguments(use_openmp)
    extensions = make_extensions(extra_compile_args, extra_link_args, libraries)

    print(f'Building {len(extensions)} extensions with {N_PROCS} processes, '
          f'using OpenMP: {use_openmp}, '
          f'libraries: {libraries}, ',
          f'compile args: {extra_compile_args},'
          f'link args: {extra_link_args}')
    if os_name.startswith('darwin') or os_name.startswith('windows'):
        ext_modules = cythonize(extensions, quiet=True, force=FORCE)
    else:
        ext_modules = cythonize(extensions, nthreads=N_PROCS, quiet=True, force=FORCE)

    packages = find_packages(exclude=('doc', 'tests*', 'pickle_python_2', 'deprecated',
                                      'ClearMap.External.elastix', 'ClearMap.External.geodesic_distance'))
    data_files = get_data_files()

    setup(
        name='ClearMap',
        version=read_clearmap_version(),
        description='3D cell counting and vasculature analysis for lightsheet microscopy',
        install_requires=requirements,
        packages=packages,
        ext_modules=ext_modules,
        cmdclass={'build_ext': parallel_build_ext},
        entry_points={
            'gui_scripts': [
                'clearmap-ui = ClearMap.gui.app:entry_point'
            ],
            'console_scripts': [
                'cell_map = ClearMap.Scripts.cell_map_new_api:main',
                'clearmap-utils = ClearMap.Scripts.clearmap_utils:main'
            ],
        },
        url='',
        license='MIT',
        author='Christoph Kirst, Sophie Skriabine, Charly Rousseau, Etienne Doumazane',
        author_email='',
        include_package_data=True,
        package_data={
            'ClearMap.config': [
                '*.cfg',
                'defaults/*/*.yml',
                'schemas/*/*.yaml',
                'defaults/*/*.cfg',
            ],
            'ClearMap.gui': [
                'creator/*.ui',
                'creator/*.qrc',
                'creator/graphics_resources/*.png',
                'creator/graphics_resources/*.jpg',
                'creator/graphics_resources/*.svg',
                'creator/icons/*.png',
                'creator/icons/*.jpg',
                'creator/icons/*.svg',
            ],
        },
        data_files=data_files,
        zip_safe=False
    )


if __name__ == '__main__':  # Not when it is imported (the tests do); setuptools and pip run it as __main__
    main()
