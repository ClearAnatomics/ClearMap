"""
run
===

Headless job runner: runs a pipeline on an experiment folder, without a display, without Qt and without a prompt,
for a cluster job (or any batch run).

Usage
-----

.. code-block:: bash

    python -m ClearMap.run tube_map /path/to/experiment
    python -m ClearMap.run cell_map /path/to/experiment --check-only

The experiment folder holds the configuration of the sample (as for the GUI and the ``*_new_api`` scripts), and the
results are written in it.  A Slurm job script is in ``ClearMap/Scripts/cluster/clearmap_job.sbatch``.

Temporary folder
----------------
ClearMap writes hundreds of GB of temporary files, so the run needs a folder for them: ``CLEARMAP_TMP`` (the folder of
this job), else the ``temp_folder`` of the machine config. There is no default: without one, or if it is not writeable
or has too little room, the run stops before it starts (also with ``--check-only``) and says how to set it, instead of
filling the system ``/tmp``.  The folder and its free space are printed at the start, for the job log.

Exit status
-----------
``ExitCode.SUCCESS`` when the pipeline ran (or, with ``--check-only``, when the experiment folder exists),
``ExitCode.FAILURE`` when it failed (the message and the traceback are on stderr, for the job log),
``ExitCode.USAGE_ERROR`` for a wrong command line (argparse).
"""
import argparse
import importlib
import shutil
import sys
import traceback
from pathlib import Path

from ClearMap.config.early_boot import MachineConfig
from ClearMap.Utils.exceptions import ClearMapFileNotFoundError
from ClearMap.Utils.utilities import ExitCode

#: pipeline name -> module (of ClearMap.Scripts) whose main(src_directory, show_plots) runs it.
#: The modules are imported when the pipeline is run, so that listing them does not import the pipelines.
PIPELINE_MODULES = {
    'align': 'ClearMap.Scripts.align_new_api',
    'tube_map': 'ClearMap.Scripts.tube_map_new_api',
    'cell_map': 'ClearMap.Scripts.cell_map_new_api',
}


def build_argument_parser():
    parser = argparse.ArgumentParser(prog='python -m ClearMap.run', description=__doc__.split('\n\n')[1])
    parser.add_argument('pipeline', choices=sorted(PIPELINE_MODULES), help='the pipeline to run')
    parser.add_argument('experiment_folder', type=Path, help='the folder of the experiment (with its configuration)')
    parser.add_argument('--check-only', action='store_true',
                        help='check the experiment folder and the pipeline, but do not run it')
    return parser


def check_experiment_folder(experiment_folder):
    """The experiment folder, resolved. Raises ClearMapFileNotFoundError if it is not a folder."""
    resolved_folder = Path(experiment_folder).expanduser().resolve()
    if not resolved_folder.is_dir():
        raise ClearMapFileNotFoundError(f'The experiment folder "{resolved_folder}" does not exist '
                                        f'(or is not a folder).')
    return resolved_folder


def patch_temporary_folder():
    """
    Set the temporary folder of the run (see the module docstring) and print it, with its free space.

    It is done before the pipelines are imported: modules must not read ``tempfile.tempdir`` before it is set.

    Raises
    ------
    ClearMapConfigError
        If there is no valid temporary folder.
    """
    temporary_folder = MachineConfig().patch_tmp(allow_prompt=False)  # Nobody to answer on a compute node
    free_gb = shutil.disk_usage(temporary_folder).free / 1024 ** 3
    print(f'Temporary folder: {temporary_folder} ({free_gb:.0f} GB free)')
    return temporary_folder


def run_pipeline(pipeline_name, experiment_folder):
    """Run the pipeline without any plot (which would need a display) on the experiment folder."""
    pipeline_module = importlib.import_module(PIPELINE_MODULES[pipeline_name])
    pipeline_module.main(str(experiment_folder), show_plots=False)


def main(arguments=None):
    parsed_arguments = build_argument_parser().parse_args(arguments)
    try:
        experiment_folder = check_experiment_folder(parsed_arguments.experiment_folder)
        patch_temporary_folder()
        if parsed_arguments.check_only:
            print(f'{parsed_arguments.pipeline} on {experiment_folder}: ready to run')
            return ExitCode.SUCCESS
        run_pipeline(parsed_arguments.pipeline, experiment_folder)
    except Exception as error:  # the boundary of the job: whatever failed goes to the job log, with a status
        traceback.print_exc(file=sys.stderr)
        print(f'{parsed_arguments.pipeline} failed: {error}', file=sys.stderr)
        return ExitCode.FAILURE
    return ExitCode.SUCCESS


if __name__ == '__main__':
    sys.exit(main())
