"""
The Slurm job script gives the run a temporary folder of the job: CLEARMAP_TMP, else TMPDIR, and never a default of
/tmp (ClearMap writes hundreds of GB there). conda and srun are replaced by stubs: the script is run for its environment.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

JOB_SCRIPT = Path(__file__).parents[2] / 'ClearMap' / 'Scripts' / 'cluster' / 'clearmap_job.sbatch'


@pytest.fixture
def stubs_folder(tmp_path):
    """A PATH with a conda that has no environment to activate, and a srun that prints the temporary folder it got"""
    conda_base = tmp_path / 'conda_base'
    (conda_base / 'etc' / 'profile.d').mkdir(parents=True)
    (conda_base / 'etc' / 'profile.d' / 'conda.sh').write_text('conda() { :; }\n')
    stubs = tmp_path / 'stubs'
    stubs.mkdir()
    conda = stubs / 'conda'
    conda.write_text(f'#!/bin/sh\necho {conda_base}\n')
    srun = stubs / 'srun'
    srun.write_text('#!/bin/sh\necho "SRUN_CLEARMAP_TMP=$CLEARMAP_TMP"\n')
    for stub in (conda, srun):
        stub.chmod(0o755)
    (stubs / 'bash').symlink_to(shutil.which('bash'))
    return stubs


def run_job_script(stubs_folder, **environment):
    env = {'PATH': f'{stubs_folder}:/usr/bin:/bin', 'SLURM_CPUS_PER_TASK': '4',
           'CLEARMAP_PIPELINE': 'tube_map', 'CLEARMAP_EXPERIMENT': '/experiment', **environment}
    return subprocess.run([shutil.which('bash'), str(JOB_SCRIPT)], env=env, capture_output=True, text=True, check=False)


def test_without_a_temporary_folder_the_script_stops_before_running_anything(stubs_folder):
    result = run_job_script(stubs_folder)

    assert result.returncode != 0
    assert 'CLEARMAP_TMP' in result.stderr
    assert 'SRUN_CLEARMAP_TMP' not in result.stdout


def test_the_tmpdir_of_the_job_is_the_temporary_folder(stubs_folder):
    result = run_job_script(stubs_folder, TMPDIR='/local/job')

    assert result.returncode == 0, result.stderr
    assert 'SRUN_CLEARMAP_TMP=/local/job' in result.stdout


def test_clearmap_tmp_comes_before_the_tmpdir_of_the_job(stubs_folder):
    result = run_job_script(stubs_folder, TMPDIR='/local/job', CLEARMAP_TMP='/scratch/me')

    assert result.returncode == 0, result.stderr
    assert 'SRUN_CLEARMAP_TMP=/scratch/me' in result.stdout
