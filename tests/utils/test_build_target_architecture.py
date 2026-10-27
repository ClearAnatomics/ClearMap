"""
The CPU the compiled code (Cython) is built for is chosen with ``CLEARMAP_TARGET_ARCHITECTURE``.

The default, ``native``, is the CPU of the machine that builds: fastest, but the code only runs on CPUs at least as
recent. A build for several machines (a module for a cluster, a wheel) names a level they all have, e.g. ``x86-64-v3``.
The former ``OPTIMISE_CLEARMAP_COMPILATION_FOR_TARGET`` still works, but only the exact string ``False`` used to turn
it off, so ``0``, ``false`` or ``no`` silently left the build native.
"""
import pytest

TARGET = 'CLEARMAP_TARGET_ARCHITECTURE'
LEGACY = 'OPTIMISE_CLEARMAP_COMPILATION_FOR_TARGET'
SPELLINGS_OF_FALSE = ('False', 'false', 'FALSE', '0', 'no', 'No', 'off', 'OFF')
SPELLINGS_OF_TRUE = ('True', 'true', '1', 'yes', 'on', '')
NATIVE_ARGUMENTS = ['-march=native', '-mtune=native']


def test_the_default_is_the_cpu_of_the_machine_that_builds(setup_script):
    assert setup_script.target_compile_arguments({}) == NATIVE_ARGUMENTS


@pytest.mark.parametrize('spelling', SPELLINGS_OF_FALSE)
def test_the_former_variable_turns_the_optimisation_off_whatever_the_spelling_of_false(setup_script, spelling):
    assert setup_script.target_compile_arguments({LEGACY: spelling}) == []


@pytest.mark.parametrize('spelling', SPELLINGS_OF_TRUE)
def test_the_former_variable_keeps_the_native_build_for_anything_else(setup_script, spelling):
    assert setup_script.target_compile_arguments({LEGACY: spelling}) == NATIVE_ARGUMENTS


def test_a_named_level_is_used_as_it_is_and_is_not_tuned_for_the_machine_that_builds(setup_script):
    assert setup_script.target_compile_arguments({TARGET: 'x86-64-v3'}) == ['-march=x86-64-v3']


def test_native_can_be_asked_for_by_name(setup_script):
    assert setup_script.target_compile_arguments({TARGET: 'native'}) == NATIVE_ARGUMENTS


@pytest.mark.parametrize('spelling', ('Native', 'X86-64-V3'))
def test_the_case_of_the_name_does_not_matter(setup_script, spelling):
    assert setup_script.target_compile_arguments({TARGET: spelling}) == \
           setup_script.target_compile_arguments({TARGET: spelling.lower()})


def test_the_new_variable_wins_over_the_former_one(setup_script):
    assert setup_script.target_compile_arguments({TARGET: 'x86-64-v2', LEGACY: 'False'}) == ['-march=x86-64-v2']


def test_the_compiler_default_adds_nothing(setup_script):
    assert setup_script.target_compile_arguments({TARGET: 'compiler-default'}) == []


@pytest.mark.parametrize('not_a_name', ('native -O0', 'x86-64-v3;rm -rf', '-march=native', 'a b', '$(id)'))
def test_a_value_that_is_not_an_architecture_name_is_refused_before_it_reaches_the_compiler(setup_script, not_a_name):
    with pytest.raises(ValueError, match=TARGET):
        setup_script.target_compile_arguments({TARGET: not_a_name})


@pytest.fixture
def requested_level(monkeypatch):
    monkeypatch.setenv(TARGET, 'x86-64-v3')


@pytest.mark.parametrize('system', ('linux', 'darwin'))
def test_the_compiler_arguments_of_unix_like_systems_ask_for_the_level(setup_script, monkeypatch, requested_level, system):
    monkeypatch.setattr(setup_script, 'os_name', system)
    assert '-march=x86-64-v3' in setup_script.default_compile_arguments()


def test_msvc_has_no_march(setup_script, monkeypatch, requested_level):
    monkeypatch.setattr(setup_script, 'os_name', 'windows')
    assert not [argument for argument in setup_script.default_compile_arguments() if argument.startswith('-m')]
