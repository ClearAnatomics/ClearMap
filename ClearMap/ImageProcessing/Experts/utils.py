import ClearMap.IO.IO as clearmap_io

from ClearMap.ParallelProcessing.DataProcessing import ArrayProcessing as ap
import ClearMap.Utils.Timer as tmr
import ClearMap.Utils.HierarchicalDict as hdict


def initialize_sinks(cell_detection_parameter, shape, order):
    for key in cell_detection_parameter.keys():
        par = cell_detection_parameter[key]
        if isinstance(par, dict):
            filename = par.get('save')
            if filename:
                dtype = par.get('save_dtype')
                if dtype is None:
                    dtype='float'
                ap.initialize_sink(filename, shape=shape, order=order, dtype=dtype)


def print_params(step_params, param_key, prefix, verbose):
    step_params = step_params.copy()
    if verbose:
        timer = tmr.Timer(prefix)
        head = f'{prefix}{param_key.replace("_", " ").title()}:'
        hdict.pprint(step_params, head=head)
        return step_params, timer
    return step_params, None


def run_step(param_key, previous_result, step_function, args=(), remove_previous_result=False,
             presave_parser=None, extra_kwargs=None, parameter=None, steps_to_measure=None, prefix='',
             base_slicing=None, valid_slicing=None):
    if steps_to_measure is None:
        steps_to_measure = {}
    if parameter is None:
        parameter = {}
    if extra_kwargs is None:
        extra_kwargs = {}
    step_param = parameter.get(param_key)
    if step_param:
        step_param, timer = print_params(step_param, param_key, prefix, parameter.get('verbose'))
        print("step_param:",step_param)
        save = step_param.pop('save', None)  # FIXME: check if always goes before step_function call
        save_dtype = step_param.pop('save_dtype', None)  
        result = step_function(previous_result, *args, **{**step_param, **extra_kwargs})

        if save:
            if save_dtype is None:
                save = clearmap_io.as_source(save)
            else:                    
                save = clearmap_io.as_source(save,dtype=save_dtype)

            to_save=presave_parser(result)
            # usefull sanity check
            if save_dtype=='bool':
                save[base_slicing] = (to_save[valid_slicing] > 0)
            else:
                save[base_slicing] = to_save[valid_slicing]

        if parameter.get('verbose'):
            timer.print_elapsed_time(param_key.title())
    else:
        result = previous_result
    if remove_previous_result:
        del previous_result
    if param_key in steps_to_measure:
        steps_to_measure[param_key] = result
    return result
