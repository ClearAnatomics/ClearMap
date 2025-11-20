import sys

from ClearMap.processors.sample_preparation import SampleManager
from ClearMap.processors.degree1_inference_processor import degree1_verification


def main(sample_directory=None, graph_step='reduced'):
    """
    Test_ the degree 1 inference processor on a CleaRMap v3 sample folder.
    This assumes that the sample folder contains a graph in the 'reduced' step from
    channel 'vasc' and that the source image channel for the inference is 'vasc'.

    Parameters
    ----------
    sample_directory: str or None
        The path to the sample directory. If None, it will take the first argument from sys.argv.
        Default is None.
    """
    if sample_directory is None:
        sample_directory = sys.argv[1]
    sample_manager = SampleManager()
    sample_manager.setup(src_dir=sample_directory)
    degree1_verification(sample_manager, graph_step=graph_step)


if __name__ == '__main__':
    #sys.path.append(str(Path().resolve().parents[3]))
    #print(Path().resolve().parents[3])
    main(sys.argv[1])