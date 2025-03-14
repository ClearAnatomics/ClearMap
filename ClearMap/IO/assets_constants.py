from copy import deepcopy

EXTENSIONS = {
    'image': ['.npy', '.tif', '.tiff'],
    'portable_image': ['.tif', '.tiff'],
    'graph': ['.gt'],
    'layout': ['.lyt'],
    'elastix_image': ['.mhd'],
    'elastix_points': ['.pts'],
    'table': ['.npy', '.feather', '.csv'],
    'log': ['.log'],
    'error_log': ['.html'],
}

COMPRESSION_ALGORITHMS = ('gzip', 'bz2', 'zip', 'lzma')
# OPTIMISE: add support for pbzip2, pigz, plzip

CHECKSUM_ALGORITHMS = ('md5', 'sha1', 'sha256', 'sha512')


"""
Mapping of resource types to the folder where they are stored.

.. warning::
    Folders are relative to the root folder of the sample by default
    but can be absolute paths in which case they are used as is,
    ignoring the root folder of the sample.
"""
RESOURCE_TYPE_TO_FOLDER = {
    'logs': '',  # in main folder
    'config_snapshots': 'config_snapshots',  # TODO: use
    'data': 'data',
    'processed': 'data',   # TODO: see if we split
    'results': 'analyzed',
    'graphs': 'graphs',  # TODO: check if analyzed/graphs
    'elastix': '',  # in main folder
    'atlas': 'atlas',
}

"""
Mapping of content types (i.e. intrinsic labels) 
to the pipeline(s) that are relevant for that content type.
"""
CONTENT_TYPE_TO_PIPELINE = {
    None: None,  # not configured
    'no-pipeline': None,  # configured but no pipeline
    'autofluorescence': 'registration',
    'nuclei': 'CellMap',  # TODO: list ?
    'cells': 'CellMap',
    'vessels': 'TubeMap',
    'veins': 'TubeMap',
    'arteries': 'TubeMap',
    'axons': 'AxonMap',
    'atlas': 'registration',
    'myelin': 'TractMap'
}
DATA_CONTENT_TYPES = list(CONTENT_TYPE_TO_PIPELINE.keys())

# TODO: add link between labels and content types
#    e.g.
#    {'cfos': 'nuclei',
#     'dapi': 'nuclei',
#     'autofluorescence': 'reference',
#     'None': 'reference',
#     }

"""
Assets that are channel specific.
They may or may not make sense for a given channel 
based on the pipeline(s) relevant for that channel.
"""
CHANNELS_ASSETS_TYPES_CONFIG = {
    'raw': {
        'file_format_category': 'image',
        'resource_type': 'data',
        'relevant_pipelines': ['all']
    },
    'stitched': {
        'file_format_category': 'image',
        'resource_type': 'processed',
        'relevant_pipelines': ['stitching']
    },
    'layout': {
        'file_format_category': 'layout',
        'resource_type': 'processed',
        'relevant_pipelines': ['stitching'],
        'sub_types': ['aligned', 'aligned_axis', 'placed']
    },
    'background': {
        'file_format_category': 'image',
        'resource_type': 'processed',
        'relevant_pipelines': ['CellMap', 'TubeMap', 'AxonMap']
    },
    'resampled': {
        'file_format_category': 'portable_image',
        'resource_type': 'processed',
        'relevant_pipelines': ['registration']
    },
    'aligned': {  # WARNING: no sample_id
        'file_format_category': 'elastix_image',
        'resource_type': 'elastix',
        'relevant_pipelines': ['registration'],
        'basename': '<moving_channel,S>_to_<fixed_channel,S>/result.<N,1>'  # WARNING: elastix always names the result as result.<N,1>
    },
    'fixed_landmarks': {  # WARNING: no sample_id
        'file_format_category': 'elastix_points',
        'resource_type': 'elastix',
        'relevant_pipelines': ['registration'],
        'basename': '<moving_channel,S>_to_<fixed_channel,S>/<fixed_channel,S>_landmarks'
    },
    'moving_landmarks': {  # WARNING: no sample_id
        'file_format_category': 'elastix_points',
        'resource_type': 'elastix',
        'relevant_pipelines': ['registration'],
        'basename': '<moving_channel,S>_to_<fixed_channel,S>/<moving_channel,S>_landmarks'
    },
    'cells': {
        'file_format_category': 'table',
        'resource_type': 'results',
        'relevant_pipelines': ['CellMap', 'TractMap'],  # TODO: tune relevance for subsets
        'sub_types': ['raw', 'filtered', 'shape']
    },
    'cells_stats': {
        'file_format_category': 'table',
        'resource_type': 'results',
        'relevant_pipelines': ['CellMap'],
        'extensions': ['.csv', '.feather']
    },
    'density': {
        'file_format_category': 'portable_image',
        'resource_type': 'results',
        'relevant_pipelines': ['CellMap', 'TubeMap', 'AxonMap', 'TractMap'],
        'sub_types': ['counts', 'intensities', 'branches']  # WARNING: different subtypes as f(pipeline)
    },
    'binary': {
        'file_format_category': 'image',
        'resource_type': 'results',
        'relevant_pipelines': ['TubeMap', 'AxonMap', 'TractMap'],
        'sub_types': ['status', 'vesselize', 'median', 'pixels_raw', 'coordinates_transformed', 'coordinates_raw', 'labels']
    },
    'skeleton': {
        'file_format_category': 'image',
        'resource_type': 'results',
        'relevant_pipelines': ['TubeMap', 'AxonMap']
    },
    'graph': {
        'file_format_category': 'graph',
        'resource_type': 'graphs',
        'relevant_pipelines': ['TubeMap', 'AxonMap'],
        'sub_types': ['raw', 'cleaned', 'reduced', 'annotated']
    },
}


"""
Assets that are not channel specific but global per sample.
"""
GLOBAL_ASSETS_TYPES_CONFIG = {
    'info': {
        'file_format_category': 'log',
        'resource_type': 'logs',
        'relevant_pipelines': ['all']
    },
    'progress': {
        'file_format_category': 'log',
        'resource_type': 'logs',
        'relevant_pipelines': ['all']
    },
    'error': {
        'file_format_category': 'error_log',
        'resource_type': 'logs',
        'relevant_pipelines': ['all']
    },
}
