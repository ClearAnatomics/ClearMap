"""
Set of classes that each focus on a specific task in the ClearMap pipeline.

The processors are the main building blocks of the ClearMap pipeline.
The base class from which all processors inherit is :class:`TabProcessor`.
Then there are 2 types of derived classes, preprocessor and the ones using a preprocessor.

Note batch_process is not a processor but a set of convenience functions to run aspects of
 the pipeline on a group of samples.
"""
