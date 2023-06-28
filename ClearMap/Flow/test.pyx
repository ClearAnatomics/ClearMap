#from __future__ import division

cdef extern from "math.h":
    double sin(double)
    double cos(double)
    double exp(double)
    double sqrt(double)
    double M_PI

# import g_math
# import units
import numpy