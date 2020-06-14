#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
"""
HessianCode
===========

Cython code for Hessian eigenvalue calculation.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

from numpy cimport uint8_t, uint16_t, float32_t, double_t

ctypedef fused source_t:
    uint8_t
    uint16_t
    float32_t
    double_t

ctypedef fused sink_t:
    uint8_t
    uint16_t
    float32_t
    double_t

ctypedef Py_ssize_t index_t


from libc.math cimport sqrt, atan2, cos, sin, pow
    
#nogil version of abs
cdef inline double _abs(double a) nogil:
    return a if a >= 0 else -a;

#debug
#cdef extern from "stdio.h":
#    int printf(char *format, ...) nogil                     


def hessian(source_t[:, :, :] source, sink_t[:, :, :, :, :] sink, index_t sink_stride, double[:] parameter):
  """Compute Hessian eigenvalues at each pixel."""
  # array sizes
  cdef index_t nx = source.shape[0]
  cdef index_t ny = source.shape[1]
  cdef index_t nz = source.shape[2]   
  
  cdef index_t max_img = nx * ny * nz

  # local variable types
  cdef index_t x,y,z,xm,ym,zm,xp,yp,zp
  
  with nogil:
    for x in range(nx):
      xm = x - 1 if x > 0 else x;
      xp = x + 1 if x < nx - 1 else nx -1;
      
      for y in range(ny):
        ym = y - 1 if y > 0 else y;
        yp = y + 1 if y < ny - 1 else ny - 1;
        
        for z in range(nz):
          zm = z - 1 if z > 0 else z;
          zp = z + 1 if z < nz - 1 else nz - 1;
          
          # create hessian
          i = 2.0 * <double>source[x,y,z];
          
          sink[x,y,z,0,0] = <sink_t> (source[xm,y, z ] - i + source[xp,y, z ]);
          sink[x,z,y,1,1] = <sink_t> (source[x, ym,z ] - i + source[x, yp,z ]);
          sink[x,y,z,2,2] = <sink_t> (source[x, y, zm] - i + source[x, y ,zp]);

          sink[x,y,z,0,1] = sink[x,y,z,1,0] = <sink_t> ((<double>source[xp,yp,z ] - source[xm,yp,z ] - source[xp,ym,z ] + source[xm,ym,z ]) / 4.0);
          sink[x,y,z,0,2] = sink[x,y,z,2,0] = <sink_t> ((<double>source[xp,y ,zp] - source[xm,y ,zp] - source[xp,y ,zm] + source[xm,y ,zm]) / 4.0);
          sink[x,y,z,1,2] = sink[x,y,z,2,1] = <sink_t> ((<double>source[x ,yp,zp] - source[x ,ym,zp] - source[x ,yp,zm] + source[x ,ym,zm]) / 4.0);


cdef void hessian_core(void kernel(sink_t*, index_t, double, double, double, double*) nogil,
                       source_t[:, :, :] source, sink_t[:, :, :, :] sink, index_t sink_stride, double[:] parameter) except *:
  """Compute Hessian eigenvalues for each pixel and apply a measure defined by the kernel."""
  
  # array sizes
  cdef index_t nx = source.shape[0]
  cdef index_t ny = source.shape[1]
  cdef index_t nz = source.shape[2]   
  
  cdef index_t max_img = nx * ny * nz
  
  #cdef index_t odepth = sink.shape[3]
  
  # local variable types
  cdef index_t x,y,z,xm,ym,zm,xp,yp,zp
  
  cdef double one_third = 1.0/3.0;
  cdef double root_three = 1.7320508075688772935;

  # Hessian matrix is represented as A = h[0,0], D = h[1,1], F = h[2,2], B = h[0,1]=h[1,0], C = h[0,2]=h[2,0], E = h[1,2]=h[2,1]
  cdef double A, B, C, D, E, F;
  
  
  # Compute eigenvalues and apply kernel
  cdef double i, discriminant, a, b, c, d, q, r, s, t, u, v
  cdef double e1, e2, e3
  cdef double e1a, e2a, e3a
  cdef double e1s, e2s, e3s
  a = -1.0;
   
  with nogil:
    for x in range(nx):
      xm = x - 1 if x > 0 else x;
      xp = x + 1 if x < nx - 1 else nx -1;
      
      for y in range(ny):
        
        ym = y - 1 if y > 0 else y;
        yp = y + 1 if y < ny - 1 else ny - 1;
        
        for z in range(nz):
          zm = z - 1 if z > 0 else z;
          zp = z + 1 if z < nz - 1 else nz - 1;
          
          # create hessian
          i = 2.0 * <double>source[x,y,z];
          
          A = source[xm,y, z ] - i + source[xp,y, z ];
          D = source[x, ym,z ] - i + source[x, yp,z ];
          F = source[x, y, zm] - i + source[x, y ,zp];

          B = (<double>source[xp,yp,z ] - source[xm,yp,z ] - source[xp,ym,z ] + source[xm,ym,z ]) / 4.0;
          C = (<double>source[xp,y ,zp] - source[xm,y ,zp] - source[xp,y ,zm] + source[xm,y ,zm]) / 4.0;
          E = (<double>source[x ,yp,zp] - source[x ,ym,zp] - source[x ,yp,zm] + source[x ,ym,zm]) / 4.0;
 
          # calculate eigenvalues
          b = A + D + F;
          c = B * B + C * C + E * E - A * D - A * F - D * F;
          d = A * D * F - A * E * E - B * B * F + 2 * B * C * E - C * C * D;
          
          q = (3*a*c - b*b) / (9*a*a);
          r = (9*a*b*c - 27*a*a*d - 2*b*b*b) / (54*a*a*a);
          
          discriminant = q*q*q + r*r;

          if discriminant < 0:
            s = pow(sqrt(r*r - discriminant), one_third);          
            t = atan2(sqrt(-discriminant), r) / 3.0;

            u = 2 * s * cos(t);
            first = - (u / 2) - (b / 3*a);
            last  = - root_three * s * sin(t);
            
            e1 = ( u - (b / (3*a)) );
            e2 = ( first + last );
            e3 = ( first - last );
            
          elif discriminant == 0:
            
            if r >= 0:
              u = 2 * pow(r,one_third);
            else:                  
              u = -2 * pow(-r,one_third);
            v = b / (3 * a);
  
            e1 = ( u   - v );
            e2 = (-u/2 - v );
            e3 = e2;
          
          else:
            # discriminant > 0 should not happen for a symmetric matrix
            e1 = e2 = e3 = 0.0;
 
          # sort eigenvalues es1 >= es2 >= es3
          # e1 = abs(e1); e2 = abs(e2); e3 = abs(e3);
          if e1 <= e2:
            if e2 <= e3 :
              e3s = e1; e2s = e2; e1s = e3;
            else:
              if e1 <= e3:
                e3s = e1; e2s = e3; e1s = e2;
              else:
                e3s = e3; e2s = e1; e1s = e2;
          else: #e2 < e1
            if e1 <= e3:
              e3s = e2; e2s = e1; e1s = e3;
            else:
              if e2 <= e3:
                e3s = e2; e2s = e3; e1s = e1;
              else:
                e3s = e3; e2s = e2; e1s = e1;
          
          # apply kernel
          kernel(&sink[x, y, z, 0], sink_stride, e1s, e2s, e3s, &parameter[0]);


#Hessina eigenvalues
cdef inline void eigenvalue_kernel(sink_t* sink, index_t sink_stride, double e1, double e2, double e3, double* par) nogil:
  sink[0              ] = <sink_t> e1; 
  sink[1 * sink_stride] = <sink_t> e2;
  sink[2 * sink_stride] = <sink_t> e3;


#Tubness part of a Frangi filter, i.e. the geometric mean of lowest two eigenvalues
cdef inline void tubeness_kernel(sink_t* sink, index_t sink_stride, double e1, double e2, double e3, double* par) nogil:
  if e2 < 0 and e3 < 0:
    sink[0] = <sink_t> sqrt(e2 * e3);
  else:
    sink[0] = 0;
        
        
#Thresholded Tubness part of a Frangi filter
cdef inline void tubeness_threshold_kernel(sink_t* sink, index_t sink_stride, double e1, double e2, double e3, double* par) nogil:
  if e2 < 0 and e3 < 0:
    if sqrt(e2 * e3) > par[0]:
      sink[0] = 1;
    #else:
    #  sink[0] = 0;
  #else:
  #  sink[0] = 0;


#Generalized Frangi filer [Sato et al, Three dimensional multi-scale line filter for segmentation and visualization of curvilinear structures in medicalimages, 1998]
cdef inline void lambda123_kernel(sink_t* sink, index_t sink_stride, double e1, double e2, double e3, double* par) nogil:
  cdef double a;
  
  if e2 < 0 and e3 < 0:
    if e1 <= 0:
      sink[0] = <sink_t> (_abs(e3) * pow((e2/e3), par[0]) * pow(1 + e1/_abs(e2), par[1]));
    else:
      a = par[2] * e1 / _abs(e2);
      if a < 1:                         
        sink[0] = <sink_t> (_abs(e3) * pow((e2/e3), par[0]) * pow(1 - a, par[1]));
      else:
        sink[0] = 0;
  else:
    sink[0] = 0;


#Generalized Frangi filer [Sato et al, Three dimensional multi-scale line filter for segmentation and visualization of curvilinear structures in medicalimages, 1998]
cdef inline void lambda123_threshold_kernel(sink_t* sink, index_t sink_stride, double e1, double e2, double e3, double* par) nogil:
  cdef double a;
  
  if e2 < 0 and e3 < 0:
    if e1 <= 0 and (_abs(e3) * pow((e2/e3), par[0]) * pow(1 + e1/_abs(e2), par[1])) >= par[3]:
      sink[0] = 1;
    else:
      a = par[2] * e1 / _abs(e2);
      if a < 1 and (_abs(e3) * pow((e2/e3), par[0]) * pow(1 - a, par[1])) >= par[3]:
        sink[0] = 1;
      #else:
      #  sink[0] = 0;
  #else:
  #  sink[0] = 0;



def eigenvalues(source_t[:, :, :] source, sink_t[:, :, :, :] sink, index_t sink_stride, double[:] parameter):
  hessian_core(eigenvalue_kernel[sink_t], source, sink, sink_stride, parameter);


def tubeness(source_t[:, :, :] source, sink_t[:, :, :, :] sink, index_t sink_stride, double[:] parameter):
  hessian_core(tubeness_kernel[sink_t], source, sink, sink_stride, parameter);

  
def tubeness_threshold(source_t[:, :, :] source, sink_t[:, :, :, :] sink, index_t sink_stride, double[:] parameter):
  hessian_core(tubeness_threshold_kernel[sink_t], source, sink, sink_stride, parameter);
                

def lambda123(source_t[:, :, :] source, sink_t[:, :, :, :] sink, index_t sink_stride, double[:] parameter):
  hessian_core(lambda123_kernel[sink_t], source, sink, sink_stride, parameter);


def lambda123_threshold(source_t[:, :, :] source, sink_t[:, :, :, :] sink, index_t sink_stride, double[:] parameter):
  hessian_core(lambda123_threshold_kernel[sink_t], source, sink, sink_stride, parameter);

