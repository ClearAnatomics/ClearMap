#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
"""
HessianCode
===========

Cython code for Hessian eigenvalue calculation.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'https://idisco.info'
__download__  = 'https://www.github.com/ChristophKirst/ClearMap2'

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


from libc.math cimport sqrt, cos, pow, acos, M_PI, fabs
    
# nogil version of abs
cdef inline double _abs(double a) nogil:
    return a if a >= 0 else -a

# debug
# cdef extern from "stdio.h":


def hessian(source_t[:, :, :] source, sink_t[:, :, :, :, :] sink, index_t sink_stride, double[:] parameter):
  """Compute Hessian eigenvalues at each pixel."""
  # array sizes
  cdef index_t nx = source.shape[0]
  cdef index_t ny = source.shape[1]
  cdef index_t nz = source.shape[2]   

  # local variable types
  cdef index_t x,y,z,xm,ym,zm,xp,yp,zp
  cdef double s;
  
  with nogil:
    for x in range(nx):
      xm = x - 1 if x > 0 else x
      xp = x + 1 if x < nx - 1 else nx -1

      for y in range(ny):
        ym = y - 1 if y > 0 else y
        yp = y + 1 if y < ny - 1 else ny - 1

        for z in range(nz):
          zm = z - 1 if z > 0 else z
          zp = z + 1 if z < nz - 1 else nz - 1

          # create hessian
          s = 2.0 * (<double>source[x,y,z])

          sink[x,y,z,0,0] = <sink_t> (source[xm,y, z ] - s + source[xp,y, z ])
          sink[x,y,z,1,1] = <sink_t> (source[x, ym,z ] - s + source[x, yp,z ])
          sink[x,y,z,2,2] = <sink_t> (source[x, y, zm] - s + source[x, y ,zp])

          sink[x,y,z,0,1] = sink[x,y,z,1,0] = <sink_t> ((<double>source[xp,yp,z ] - source[xm,yp,z ] - source[xp,ym,z ] + source[xm,ym,z ]) / 4.0)
          sink[x,y,z,0,2] = sink[x,y,z,2,0] = <sink_t> ((<double>source[xp,y ,zp] - source[xm,y ,zp] - source[xp,y ,zm] + source[xm,y ,zm]) / 4.0)
          sink[x,y,z,1,2] = sink[x,y,z,2,1] = <sink_t> ((<double>source[x ,yp,zp] - source[x ,ym,zp] - source[x ,yp,zm] + source[x ,ym,zm]) / 4.0)

cdef void hessian_eigenvalue_core(void kernel(sink_t*, index_t, double, double, double, double*) nogil,
                                  source_t[:, :, :] source, sink_t[:, :, :, :] sink, index_t sink_stride, double[:] parameter) except *:
  """Compute Hessian eigenvalues for each pixel and apply a measure defined by the kernel."""
  #eigenvalues via affine change of A (see https://en.wikipedia.org/wiki/Eigenvalue_algorithm 3x3 matrices)
  
  # array sizes
  cdef index_t nx = source.shape[0]
  cdef index_t ny = source.shape[1]
  cdef index_t nz = source.shape[2]   
  
  #cdef index_t odepth = sink.shape[3]
  
  # local variable types
  cdef index_t x,y,z,xm,ym,zm,xp,yp,zp
 
  # Hessian matrix is represented as A = h[0,0], D = h[1,1], F = h[2,2], B = h[0,1]=h[1,0], C = h[0,2]=h[2,0], E = h[1,2]=h[2,1]
  cdef double A, B, C, D, E, F;
  
  
  # Compute eigenvalues and apply kernel
  cdef double s, p1, p2, p, r, phi, pi_3, pi_2
  cdef double e1, e2, e3
  
  pi_3 = M_PI / 3
  pi_3_2 = 2 * pi_3

  with nogil:
    for x in range(nx):
      xm = x - 1 if x > 0 else x
      xp = x + 1 if x < nx - 1 else nx -1

      for y in range(ny):
        
        ym = y - 1 if y > 0 else y
        yp = y + 1 if y < ny - 1 else ny - 1

        for z in range(nz):
          zm = z - 1 if z > 0 else z
          zp = z + 1 if z < nz - 1 else nz - 1

          # create hessian
          s = 2.0 * <double>source[x,y,z]

          A = source[xm,y, z ] - s + source[xp,y, z ]
          D = source[x, ym,z ] - s + source[x, yp,z ]
          F = source[x, y, zm] - s + source[x, y ,zp]

          B = (<double>source[xp,yp,z ] - source[xm,yp,z ] - source[xp,ym,z ] + source[xm,ym,z ]) / 4.0
          C = (<double>source[xp,y ,zp] - source[xm,y ,zp] - source[xp,y ,zm] + source[xm,y ,zm]) / 4.0
          E = (<double>source[x ,yp,zp] - source[x ,ym,zp] - source[x ,yp,zm] + source[x ,ym,zm]) / 4.0

          # calculate eigenvalues
          p1 = B * B + C * C + E * E
          if p1 == 0:
            # eigen values are A, D, F
            # sort e1 >= e2 >= e3
            if A <= D:
              if D <= F :
                e3 = A; e2 = D; e1 = F
              else: # D > F
                if A <= F:
                  e3 = A; e2 = F; e1 = D
                else: # A > F
                  e3 = F; e2 = A; e1 = D
            else: #D < A
              if A <= F:
                e3 = D; e2 = A; e1 = F
              else: # A > F
                if D <= F:
                  e3 = D; e2 = F; e1 = A
                else: # D > F
                  e3 = F; e2 = D; e1 = A

          else: 
            q  = (A + D + F) / 3
            p2 = (A - q) * (A - q) + (D - q) * (D - q) + (F - q) * (F - q) + 2 * p1
            p  = sqrt(p2 / 6)

            # affine matrix A = pB + qI
            A = (A - q)
            D = (D - q)
            F = (F - q)

            # det B  / 2
            r = (A * D * F + 2 * B * C * E - A * E * E - B * B * F - C * C * D) / 2 / (p * p * p)

            if r <= -1:
              phi = pi_3
            elif r >= 1:
              phi = 0
            else:
              phi = acos(r) / 3

            e1 = q + 2 * p * cos(phi)
            e3 = q + 2 * p * cos(phi + pi_3_2)
            e2 = 3 * q - e1 - e3

          # apply kernel
          kernel(&sink[x, y, z, 0], sink_stride, e1, e2, e3, &parameter[0])

#Hessian eigenvalues
cdef inline void eigenvalue_kernel(sink_t* sink, index_t sink_stride, double e1, double e2, double e3, double* par) nogil:
  sink[0              ] = <sink_t> e1
  sink[1 * sink_stride] = <sink_t> e2
  sink[2 * sink_stride] = <sink_t> e3

#Tubness part of a Frangi filter, i.e. the geometric mean of lowest two eigenvalues
cdef inline void tubeness_kernel(sink_t* sink, index_t sink_stride, double e1, double e2, double e3, double* par) nogil:
  if e2 < 0 and e3 < 0:
    sink[0] = <sink_t> sqrt(e2 * e3)
  else:
    sink[0] = 0

#Thresholded Tubness part of a Frangi filter
cdef inline void tubeness_threshold_kernel(sink_t* sink, index_t sink_stride, double e1, double e2, double e3, double* par) nogil:
  if e2 < 0 and e3 < 0:
    if sqrt(e2 * e3) > par[0]:
      sink[0] = 1
    #else:
    #  sink[0] = 0;
  #else:
  #  sink[0] = 0;


#Generalized Frangi filer [Sato et al, Three dimensional multi-scale line filter for segmentation and visualization of curvilinear structures in medicalimages, 1998]
cdef inline void lambda123_kernel(sink_t* sink, index_t sink_stride, double e1, double e2, double e3, double* par) nogil:
  cdef double a;
  
  if e2 < 0 and e3 < 0:
    if e1 <= 0:
      sink[0] = <sink_t> (_abs(e3) * pow((e2/e3), par[0]) * pow(1 + e1/_abs(e2), par[1]))
    else:
      a = par[2] * e1 / _abs(e2)
      if a < 1:                         
        sink[0] = <sink_t> (_abs(e3) * pow((e2/e3), par[0]) * pow(1 - a, par[1]))
      else:
        sink[0] = 0
  else:
    sink[0] = 0

#Generalized Frangi filer [Sato et al, Three dimensional multi-scale line filter for segmentation and visualization of curvilinear structures in medicalimages, 1998]
cdef inline void lambda123_threshold_kernel(sink_t* sink, index_t sink_stride, double e1, double e2, double e3, double* par) nogil:
  cdef double a;
  
  if e2 < 0 and e3 < 0:
    if e1 <= 0 and (_abs(e3) * pow((e2/e3), par[0]) * pow(1 + e1/_abs(e2), par[1])) >= par[3]:
      sink[0] = 1
    else:
      a = par[2] * e1 / _abs(e2)
      if a < 1 and (_abs(e3) * pow((e2/e3), par[0]) * pow(1 - a, par[1])) >= par[3]:
        sink[0] = 1
      #else:
      #  sink[0] = 0;
  #else:
  #  sink[0] = 0;



def eigenvalues(source_t[:, :, :] source, sink_t[:, :, :, :] sink, index_t sink_stride, double[:] parameter):
  hessian_eigenvalue_core(eigenvalue_kernel[sink_t], source, sink, sink_stride, parameter)

def tubeness(source_t[:, :, :] source, sink_t[:, :, :, :] sink, index_t sink_stride, double[:] parameter):
  hessian_eigenvalue_core(tubeness_kernel[sink_t], source, sink, sink_stride, parameter)

def tubeness_threshold(source_t[:, :, :] source, sink_t[:, :, :, :] sink, index_t sink_stride, double[:] parameter):
  hessian_eigenvalue_core(tubeness_threshold_kernel[sink_t], source, sink, sink_stride, parameter)

def lambda123(source_t[:, :, :] source, sink_t[:, :, :, :] sink, index_t sink_stride, double[:] parameter):
  hessian_eigenvalue_core(lambda123_kernel[sink_t], source, sink, sink_stride, parameter)

def lambda123_threshold(source_t[:, :, :] source, sink_t[:, :, :, :] sink, index_t sink_stride, double[:] parameter):
  hessian_eigenvalue_core(lambda123_threshold_kernel[sink_t], source, sink, sink_stride, parameter)

cdef void hessian_eigensystem_core(void kernel(sink_t*, index_t, double, double, double, double, double, double, double, double, double, double*) nogil,
                                   source_t[:, :, :] source, sink_t[:, :, :, :] sink, index_t sink_stride, double[:] parameter) except *:
  """Compute Hessian eigenvalues for each pixel and apply a measure defined by the kernel that also recieves the Hessian matrix."""
  # eigenvalues via affine change of A (see https://en.wikipedia.org/wiki/Eigenvalue_algorithm 3x3 matrices)
  
  # array sizes
  cdef index_t nx = source.shape[0]
  cdef index_t ny = source.shape[1]
  cdef index_t nz = source.shape[2]   
  
  #cdef index_t odepth = sink.shape[3]
  
  # local variable types
  cdef index_t x,y,z,xm,ym,zm,xp,yp,zp
 
  # Hessian matrix is represented as A = h[0,0], D = h[1,1], F = h[2,2], B = h[0,1]=h[1,0], C = h[0,2]=h[2,0], E = h[1,2]=h[2,1]
  cdef double A, B, C, D, E, F;
  
  
  # Compute eigenvalues and apply kernel
  cdef double s, p1, p2, p, r, phi, pi_3, pi_2, a, d, f
  cdef double e1, e2, e3
  
  pi_3 = M_PI / 3
  pi_3_2 = 2 * pi_3

  with nogil:
    for x in range(nx):
      xm = x - 1 if x > 0 else x
      xp = x + 1 if x < nx - 1 else nx -1

      for y in range(ny):
        
        ym = y - 1 if y > 0 else y
        yp = y + 1 if y < ny - 1 else ny - 1

        for z in range(nz):
          zm = z - 1 if z > 0 else z
          zp = z + 1 if z < nz - 1 else nz - 1

          # create hessian
          s = 2.0 * <double>source[x,y,z]

          A = source[xm,y, z ] - s + source[xp,y, z ]
          D = source[x, ym,z ] - s + source[x, yp,z ]
          F = source[x, y, zm] - s + source[x, y ,zp]

          B = (<double>source[xp,yp,z ] - source[xm,yp,z ] - source[xp,ym,z ] + source[xm,ym,z ]) / 4.0
          C = (<double>source[xp,y ,zp] - source[xm,y ,zp] - source[xp,y ,zm] + source[xm,y ,zm]) / 4.0
          E = (<double>source[x ,yp,zp] - source[x ,ym,zp] - source[x ,yp,zm] + source[x ,ym,zm]) / 4.0

          # calculate eigenvalues
          p1 = B * B + C * C + E * E
          if p1 == 0:
              # eigen values are A, D, F
              # sort e1 > = e2 >= e3
              if A <= D:
                if D <= F :
                  e3 = A; e2 = D; e1 = F
                else: # D > F
                  if A <= F:
                    e3 = A; e2 = F; e1 = D
                  else: # A > F
                    e3 = F; e2 = A; e1 = D
              else: #D < A
                if A <= F:
                  e3 = D; e2 = A; e1 = F
                else: # A > F
                  if D <= F:
                    e3 = D; e2 = F; e1 = A
                  else: # D > F
                    e3 = F; e2 = D; e1 = A

          else: 
              q  = (A + D + F) / 3
              p2 = (A - q) * (A - q) + (D - q) * (D - q) + (F - q) * (F - q) + 2 * p1
              p  = sqrt(p2 / 6)

              # affine matrix A = pB + qI
              a = (A - q)
              d = (D - q)
              f = (F - q)

              # det B  / 2     
              r = (a * d * f + 2 * B * C * E - a * E * E - B * B * f - C * C * d) / 2 / (p * p * p)

              if r <= -1:
                  phi = pi_3
              elif r >= 1:
                  phi = 0
              else:
                  phi = acos(r) / 3

              e1 = q + 2 * p * cos(phi)
              e3 = q + 2 * p * cos(phi + pi_3_2)
              e2 = 3 * q - e1 - e3

          # apply kernel
          kernel(&sink[x, y, z, 0], sink_stride, e1, e2, e3, A, B, C, D, E, F, &parameter[0])

# eigensystem kernel
cdef inline void eigensystem_kernel(sink_t* sink, index_t sink_stride, double e1, double e2, double e3,
                                    double A, double B, double C, double D, double E, double F, 
                                    double* par) nogil:
  #par[0] = 0 # eigenvalues only
  #par[0] = 1,2,3 number of eigenvectors to compute
  # eigenvector calculation based on https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
  cdef double r1, r2, r3
  cdef double c11, c12, c13
  cdef double c21, c22, c23
  cdef double c31, c32, c33
  cdef double d1, d2, d3, dmax
  cdef int imax;
    
  sink[0              ] = <sink_t> e1
  sink[1 * sink_stride] = <sink_t> e2
  sink[2 * sink_stride] = <sink_t> e3

  if par[0] <= 0:
    return

  # e1 = e2 = e3 = 0 => arbitrary basis
  if e1 == 0 and e2 == 0 and e3 == 0:
      sink[3 * sink_stride] = <sink_t> 1
      sink[4 * sink_stride] = <sink_t> 0
      sink[5 * sink_stride] = <sink_t> 0

      if par[0] <= 1:
        return

      sink[6 * sink_stride] = <sink_t> 0
      sink[7 * sink_stride] = <sink_t> 1
      sink[8 * sink_stride] = <sink_t> 0

      if par[0] <= 2:
        return

      sink[9  * sink_stride] = <sink_t> 0
      sink[10 * sink_stride] = <sink_t> 0
      sink[11 * sink_stride] = <sink_t> 1

      return

  # H - lambda I
  r1 = A - e1
  r2 = D - e1
  r3 = F - e1

  # row1 x row 2
  c11 = B * E - C * r2
  c12 = B * C - E * r1
  c13 = r1 * r2 - B * B

  # row1 x row 3
  c21 = B * r3 - C * E
  c22 = C * C - r1 * r3
  c23 = E * r1 - B * C

  # row 2 x row 3
  c31 = r2 * r3 - E * E
  c32 = C * E - B * r3
  c33 = B * E - C * r2

  # dot products
  d1 = c11 * c11 + c12 * c12 + c13 * c13
  d2 = c21 * c21 + c22 * c22 + c23 * c23
  d3 = c31 * c31 + c32 * c32 + c33 * c33

  if d1 > d2:
    if d1 > d3:
      d1 = sqrt(d1)
      sink[3 * sink_stride] = <sink_t> (c11 / d1)
      sink[4 * sink_stride] = <sink_t> (c12 / d1)
      sink[5 * sink_stride] = <sink_t> (c13 / d1)
    else: # d3 > d1 > d2
      d3 = sqrt(d3)
      sink[3 * sink_stride] = <sink_t> (c31 / d3)
      sink[4 * sink_stride] = <sink_t> (c32 / d3)
      sink[5 * sink_stride] = <sink_t> (c33 / d3)
  else: # d2 > d1
    if d2 > d3:
      d2 = sqrt(d2)
      sink[3 * sink_stride] = <sink_t> (c21 / d2)
      sink[4 * sink_stride] = <sink_t> (c22 / d2)
      sink[5 * sink_stride] = <sink_t> (c23 / d2)
    else: # d3 > d2 > d3
      d3 = sqrt(d3)
      sink[3 * sink_stride] = <sink_t> (c31 / d3)
      sink[4 * sink_stride] = <sink_t> (c32 / d3)
      sink[5 * sink_stride] = <sink_t> (c33 / d3)

  if par[0] <= 1:
    return

  # compute eigenvectors 2,3
  cdef double w1, w2, w3
  cdef double u1, u2, u3
  cdef double v1, v2, v3
  cdef double il
  cdef double au1, au2, au3
  cdef double av1, av2, av3
  cdef double uau, uav, vav
  cdef double uau_abs, uav_abs, vav_abs
  
  w1 =  sink[3 * sink_stride]
  w2 =  sink[4 * sink_stride]
  w3 =  sink[5 * sink_stride]

  # orthonomal complemetns U,V
  if fabs(w1) > fabs(w2):
    il = 1 / sqrt(w1 * w1 + w3 * w3)
    u1 = -w3 * il
    u2 = 0
    u3 = w1 * il
  else:
    il = 1 / sqrt(w2 * w2 + w3 * w3)
    u1 = 0
    u2 = w3 * il
    u3 = - w2 * il

  # v = w x u
  v1 = w2 * u3 - w3 * u2
  v2 = w3 * u1 - w1 * u3
  v3 = w1 * u2 - u1 * w2

  # A v, A u
  av1 = A * v1 + B * v2 + C * v3
  av2 = B * v1 + D * v2 + E * v3
  av3 = C * v1 + E * v2 + F * v3

  au1 = A * u1 + B * u2 + C * u3
  au2 = B * u1 + D * u2 + E * u3
  au3 = C * u1 + E * u2 + F * u3

  # uAu, uAv, vAv
  uau = u1 * au1 + u2 * au2 + u3 * au3 - e2
  uav = u1 * av1 + u2 * av2 + u3 * av3
  vav = v1 * av1 + v2 * av2 + v3 * av3 - e2

  uau_abs = fabs(uau)
  uav_abs = fabs(uav)
  vav_abs = fabs(vav)

  if uau_abs >= vav_abs:
    if uau_abs > 0 or uav_abs > 0:
      if uau_abs >= uav_abs:
        uav /= uau
        uau = 1 / sqrt(1 + uav * uav)
        uav *= uau
      else:
        uau /= uav
        uav = 1 / sqrt(1 + uau * uau)
        uau *= uav

      sink[6 * sink_stride] = <sink_t> (uav * u1 - uau * v1)
      sink[7 * sink_stride] = <sink_t> (uav * u2 - uau * v2)
      sink[8 * sink_stride] = <sink_t> (uav * u3 - uau * v3)

    else:
      sink[6 * sink_stride] = <sink_t> u1
      sink[7 * sink_stride] = <sink_t> u2
      sink[8 * sink_stride] = <sink_t> u3
  else:
    if vav_abs > 0 or uav_abs > 0:
      if vav_abs >= uav_abs:
        uav /= vav
        vav = 1 / sqrt(1 + uav * uav)
        uav *= vav
      else:
        vav /= uav
        uav = 1 / sqrt(1 + vav * vav)
        vav *= uav

      sink[6 * sink_stride] = <sink_t> (vav * u1 - uav * v1)
      sink[7 * sink_stride] = <sink_t> (vav * u2 - uav * v2)
      sink[8 * sink_stride] = <sink_t> (vav * u3 - uav * v3)

    else:
      sink[6 * sink_stride] = <sink_t> u1
      sink[7 * sink_stride] = <sink_t> u2
      sink[8 * sink_stride] = <sink_t> u3

  if par[0] <= 2:
      return

  u1 = sink[6 * sink_stride]
  u2 = sink[7 * sink_stride]
  u3 = sink[8 * sink_stride]

  sink[9 * sink_stride] = <sink_t> (w2 * u3 - w3 * u2)
  sink[10* sink_stride] = <sink_t> (w3 * u1 - w1 * u3)
  sink[11* sink_stride] = <sink_t> (w1 * u2 - u1 * w2)

def eigensystem(source_t[:, :, :] source, sink_t[:, :, :, :] sink, index_t sink_stride, double[:] parameter):
  hessian_eigensystem_core(eigensystem_kernel[sink_t], source, sink, sink_stride, parameter)

#Note: this is original ClearMap 2.0 code for reference
# cdef void hessian_core_old(void kernel(sink_t*, index_t, double, double, double, double*) nogil,
#                        source_t[:, :, :] source, sink_t[:, :, :, :] sink, index_t sink_stride, double[:] parameter) except *:
#   """Compute Hessian eigenvalues for each pixel and apply a measure defined by the kernel."""
  
#   # array sizes
#   cdef index_t nx = source.shape[0]
#   cdef index_t ny = source.shape[1]
#   cdef index_t nz = source.shape[2]   
  
#   cdef index_t max_img = nx * ny * nz
  
#   #cdef index_t odepth = sink.shape[3]
  
#   # local variable types
#   cdef index_t x,y,z,xm,ym,zm,xp,yp,zp
  
#   cdef double one_third = 1.0/3.0;
#   cdef double root_three = 1.7320508075688772935;

#   # Hessian matrix is represented as A = h[0,0], D = h[1,1], F = h[2,2], B = h[0,1]=h[1,0], C = h[0,2]=h[2,0], E = h[1,2]=h[2,1]
#   cdef double A, B, C, D, E, F;
  
  
#   # Compute eigenvalues and apply kernel
#   cdef double i, discriminant, a, b, c, d, q, r, s, t, u, v
#   cdef double e1, e2, e3
#   cdef double e1a, e2a, e3a
#   cdef double e1s, e2s, e3s
#   a = -1.0;
   
#   with nogil:
#     for x in range(nx):
#       xm = x - 1 if x > 0 else x;
#       xp = x + 1 if x < nx - 1 else nx -1;
      
#       for y in range(ny):
        
#         ym = y - 1 if y > 0 else y;
#         yp = y + 1 if y < ny - 1 else ny - 1;
        
#         for z in range(nz):
#           zm = z - 1 if z > 0 else z;
#           zp = z + 1 if z < nz - 1 else nz - 1;
          
#           # create hessian
#           i = 2.0 * <double>source[x,y,z];
          
#           A = source[xm,y, z ] - i + source[xp,y, z ];
#           D = source[x, ym,z ] - i + source[x, yp,z ];
#           F = source[x, y, zm] - i + source[x, y ,zp];

#           B = (<double>source[xp,yp,z ] - source[xm,yp,z ] - source[xp,ym,z ] + source[xm,ym,z ]) / 4.0;
#           C = (<double>source[xp,y ,zp] - source[xm,y ,zp] - source[xp,y ,zm] + source[xm,y ,zm]) / 4.0;
#           E = (<double>source[x ,yp,zp] - source[x ,ym,zp] - source[x ,yp,zm] + source[x ,ym,zm]) / 4.0;
 
#           # calculate eigenvalues
#           b = A + D + F;
#           c = B * B + C * C + E * E - A * D - A * F - D * F;
#           d = A * D * F - A * E * E - B * B * F + 2 * B * C * E - C * C * D;
          
#           q = (3*a*c - b*b) / (9*a*a);
#           r = (9*a*b*c - 27*a*a*d - 2*b*b*b) / (54*a*a*a);
          
#           discriminant = q*q*q + r*r;

#           if discriminant < 0:
#             s = pow(sqrt(r*r - discriminant), one_third);          
#             t = atan2(sqrt(-discriminant), r) / 3.0;

#             u = 2 * s * cos(t);
#             first = - (u / 2) - (b / 3*a);
#             last  = - root_three * s * sin(t);
            
#             e1 = ( u - (b / (3*a)) );
#             e2 = ( first + last );
#             e3 = ( first - last );
            
#           elif discriminant == 0:
            
#             if r >= 0:
#               u = 2 * pow(r,one_third);
#             else:                  
#               u = -2 * pow(-r,one_third);
#             v = b / (3 * a);
  
#             e1 = ( u   - v );
#             e2 = (-u/2 - v );
#             e3 = e2;
          
#           else:
#             # discriminant > 0 should not happen for a symmetric matrix
#             e1 = e2 = e3 = 0.0;
 
#           # sort eigenvalues es1 >= es2 >= es3
#           # e1 = abs(e1); e2 = abs(e2); e3 = abs(e3);
#           if e1 <= e2:
#             if e2 <= e3 :
#               e3s = e1; e2s = e2; e1s = e3;
#             else:
#               if e1 <= e3:
#                 e3s = e1; e2s = e3; e1s = e2;
#               else:
#                 e3s = e3; e2s = e1; e1s = e2;
#           else: #e2 < e1
#             if e1 <= e3:
#               e3s = e2; e2s = e1; e1s = e3;
#             else:
#               if e2 <= e3:
#                 e3s = e2; e2s = e3; e1s = e1;
#               else:
#                 e3s = e3; e2s = e2; e1s = e1;
          
#           # apply kernel
#           kernel(&sink[x, y, z, 0], sink_stride, e1s, e2s, e3s, &parameter[0]);


# def eigenvalues_old(source_t[:, :, :] source, sink_t[:, :, :, :] sink, index_t sink_stride, double[:] parameter):
#   hessian_core_old(eigenvalue_kernel[sink_t], source, sink, sink_stride, parameter);

# cdef inline void test_kernel(sink_t* sink, index_t sink_stride, double e1, double e2, double e3,
#                              double A, double B, double C, double D, double E, double F, 
#                              double* par) nogil:

#   sink[0              ] = <sink_t> e1; 
#   sink[1 * sink_stride] = <sink_t> e2;
#   sink[2 * sink_stride] = <sink_t> e3;

#   sink[3 * sink_stride] = <sink_t> A; 
#   sink[4 * sink_stride] = <sink_t> B;
#   sink[5 * sink_stride] = <sink_t> C;
#   sink[6 * sink_stride] = <sink_t> D; 
#   sink[7 * sink_stride] = <sink_t> E;
#   sink[8 * sink_stride] = <sink_t> F;


# def eigensystem_test(source_t[:, :, :] source, sink_t[:, :, :, :] sink, index_t sink_stride, double[:] parameter):
#   hessian_eigensystem_core(test_kernel[sink_t], source, sink, sink_stride, parameter);
