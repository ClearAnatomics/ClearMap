
import numpy as np

def myconvolve(x,h, is_fft = False):
    if is_fft:
        x_f = x
        h_f = h
    else:
        x_f = np.fft.rfftn(x)
        h_f = np.fft.rfftn(h)
    
    return np.abs(np.fft.irfftn(x_f*h_f))

def psf(dshape,sigmas = (2.,2.)):
    Xs = np.meshgrid(*[np.arange(-_s/2,_s/2) for _s in dshape], indexing="ij")

    h = np.exp(-np.sum([_X**2/2./_s**2 for _X,_s in zip(Xs,sigmas)],axis=0))

    h *= 1./np.sum(h)
    return np.fft.ifftshift(h)

    


def blur(d,h):
        d_f = np.fft.rfftn(d)
        h_f = np.fft.rfftn(h)
        return np.fft.irfftn(d_f*h_f)


def blur_kernel2(N,rad):
        k = np.fft.fftfreq(N)
        KY,KX = np.meshgrid(k,k,indexing="ij")
        KR = np.hypot(KX,KY)
        u = 1.*(KR<=1./rad)
        h = np.abs(np.fft.ifftn(u))**2
        h *= 1./np.sum(h)
        return np.fft.fftshift(h)

def blur_kernel3(N,rad):
        k = np.fft.fftfreq(N)
        KZ,KY,KX = np.meshgrid(k,k,k,indexing="ij")
        KR = np.sqrt(KX**2+KY**2+KZ**2)
        u = 1.*(KR<=1./rad)
        h = np.abs(np.fft.ifftn(u))**2
        h *= 1./np.sum(h)
        return np.fft.fftshift(h)

def blur_disk(N,rad):
        k = np.arange(N)-N/2.
        KY,KX = np.meshgrid(k,k,indexing="ij")
        KR = np.hypot(KX,KY)
        h = 1.*(KR<=rad)
        h *= 1./np.sum(h)
        return h



def soft_thresh(x,lam):
    """
    the solution to
        w = argmin 1/2*(w-x)^2+lam*|w|
    """

    return np.sign(x)*np.maximum(0.,np.abs(x)-lam)

def divergence(u):
    """the divergence of vector field u where u[i,...] are the components """
    return reduce(np.add,[np.gradient(u[i])[i] for i in range(len(u))])


def finite_deriv_dft_central(dshape,units = None, use_rfft = False):
    """ the dft of the central finite differences in 2d

    i.e. the fft of the stencil [1,0,-1]
    """
    if units is None:
        units = (1.,)*len(dshape)

    kxs = [np.fft.fftfreq(_s,_u) for _s,_u in zip(dshape,units)]

    if use_rfft:
        kxs[-1] = kxs[-1][:dshape[-1]//2+1]

    if len(kxs)>1:
        KXs = np.meshgrid(*kxs,indexing="ij")
    else:
        KXs = kxs

    return [1.j*np.sin(2.*np.pi*_K)/np.prod(units) for _K in KXs]

def dft_lap(dshape,units = None, use_rfft = False):
    """ the dft of the laplacian stencil in 2d"""
    if units is None:
        units = (1.,)*len(dshape)

    kxs = [np.fft.fftfreq(_s,_u) for _s,_u in zip(dshape,units)]

    if use_rfft:
        kxs[-1] = kxs[-1][:dshape[-1]//2+1]

    KXs = np.meshgrid(*kxs,indexing="ij")

    h = np.sum([2*np.cos(2.*np.pi*_K) for _K in KXs],axis=0) - 2.*len(KXs)
    h *= 1./np.prod(units)
    return h


def psf_airy(dshape,rads = (2.,2.)):
    ks = [np.fft.fftfreq(d) for d in dshape]

    Ks = np.meshgrid(*ks,indexing="ij")
    KR = np.sqrt(reduce(np.add,[K**2*4*r**2 for K,r in zip(Ks,rads)]))
    u = 1.*(KR<=1.)
    h = np.abs(np.fft.ifftn(u))**2
    h *= 1./np.sum(h)
    return h
