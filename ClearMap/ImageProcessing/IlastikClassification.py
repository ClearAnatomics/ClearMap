# -*- coding: utf-8 -*-
"""
Inteface to Illastik pixel classification

This module allows to integrate ilastik pixel classification into the *ClearMap*
pipeline. 

To train a classifier ilastik should be used:

  * 

Note:
    Note that ilastik classification works in parallel, thus it is advised to 
    process the data sequentially, see 
    :func:`~ClearMap.Imageprocessing.StackProcessing.sequentiallyProcessStack`  

Note:
    Ilastik 0.5 works for images in uint8 format !

References:
    * `Ilastik <http://ilastik.org/>`_
    * Based on the ilastik interface from `cell profiler <http://www.cellprofiler.org/>`_
"""
#:copyright: Copyright 2015 by Christoph Kirst, The Rockefeller University, New York City
#:license: GNU, see LICENSE.txt for details.

import sys
import numpy

import scipy.ndimage.measurements as sm;

import ClearMap.ImageProcessing.Ilastik as ilastik
#from ClearMap.ImageProcessing.BackgroundRemoval import removeBackground
from ClearMap.ImageProcessing.MaximaDetection import findCenterOfMaxima, findIntensity
from ClearMap.ImageProcessing.CellSizeDetection import detectCellShape, findCellSize, findCellIntensity
from ClearMap.ImageProcessing.StackProcessing import writeSubStack

from ClearMap.Utils.Timer import Timer
from ClearMap.Utils.ParameterTools import getParameter, writeParameter

from ClearMap.Visualization.Plot import plotTiling


def isInitialized():
  """Check if Ilastik is useable
  
  Returns:
    bool: True if Ilastik is installed and useable by *ClearMap*
  """
  
  return ilastik.isInitialized();
 
 
def checkInitialized():
    """Checks if ilastik is initialized
    
    Returns:
        bool: True if ilastik paths are set.
    """
    
    if not isInitialized():
        raise RuntimeError("Ilastik not initialized: run initializeIlastik(path) with proper path to ilastik first");

    return True;


def classifyPixel(img, classifyPixelParameter = None, subStack = None, verbose = False, out = sys.stdout, **parameter):
    """Detect Cells Using a trained classifier in Ilastik
    
    Arguments:from ClearMap.ImageProcessing.CellSizeDetection import detectCellShape, findCellSize, findCellIntensity
        img (array): image data
        classifyPixelParameter (dict):
            ============ ==================== ===========================================================
            Name         Type                 Descritption
            ============ ==================== ===========================================================
            *classifier* (str or  None)       Ilastik project file with trained pixel classifier
            *save*       (str or None)        save the classification propabilities to a file
            *verbose*    (bool or int)        print / plot information about this step 
            ============ ==================== ===========================================================
        subStack (dict or None): sub-stack information 
        verbose (bool): print progress info 
        out (object): object to write progress info to
    
    Returns:
        array: probabilities for each pixel to belong to a class in the classifier, shape is (img.shape, number of classes)
    """

    ilastik.checkInitialized();
    
    classifier = getParameter(classifyPixelParameter, "classifier", None);  
    save       = getParameter(classifyPixelParameter, "save", None);   
    verbose    = getParameter(classifyPixelParameter, "verbose", verbose);
     
    if verbose:
        writeParameter(out = out, head = 'Ilastik classification:', classifier = classifier, save = save);        
    
    
    timer = Timer(); 
        
    #remove background
    #img2 = removeBackground(img, verbose = verbose, out = out, **parameter);
      
    #classify image
    if classifier is None:        
        return img;
    
    imgclass = ilastik.classifyPixel(classifier, img);
    
    if not save is None:
        for i in range(imgclass.shape[4]):
            fn = save[:-4] + '_class_' + str(i) + save[-4:];
            writeSubStack(fn, imgclass[:,:,:,i], subStack = subStack)
      
    if verbose > 1:
        for i in range(imgclass.shape[4]):
            plotTiling(imgclass[:,:,:,i]);
    
    if verbose:
        out.write(timer.elapsedTime(head = 'Ilastik classification') + '\n');    
    
    return imgclass;



def classifyCells(img, classifyCellsParameter = None, classifier = None, classindex = 0, save = None, verbose = False,
                  detectCellShapeParameter = None,
                  subStack = None, out = sys.stdout, **parameter):
    """Detect Cells Using a trained classifier in Ilastik
    
    The routine assumes that the first class is identifying the cells.
        
    Arguments:    
        img (array): image data
        classifyPixelParameter (dict):
            ============ ==================== ===========================================================
            Name         Type                 Descritption
            ============ ==================== ===========================================================
            *classifier* (str or  None)       Ilastik project file with trained pixel classifier
            *classindex* (int)                class index considered to be cells
            *save*       (str or None)        save the detected cell pixel to a file
            *verbose*    (bool or int)        print / plot information about this step 
            ============ ==================== ===========================================================
        subStack (dict or None): sub-stack information 
        verbose (bool): print progress info 
        out (object): object to write progress info to
    
    Returns:
        tuple: centers of the cells, intensity measurments
        
    Note:    
        The routine could be potentially refined to make use of background 
        detection in ilastik
    """
    
    classifier = getParameter(classifyCellsParameter, "classifier", classifier);
    classindex = getParameter(classifyCellsParameter, "classindex", classindex);
    save       = getParameter(classifyCellsParameter, "save", save);   
    verbose    = getParameter(classifyCellsParameter, "verbose", verbose);
     
    if verbose:
        writeParameter(out = out, head = 'Ilastik cell detection:', classifier = classifier, classindex = classindex, save = save);        

    timer = Timer(); 

    ilastik.isInitialized();
    
    #remove background
    #img = removeBackground(img, verbose = verbose, out = out, **parameter);
      
    #classify image / assume class 1 are the cells !  
    timer = Timer();  
    
    imgmax = ilastik.classifyPixel(classifier, img);
    #print imgmax.shape
    #max probability gives final class, last axis is class axis
    imgmax = numpy.argmax(imgmax, axis = -1);
    
    if save:
        writeSubStack(save, numpy.asarray(imgmax, dtype = 'float32'), subStack = subStack)    

    # class 0 is used as cells 
    imgmax = imgmax == classindex; # class 1 is used as cells 
    imgshape, nlab = sm.label(imgmax);
    
    if verbose > 1:
        plotTiling(imgmax);
        
    #center of maxima
    centers = findCenterOfMaxima(img, imgmax, imgshape, verbose = verbose, out = out, **parameter);
    
    #intensity of cells
    #cintensity = findIntensity(img, centers, verbose = verbose, out = out, **parameter);

    #intensity of cells in filtered image
    #cintensity2 = findIntensity(img, centers, verbose = verbose, out = out, **parameter);
    
    #if verbose:
    #    out.write(timer.elapsedTime(head = 'Ilastik cell detection') + '\n');    
    
    #return ( centers, numpy.vstack((cintensity, cintensity2)).transpose() );   
    #return ( centers, cintensity ); 
    
    
    #cell size detection
    #detectCellShapeParameter = getParameter(classifyCellsParameter, "detectCellShapeParameter", detectCellShapeParameter);
    #cellShapeThreshold = getParameter(detectCellShapeParameter, "threshold", None);
    
    #if not cellShapeThreshold is None:
        
    # cell shape via watershed
    #imgshape = detectCellShape(img, centers, detectCellShapeParameter = detectCellShapeParameter, verbose = verbose, out = out, **parameter);
    
    #size of cells        
    csize = findCellSize(imgshape, maxLabel = centers.shape[0], out = out, **parameter);
    
    #intensity of cells
    cintensity = findCellIntensity(img, imgshape,  maxLabel = centers.shape[0], verbose = verbose, out = out, **parameter);

    #intensity of cells in background image
    #cintensity2 = findCellIntensity(img2, imgshape,  maxLabel = centers.shape[0], verbose = verbose, out = out, **parameter);

    #intensity of cells in dog filtered image
    #if dogSize is None:
    #    cintensity3 = cintensity2;
    #else:
    #    cintensity3 = findCellIntensity(img3, imgshape,  maxLabel = centers.shape[0], verbose = verbose, out = out, **parameter);
    
    if verbose:
        out.write(timer.elapsedTime(head = 'Ilastik Cell Detection') + '\n');
    
    #remove cell;s of size 0
    idz = csize > 0;
                   
    #return ( centers[idz], numpy.vstack((cintensity[idz], cintensity3[idz], cintensity2[idz], csize[idz])).transpose());        
    return ( centers[idz], numpy.vstack((cintensity[idz], csize[idz])).transpose() ); 

#    else:
#        #intensity of cells
#        cintensity = findIntensity(img, centers, verbose = verbose, out = out, **parameter);
#
#        #intensity of cells in background image
#        #cintensity2 = findIntensity(img2, centers, verbose = verbose, out = out, **parameter);
#    
#        #intensity of cells in dog filtered image
#        #if dogSize is None:
#        #    cintensity3 = cintensity2;
#        #else:
#        #    cintensity3 = findIntensity(img3, centers, verbose = verbose, out = out, **parameter);
#
#        if verbose:
#            out.write(timer.elapsedTime(head = 'Ilastik Cell Detection') + '\n');
#    
#        #return ( centers, numpy.vstack((cintensity, cintensity3, cintensity2)).transpose());
#        return ( centers, numpy.vstack((cintensity)).transpose());
        
    



