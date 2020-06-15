__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

#optiimze image itme for speed !!
# only read / display visible region in plot !


  
  
try:
    from collections.abc import Callable
except ImportError:
    # fallback for python < 3.3
    from collections import Callable

from pyqtgraph.Point import Point



class ImageItem(pg.graphicsItems.ImageItem.ImageItem):

  def init(self, *args, **kwargs):
    print('init')
    super(ImageItem, self).__init__(*args, **kwargs);
  
  
  def render(self):
    # Convert data to QImage for display.
   
    print('redering');
    if self.image is None or self.image.size == 0:
        return

    # Request a lookup table if this image has only one channel
    if self.image.ndim == 2 or self.image.shape[2] == 1:
        if isinstance(self.lut, Callable):
            lut = self.lut(self.image)
        else:
            lut = self.lut
    else:
        lut = None

    print('lut');

    if self.autoDownsample:
        # reduce dimensions of image based on screen resolution
        o = self.mapToDevice(pg.QtCore.QPointF(0,0))
        x = self.mapToDevice(pg.QtCore.QPointF(1,0))
        y = self.mapToDevice(pg.QtCore.QPointF(0,1))

        print(o,x,y)

        # Check if graphics view is too small to render anything
        if o is None or x is None or y is None:
            return

        w = Point(x-o).length()
        h = Point(y-o).length()
        if w == 0 or h == 0:
            self.qimage = None
            return
          
        print(w,h)  
          
        xds = max(1, int(1.0 / w))
        yds = max(1, int(1.0 / h))
        
        
        print(xds, yds);
        
        axes = [1, 0] if self.axisOrder == 'row-major' else [0, 1]
        image = pg.fn.downsample(self.image, xds, axis=axes[0])
        image = pg.fn.downsample(image, yds, axis=axes[1])
        self._lastDownsample = (xds, yds)

        # Check if downsampling reduced the image size to zero due to inf values.
        if image.size == 0:
            return
    else:
        image = self.image
        
    print(image.shape)

    # if the image data is a small int, then we can combine levels + lut
    # into a single lut for better performance
    levels = self.levels
    if levels is not None and levels.ndim == 1 and image.dtype in (np.ubyte, np.uint16):
        if self._effectiveLut is None:
            eflsize = 2**(image.itemsize*8)
            ind = np.arange(eflsize)
            minlev, maxlev = levels
            levdiff = maxlev - minlev
            levdiff = 1 if levdiff == 0 else levdiff  # don't allow division by 0
            if lut is None:
                efflut = pg.fn.rescaleData(ind, scale=255./levdiff,
                                        offset=minlev, dtype=np.ubyte)
            else:
                lutdtype = np.min_scalar_type(lut.shape[0]-1)
                efflut = pg.fn.rescaleData(ind, scale=(lut.shape[0]-1)/levdiff,
                                        offset=minlev, dtype=lutdtype, clip=(0, lut.shape[0]-1))
                efflut = lut[efflut]

            self._effectiveLut = efflut
        lut = self._effectiveLut
        levels = None

    # Convert single-channel image to 2D array
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]

    # Assume images are in column-major order for backward compatibility
    # (most images are in row-major order)
    if self.axisOrder == 'col-major':
        image = image.transpose((1, 0, 2)[:image.ndim])

    argb, alpha = pg.fn.makeARGB(image, lut=lut, levels=levels)
    self.qimage = pg.fn.makeQImage(argb, alpha, transpose=False, copy=False)
    print('redering done')


  def drawAt(self, pos, ev=None):
        print('draw at')
    
    
        pos = [int(pos.x()), int(pos.y())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0,dk.shape[0]]
        sy = [0,dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0]+ dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1]+ dk.shape[1]]

        for i in [0,1]:
            dx1 = -min(0, tx[i])
            dx2 = min(0, self.image.shape[0]-tx[i])
            tx[i] += dx1+dx2
            sx[i] += dx1+dx2

            dy1 = -min(0, ty[i])
            dy2 = min(0, self.image.shape[1]-ty[i])
            ty[i] += dy1+dy2
            sy[i] += dy1+dy2

        ts = (slice(tx[0],tx[1]), slice(ty[0],ty[1]))
        ss = (slice(sx[0],sx[1]), slice(sy[0],sy[1]))
        mask = self.drawMask
        src = dk

        print(pos, dk, kc, ts, ss)

        if isinstance(self.drawMode, Callable):
            self.drawMode(dk, self.image, mask, ss, ts, ev)
        else:
            src = src[ss]
            if self.drawMode == 'set':
                if mask is not None:
                    mask = mask[ss]
                    self.image[ts] = self.image[ts] * (1-mask) + src * mask
                else:
                    self.image[ts] = src
            elif self.drawMode == 'add':
                self.image[ts] += src
            else:
                raise Exception("Unknown draw mode '%s'" % self.drawMode)
            self.updateImage()


  def paint(self, p, *args):
        print('paint', p);
        
        if self.image is None:
            return
        if self.qimage is None:
            self.render()
            if self.qimage is None:
                return

        if self.paintMode is not None:
            p.setCompositionMode(self.paintMode)


        shape = self.image.shape[:2] if self.axisOrder == 'col-major' else self.image.shape[:2][::-1]
        p.drawImage(pg.QtCore.QRectF(0,0,*shape), self.qimage)

        if self.border is not None:
            p.setPen(self.border)
            p.drawRect(self.boundingRect())
