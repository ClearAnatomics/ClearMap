import os
import sys
import numpy as np
work_dir='/data_SSD_1to/blinder'#'/data_SSD_2to/whiskers_graphs'
brain='200916'
points=np.load(work_dir + '/' + brain + '/' + 'cells.npy')
ct_x=[c[0] for c in points]
ct_y=[c[1] for c in points]
ct_z=[c[2] for c in points]
points=[ct_x, ct_y, ct_z]
points=np.array(points)
points=points.T
points=points[minDistances<=3]
# points = np.load('/mnt/vol00-renier/Sophie/presentation/imageActa2VSiggPodo/iggpodoacta2_capillaries_grid_dnn.npy')
# indices = np.load('/mnt/vol00-renier/Sophie/presentation/imageActa2VSiggPodo/iggpodoacta2_capillaries_indices_dnn.npy')
# points = points[:, ]
filepath = work_dir + '/' + brain + '/'+'cells_spheres_short_dist.iv'


filepath = '/data_2to/p4/4_graph_end_point_2.iv'

binary = False


'''
Separator {
        Transform {
          translation  0.0 0.0 0.0 
        }
        Sphere { radius 1.0 }
      }
    }'''

def writeSpheres(pts):
    bool = True
    for pt in pts:
        st = """Separator {
            Material {
                diffuseColor 50 50 50
            }
            Separator {
                        Transform {
                            """
        if binary:
            f = open(filepath, 'a+b')
            f.write(bytearray(st))
        else:
            f = open(filepath, 'a+')
            f.write(st)


        if not np.isnan(pt[0]) and not np.isnan(pt[1]) and not np.isnan(pt[2]):
            if bool == True:
                if binary:
                    f.write('translation'+' '+bytearray(str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2])))
                else:
                    f.write('translation'+' '+str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]))
                bool = False
            else:
                if binary:
                    f.write('translation'+' '+bytearray(',' + str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2])))
                else:
                    f.write('translation'+' '+ str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]))
        else:
            print('nan  grid value !', str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]))
            if bool == True:
                if binary:
                    f.write(bytearray(str(0.0) + ' ' + str(0.0) + ' ' + str(0.0)))
                else:
                    f.write(str(0.0) + ' ' + str(0.0) + ' ' + str(0.0))
                bool = False
            else:
                if binary:
                    f.write(bytearray(',' + str(0.0) + ' ' + str(0.0) + ' ' + str(0.0)))
                else:
                    f.write(',' + str(0.0) + ' ' + str(0.0) + ' ' + str(0.0))

        st = """}
                Sphere { radius 10.0 }
             }
          }"""
        if binary:
            f.write(bytearray(st))
        else:
            f.write(st)
        f.close()


def writegrid(pts):
    bool = True
    st = """Coordinate3 { 
               point [
               """

    if binary:
        f = open(filepath, 'a+b')
        f.write(bytearray(st))
    else:
        f = open(filepath, 'a+')
        f.write(st)

    for pt in pts:
        if not np.isnan(pt[0]) and not np.isnan(pt[1]) and not np.isnan(pt[2]):
            if bool == True:
                if binary:
                    f.write(bytearray(str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2])))
                else:
                    f.write(str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]))
                bool = False
            else:
                if binary:
                    f.write(bytearray(',' + str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2])))
                else:
                    f.write(',' + str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]))
        else:
            print('nan  grid value !', str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]))
            if bool == True:
                if binary:
                    f.write(bytearray(str(0.0) + ' ' + str(0.0) + ' ' + str(0.0)))
                else:
                    f.write(str(0.0) + ' ' + str(0.0) + ' ' + str(0.0))
                bool = False
            else:
                if binary:
                    f.write(bytearray(',' + str(0.0) + ' ' + str(0.0) + ' ' + str(0.0)))
                else:
                    f.write(',' + str(0.0) + ' ' + str(0.0) + ' ' + str(0.0))
    st = """]
         }
       """
    if binary:
        f.write(bytearray(st))
    else:
        f.write(st)
    f.close()


def writeindices(inds):
    bool = True
    st = """IndexedFaceSet { 
               coordIndex [
               """

    if binary:
        f = open(filepath, 'a+b')
        f.write(bytearray(st))
    else:
        f = open(filepath, 'a+')
        f.write(st)

    for i in inds:
        if not np.isnan(i[0]) and not np.isnan(i[1]) and not np.isnan(i[2]):
            if bool == True:
                if binary:
                    f.write(bytearray(str(i[0]) + ',' + str(i[1]) + ',' + str(i[2])))
                else:
                    f.write(str(i[0]) + ',' + str(i[1]) + ',' + str(i[2]))
                bool = False
            else:
                if binary:
                    f.write(bytearray(',-1,' + str(i[0]) + ',' + str(i[1]) + ',' + str(i[2])))
                else:
                    f.write(',-1,' + str(i[0]) + ',' + str(i[1]) + ',' + str(i[2]))
        else:
            print('nan indices !')

    st = """]
         }
       """
    if binary:
        f.write(bytearray(st))
    else:
        f.write(st)

    f.close()


if binary:
    st = """#Inventor V2.0 ascii
            """
else:
    st = """#Inventor V2.0 ascii
            """



if binary:
    f = open(filepath, 'a+b')
    f.write(bytearray(st))
else:
    f = open(filepath, 'a+')
    f.write(st)

f.close()

# writegrid(points)
# writeindices(indices)
writeSpheres(points)

st = """
     }
     """
if binary:
    f = open(filepath, 'a+b')
    f.write(bytearray(st))
else:
    f = open(filepath, 'a+')
    f.write(st)
f.close()