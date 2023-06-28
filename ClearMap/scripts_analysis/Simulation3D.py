from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from sklearn import preprocessing
import pylab as plt
import numpy as np
from numpy import linalg as LA
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import ClearMap.Analysis.Graphs.GraphGt as ggt;
import ClearMap.Visualization.Plot3d as p3d
import os
from multiprocessing import Pool
import time
import ClearMap.Analysis.Graphs.GraphGt_new as ggtn
import ClearMap.Analysis.Graphs.GraphGt_updated as ggtu
import ClearMap.Analysis.Graphs.GraphGt_old as ggto
from scipy.spatial import Voronoi
import graph_tool.inference as gti
import graph_tool.generation as gtg

IDs=0



def gaussian(x, a, x0, sigma):
    return a * np.exp(-(np.power((x - x0), 2.)/(2*(sigma**2))))* (1. / (sigma*math.sqrt(2 * math.pi)))

def Xgaussianderivative3D(x,y,z, a, x0, y0,z0, sigma):
    # print(x, x0, a, sigma)
    res= - (x - x0)*a * (1. / 2 * math.pi*sigma**4) * np.exp(-(1./(2*(sigma**2))) * (np.power((x - x0), 2.)+np.power((y - y0), 2.)+np.power((z - z0), 2.)))
    # print(- x*a * (1. / 2 * math.pi*sigma**4) )
    # print(np.exp(-(1./(2*(sigma**2))) * (np.power((x - x0), 2.)+np.power((y - y0), 2.))))
    return res

def Ygaussianderivative3D(x,y,z, a, x0, y0,z0, sigma):
    # print(y, y0, a, sigma)
    # print(- y * a * (1. / 2 * math.pi * sigma ** 4))
    # print(np.exp(-(1. / (2 * (sigma ** 2))) * (np.power((x - x0), 2.) + np.power((y - y0), 2.))))
    res=- (y - y0)*a * (1. / 2 * math.pi*sigma**4) * np.exp(-(1./(2*(sigma**2))) * (np.power((x - x0), 2.)+np.power((y - y0), 2.)+np.power((z - z0), 2.)))
    return res


def Zgaussianderivative3D(x,y,z, a, x0, y0,z0, sigma):
    # print(y, y0, a, sigma)
    # print(- y * a * (1. / 2 * math.pi * sigma ** 4))
    # print(np.exp(-(1. / (2 * (sigma ** 2))) * (np.power((x - x0), 2.) + np.power((y - y0), 2.))))
    res=- (z - z0)*a * (1. / 2 * math.pi*sigma**4) * np.exp(-(1./(2*(sigma**2))) * (np.power((x - x0), 2.)+np.power((y - y0), 2.)+np.power((z - z0), 2.)))
    return res

def sigmoid(Z,a,b):
    return [a/(b + np.exp(-z)) for z in Z]


def reverted_sigmoid(Z,a,b):
    return np.exp(a*(Z-b)) / (1 + np.exp(a*(Z-b)))


def generateNeurons(Nb=1000, size=[0,100], weightype='gaussian', a=1, sigma=10):
    xweight = np.ones(size[1]) / size[1]
    zweight = np.ones(size[1]) / size[1]
    if weightype=='random':
        yweight = np.ones(size[1]) / size[1]
        yweight = np.nan_to_num(yweight)

    elif weightype=='gaussian':
        x = np.linspace(0, size[1], size[1])
        yweight = gaussian(x, a, size[1]*30/100, sigma )
        yweight = np.nan_to_num(yweight)

    zposition = np.random.choice(range(size[1]), int(np.round(Nb)), p=zweight / np.sum(zweight), replace=True)[:,np.newaxis]
    yposition=np.random.choice(range(size[1]), int(np.round(Nb)), p=yweight / np.sum(yweight), replace=True)[:, np.newaxis]
    xposition=np.random.choice(range(size[1]), int(np.round(Nb)), p=xweight / np.sum(xweight), replace=True)[:, np.newaxis]

    neuronposition=np.concatenate((xposition,yposition,zposition), axis=1)
    return neuronposition


def generateCapillariesTC(Nb, vasccells,size=[0,100], weightype=gaussian, a=1, sigma=10):

    weight=[gaussian(vasccells[i][1], a, size[1]*40/100, sigma ) for i in range(len(vasccells))]

    indices = np.random.choice(range(len(vasccells)), int(np.round(Nb)), p=weight / np.sum(weight), replace=False)[:,
                np.newaxis]


    tipsccells= [vasccells[id[0]]  for id in indices]

    return tipsccells

def neuronActivity(neuronposition, activitylevel, activitylocation):
    return


def O2production(vascposition, pos):
    O2 = [1. / (1+(math.pow(math.sqrt( ((pos[0]-vascposition[i][0])**2)+((pos[1]-vascposition[i][1])**2)+((pos[2]-vascposition[i][2])**2)), 2))) for i in range(len(vascposition))]
    return np.sum(O2)


def vegflevel(neuronposition,vascposition, siga, sigb):
    O2=[np.log(O2production(vascposition,pos)) for pos in neuronposition]
    # O2 = [O2production(vascposition, pos) for pos in neuronposition]
    vegf=reverted_sigmoid(np.array(O2), siga, sigb)
    # print(O2)
    # print(vegf)
    return vegf, O2


def computeRepulsion(t, tipscells, a, sigma):
    # repulsioncontib=[((tc[0]-tipscells[i][0]) / (1+math.pow(math.sqrt((tc[0]-tipscells[i][0])**2), 2)), (tc[1]-tipscells[i][1])/ (1+math.pow(math.sqrt(((tc[1]-tipscells[i][1])**2)), 2))) for i in range(len(tipscells))]
    # repulsioncontib=np.stack(repulsioncontib)
    # return np.sum(repulsioncontib, axis=0)

    Xvegftc = np.sum(
        [Xgaussianderivative3D(t[0], t[1],t[2], a, tipscells[i][0], tipscells[i][1],tipscells[i][2], sigma) for i in range(len(tipscells))])
    Xvegftc=Xvegftc- Xgaussianderivative3D(t[0], t[1], t[2], a, t[0], t[1], t[2], sigma)
    Yvegftc = np.sum(
        [Ygaussianderivative3D(t[0], t[1],t[2], a, tipscells[i][0], tipscells[i][1],tipscells[i][2], sigma) for i in
         range(len(tipscells))])
    Yvegftc=Yvegftc- Ygaussianderivative3D(t[0], t[1],t[2], a, t[0], t[1],t[2], sigma)
    Zvegftc = np.sum(
        [Zgaussianderivative3D(t[0], t[1],t[2], a, tipscells[i][0], tipscells[i][1],tipscells[i][2], sigma) for i in
         range(len(tipscells))])
    Zvegftc=Zvegftc- Zgaussianderivative3D(t[0], t[1],t[2], a, t[0], t[1],t[2], sigma)

    norm = LA.norm(np.array([Xvegftc, Yvegftc,Zvegftc]))
    print(Xvegftc,Yvegftc,Zvegftc,norm)
    if norm==0:
        return np.array([0.0, 0.0, 0.0])
    else:
        repulsioncontib = np.array([Xvegftc / norm, Yvegftc / norm, Zvegftc / norm])
    return repulsioncontib


def computevegfgradient(neuronposition,vsc,tps, tipscells_reco,siga, sigb, lr, a_rep, sigma_rep,tipscells_age,edges, vID,ids, sizemax=100):
    vegf, O2=vegflevel(neuronposition,vsc, siga, sigb)
    tipscells_new=[]
    tipscells_new_age=[]
    vID_new=[]
    print(len(tps))
    for n, tc in enumerate(tps):
        print(n, tc,tps)
        Xvegftc=np.sum([Xgaussianderivative3D(tc[0], tc[1],tc[2], vegf[i], neuronposition[i, 0], neuronposition[i, 1],neuronposition[i, 2], sigma) for i in range(neuronposition.shape[0])])
        Yvegftc=np.sum([Ygaussianderivative3D(tc[0], tc[1],tc[2], vegf[i], neuronposition[i, 0], neuronposition[i, 1],neuronposition[i, 2], sigma) for i in range(neuronposition.shape[0])])
        Zvegftc = np.sum([Zgaussianderivative3D(tc[0], tc[1],tc[2], vegf[i], neuronposition[i, 0], neuronposition[i, 1],neuronposition[i, 2], sigma) for i in range(neuronposition.shape[0])])
        norm=LA.norm(np.array([Xvegftc,Yvegftc,Zvegftc]))

        # preprocessing.normalize(np.array([Xvegftc,Yvegftc]).reshape(1, -1), norm='l2').reshape((2))
        print(norm)
        if norm<=50000:
            vsc.append(tc)

        # if norm<=300000:
        #     tipscells_reco.append(tc)
        else:
            # vsc.append(tc)
            repulsion=computeRepulsion(tc, vsc, a_rep, sigma_rep)
            # Xvegftc=Xvegftc/repulsion[0]
            # Yvegftc = Yvegftc / repulsion[1]
            # norm = LA.norm(np.array([Xvegftc, Yvegftc]))
            vector = np.array([Xvegftc / norm, Yvegftc / norm, Zvegftc/norm])
            new_tc=(tc[0]+(vegf_coeff*vector[0]), tc[1]+(vegf_coeff*vector[1]),tc[2]+(vegf_coeff*vector[2]))
            print(new_tc)
            if compute_vein==False:
                vect_f=((vegf_coeff*vector[0]) - (rep_coeff*repulsion[0]),
                        (vegf_coeff*vector[1])-(rep_coeff*repulsion[1]),
                        (vegf_coeff*vector[2])-(rep_coeff*repulsion[2]))
                norm = LA.norm(np.array([vect_f[0], vect_f[1],vect_f[2]]))
                vect_f = np.array([vect_f[0] / norm, vect_f[1] / norm, vect_f[2] / norm])
                new_tc = (lr * (tc[0] +vect_f[0]), lr *  (tc[1] +vect_f[1]), lr * (tc[2] +vect_f[2]))
                # new_tc=(lr*(tc[0]+(vegf_coeff*vector[0]) - (rep_coeff*repulsion[0])), lr*(tc[1]+(vegf_coeff*vector[1])-(rep_coeff*repulsion[1])))
            else:
                Xvegftc = np.sum(
                    [Xgaussianderivative3D(tc[0], tc[1],tc[2], 1, veins[i][0], veins[i][1],veins[i][2], 20) for i in
                     range(len(veins))])
                Yvegftc = np.sum(
                    [Ygaussianderivative3D(tc[0], tc[1],tc[2], 1, veins[i][0], veins[i][1],veins[i][2], 20) for i in
                     range(len(veins))])
                Zvegftc = np.sum(
                    [Ygaussianderivative3D(tc[0], tc[1], tc[2], 1, veins[i][0], veins[i][1], veins[i][2], 20) for i in
                     range(len(veins))])
                norm = LA.norm(np.array([Xvegftc, Yvegftc,Zvegftc]))
                vectorvein = np.array([Xvegftc / norm, Yvegftc / norm, Zvegftc/norm])

                vect_f = ((vegf_coeff * vector[0]) - (rep_coeff*repulsion[0])+ (vein_coeff * vectorvein[0]),
                           (vegf_coeff * vector[1]) - (rep_coeff * repulsion[1]) + (vein_coeff * vectorvein[1]),
                          (vegf_coeff * vector[2]) - (rep_coeff * repulsion[2]) + (vein_coeff * vectorvein[2]))
                norm = LA.norm(np.array([vect_f[0], vect_f[1], vect_f[2]]))
                vect_f = np.array([vect_f[0] / norm, vect_f[1] / norm, vect_f[2] / norm])
                new_tc = (lr * (tc[0] + vect_f[0]), lr * (tc[1] + vect_f[1]), lr * (tc[2] + vect_f[2]))

                # new_tc = (lr*(tc[0] + (vegf_coeff * vector[0]) - (rep_coeff*repulsion[0]))+ (vein_coeff * vectorvein[0]),
                #           lr * (tc[1] + (vegf_coeff * vector[1]) - (rep_coeff * repulsion[1])) + (vein_coeff * vectorvein[1]))

            if compute_vein == False:
                if ((new_tc[0]<0) or (new_tc[1]<0) or (new_tc[2]<0)):
                    vsc.append(new_tc)

                elif ((new_tc[0] > sizemax[1]) or (new_tc[1] > sizemax[1]) or (new_tc[2] > sizemax[1])):
                    vsc.append(new_tc)

                else:
                    tipscells_new.append(new_tc)
                    vsc.append(new_tc)
                    tipscells_new_age.append(tipscells_age[n]+1)
                    ids = len(vsc)
                    vID_new.append(ids)
                    edges.append((vID[n], ids))
            else :
                if new_tc in veins:
                    vsc.append(tc)

                else:
                    ids = ids + 1
                    vID_new.append(ids)
                    tipscells_new.append(new_tc)
                    vsc.append(new_tc)
                    tipscells_new_age.append(tipscells_age[n] + 1)
                    ids = len(vsc)
                    vID_new.append(ids)
                    edges.append((vID[n], ids))

    # tipscells_age=np.array(tipscells_new_age)
    print(tipscells_new_age, tipscells_new)
    return tipscells_new, vsc, vegf, O2,tipscells_reco,tipscells_new_age,edges, vID_new,ids


def computerecogradient(vasccells,tipscells_reco,tipscells_age, sizemax=100):
    tipscells_new = []
    tipscells_new_age=[]
    for n,tc in enumerate(tipscells_reco):
        Xvegftc = np.sum(
            [Xgaussianderivative3D(tc[0], tc[1], tc[2], vegf[i], vasccells[i][0], vasccells[i][1],vasccells[i][2], sigma) for i in
             range(len(vasccells))])
        Yvegftc = np.sum(
            [Ygaussianderivative3D(tc[0], tc[1], tc[2], vegf[i], vasccells[i][0], vasccells[i][1],vasccells[i][2], sigma) for i in
             range(len(vasccells))])
        Zvegftc = np.sum(
            [Zgaussianderivative3D(tc[0], tc[1], tc[2], vegf[i], vasccells[i][0], vasccells[i][1], vasccells[i][2],sigma) for i in
             range(len(vasccells))])

        norm = LA.norm(np.array([Xvegftc, Yvegftc,Zvegftc]))

        # preprocessing.normalize(np.array([Xvegftc,Yvegftc]).reshape(1, -1), norm='l2').reshape((2))
        print(norm)

        vasccells.append(tc)

        vector = np.array([Xvegftc / norm, Yvegftc / norm,Zvegftc / norm])
        new_tc=(tc[0]+(vegf_coeff*vector[0]), tc[1]+(vegf_coeff*vector[1]),tc[2]+(vegf_coeff*vector[2]))
        print(new_tc)

        if ((new_tc[0]<0) or (new_tc[1]<0) or (new_tc[2]<0)):
            vasccells.append(tc)
        elif ((new_tc[0] > sizemax[1]) or (new_tc[1] > sizemax[1]) or (new_tc[2] > sizemax[1])):
            vasccells.append(tc)
        else:
            tipscells_new.append(new_tc)
            tipscells_new_age.append(tipscells_age[n] + 1)
    return tipscells_new, vasccells




def computerecoVeins(vasccells,tipscells,veins, sizemax=100):
    tipscells_new = []
    for tc in tipscells:
        # compute vegf contribution
        vegfX = np.sum(
            [Xgaussianderivative3D(tc[0], tc[1], vegf[i], neuronposition[i, 0], neuronposition[i, 1], sigma) for i in
             range(neuronposition.shape[0])])
        vegfY = np.sum(
            [Ygaussianderivative3D(tc[0], tc[1], vegf[i], neuronposition[i, 0], neuronposition[i, 1], sigma) for i in
             range(neuronposition.shape[0])])
        normvegf = LA.norm(np.array([vegfX, vegfY]))

        # compute vein contribution
        Xvegftc = np.sum(
            [Xgaussianderivative3D(tc[0], tc[1], 1, veins[i][0], veins[i][1], 20) for i in
             range(len(veins))])
        Yvegftc = np.sum(
            [Ygaussianderivative3D(tc[0], tc[1], 1, veins[i][0], veins[i][1], 20) for i in
             range(len(veins))])
        norm = LA.norm(np.array([Xvegftc, Yvegftc]))

        # preprocessing.normalize(np.array([Xvegftc,Yvegftc]).reshape(1, -1), norm='l2').reshape((2))
        print(Xvegftc,Yvegftc)

        vasccells.append(tc)

        vectorvegf=np.array([vegfX / normvegf, vegfY / normvegf])
        vectorvein = np.array([Xvegftc / norm, Yvegftc / norm])
        new_tc=(tc[0]+(vegf_coeff*vectorvegf[0])+ (vein_coeff*vectorvein[0]), tc[1]+(vegf_coeff*vectorvegf[1])+(vein_coeff*vectorvein[1]))
        print(new_tc)

        if ((new_tc[0]<0) or (new_tc[1]<0)):
            vasccells.append(tc)
        elif ((new_tc[0] > sizemax[1]) or (new_tc[1] > sizemax[1])):
            vasccells.append(tc)
        else:
            tipscells_new.append(new_tc)
    return tipscells_new, vasccells

def forkingUpdate(vasccells,tipscells, tipscells_age, vID, edges, ids, length=10, proba=30):

    # i=0
    # if len(vasccells)>length:
    #     vasc=vasccells[-length:]
    # else:
    #     vasc=vasccells
    # for vc in vasc:
    #     if i==0:
    #         r = random.randint(0, proba)
    #         # print(r)
    #         if r==7:
    #             tipscells.append(vc)
    #             i=7
    #     else:
    #         i=i-1
    for n, tc in enumerate(tipscells):
        if tipscells_age[n]>0:
            r = [random.randint(0, proba) for i in range(tipscells_age[n])]
            # if r == 1:
            if (1 in r):

                tipscells.append((tc[0]-1, tc[1]-1,tc[2]-1))
                vasccells.append((tc[0]-1, tc[1]-1,tc[2]-1))
                ids = len(vasccells)
                vID.append(ids)
                tipscells_age[n]=0
                tipscells_age.append(0)
                edges.append((vID[n], ids))

    return(tipscells,vasccells, ids,vID, edges)





def graphUpdate(graph, vegf,tps, vsc, tps2, vsc2, col='r', col2='b'):
    # fig.clear()

    # ax.scatter(neuronposition[:, 0], neuronposition[:, 1],neuronposition[:, 2],c=vegf, alpha=0.3)
    T=np.stack(tps)
    V=np.stack(vsc)

    T2 = np.stack(tps2)
    V2 = np.stack(vsc2)
    # Veins=np.stack(veins)
    # graph.set_ydata(V)

    # array = graph.get_offsets()
    #
    # # add the points to the plot
    # array = np.append(array, T)
    # graph.set_offsets(array, c='g')
    #
    # array = graph.get_offsets()
    #
    # # add the points to the plot
    # array = np.append(array, V)
    # graph.set_offsets(array, c='r')
    # plt.colorbar()
    ax.scatter(V[:, 0], V[:, 1], V[:, 2], c=col)
    ax.scatter(T[:, 0], T[:, 1],T[:, 2], c='g')

    ax.scatter(V2[:, 0], V2[:, 1],V2[:, 2], c=col2)
    ax.scatter(T2[:, 0], T2[:, 1],T2[:, 2], c='g')

    # plt.scatter(Veins[:, 0], Veins[:, 1], c='b', s=100)
    plt.draw()
    plt.pause(0.01)
    # plt.colorbar()


#
#
# def graphUpdate(graph, vegf,tipscells, vasccells, color='r', color2='b'):
#     # fig.clear()
#
#     ax.scatter(neuronposition[:, 0], neuronposition[:, 1],neuronposition[:, 2],c=vegf)
#     T=np.stack(tipscells)
#     V=np.stack(vasccells)
#
#     # T2 = np.stack(tipscells2)
#     # V2 = np.stack(vasccells2)
#     # Veins=np.stack(veins)
#     # graph.set_ydata(V)
#
#     # array = graph.get_offsets()
#     #
#     # # add the points to the plot
#     # array = np.append(array, T)
#     # graph.set_offsets(array, c='g')
#     #
#     # array = graph.get_offsets()
#     #
#     # # add the points to the plot
#     # array = np.append(array, V)
#     # graph.set_offsets(array, c='r')
#     # plt.colorbar()
#     ax.scatter(V[:, 0], V[:, 1], V[:, 2], c=color)
#     ax.scatter(T[:, 0], T[:, 1],T[:, 2], c='g')
#
#     # ax.scatter(V2[:, 0], V2[:, 1],V2[:, 2], c=color2)
#     # ax.scatter(T2[:, 0], T2[:, 1],T2[:, 2], c='g')
#
#     # plt.scatter(Veins[:, 0], Veins[:, 1], c='b', s=100)
#     plt.draw()
#     plt.pause(0.01)
#     # plt.colorbar()


def plotGraph(V, E, mins=[0, 0, 0], maxs=[100, 100, 100]):
    plt.ion()
    fig, ax = plt.subplots()
    ax = plt.axes(projection='3d')

    vertices = V


    VecStart_x = vertices[E[:, 0]][:, 0]
    VecEnd_x = vertices[E[:, 1]][:, 0]

    VecStart_y = vertices[E[:, 0]][:, 1]
    VecEnd_y = vertices[E[:, 1]][:, 1]

    VecStart_z = vertices[E[:, 0]][:, 2]
    VecEnd_z = vertices[E[:, 1]][:, 2]

    # mins = [0, 0, 0]
    # maxs = [100, 100, 100]

    isOverStart = (np.concatenate(
        (np.concatenate((VecStart_x[:, np.newaxis], VecStart_y[:, np.newaxis]), axis=1), VecStart_z[:, np.newaxis]),
        axis=1) > mins).all(axis=1)
    isUnderStart = (np.concatenate(
        (np.concatenate((VecStart_x[:, np.newaxis], VecStart_y[:, np.newaxis]), axis=1), VecStart_z[:, np.newaxis]),
        axis=1) < maxs).all(axis=1)

    isOverEnd = (np.concatenate(
        (np.concatenate((VecEnd_x[:, np.newaxis], VecEnd_y[:, np.newaxis]), axis=1), VecEnd_z[:, np.newaxis]),
        axis=1) > mins).all(axis=1)
    isUnderEnd = (np.concatenate(
        (np.concatenate((VecEnd_x[:, np.newaxis], VecEnd_y[:, np.newaxis]), axis=1), VecEnd_z[:, np.newaxis]),
        axis=1) < maxs).all(axis=1)

    mask = isOverStart * isUnderStart * isOverEnd * isUnderEnd

    VecStart_x = VecStart_x[mask]
    VecEnd_x = VecEnd_x[mask]
    VecStart_y = VecStart_y[mask]
    VecEnd_y = VecEnd_y[mask]
    VecStart_z = VecStart_z[mask]
    VecEnd_z = VecEnd_z[mask]

    for i in range(VecStart_x.shape[0]):
        # if (abs(VecStart_x[i])<100 and abs(VecEnd_x[i]<100) and abs(VecStart_y[i]<100)  and abs(VecEnd_y[i]<100)  and abs(VecStart_z[i]<100)  and VecEnd_z[i]<100 ):
        ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i], VecEnd_y[i]], [VecStart_z[i], VecEnd_z[i]], c='r')

def fromEVtoGraph(V, E, Vartshape,Vveinshape, VID0,VID02,radius=1, ratio=1):
    n_vertices = V.shape[0]
    print('n_vertices ', n_vertices)
    g = ggtu.Graph(n_vertices=n_vertices - 1, directed=False)
    print(g)
    # try:
    #     g.add_edge(E)
    # except:
    i=0
    for e in E:
        # i=i+1
        # print(i)
        try:
            g.add_edge([e])
        except:
            print(e)
    #radii=np.ones(edges_all.shape[0])
    print(g)
    radii=np.ones(g.n_edges)*radius
    g.set_edge_geometry(name='radii', values=radii)
    g.set_vertex_coordinates(V*ratio)
    arteries=np.zeros(g.n_vertices)
    veins = np.zeros(g.n_vertices)
    arteries[:Vartshape+1]=1
    veins[Vartshape+1:Vartshape+Vveinshape] = 1
    g.add_vertex_property('arteries', arteries)
    g.add_vertex_property('veins', veins)
    radii[:Vartshape] = 3
    radii[Vartshape:Vartshape+Vveinshape] = 3
    g.add_edge_property('radii', radii)
    g.set_edge_geometry(name='radii', values=radii)
    radii = np.ones(g.n_vertices) * radius
    radii[arteries.astype(bool)] = 3
    radii[veins.astype(bool)] = 3
    g.add_vertex_property('radii', radii)
    startPoint=np.zeros(g.n_vertices)
    startPoint[VID0]=1
    startPoint[Vartshape+np.array(VID02)+1]=1
    startPoint[[0,Vartshape+1] ]=1
    g.add_vertex_property('startPoint', startPoint)
    return(g)

def fromVoronoi2Graph(V, E, Vartshape,Vveinshape, VID0,VID02,radius=1, ratio=1):
    n_vertices = V.shape[0]
    print('n_vertices ', n_vertices)
    g = ggtu.Graph(n_vertices=n_vertices - 1, directed=False)
    print(g)
    g.add_edge(E)
    #radii=np.ones(edges_all.shape[0])
    print(g)
    radii=np.ones(E.shape[0])*radius
    g.set_edge_geometry(name='radii', values=radii)
    g.set_vertex_coordinates(V*ratio)
    arteries=np.zeros(g.n_vertices)
    veins = np.zeros(g.n_vertices)
    arteries[:Vartshape+1]=1
    veins[Vartshape+1:Vartshape+Vveinshape] = 1
    g.add_vertex_property('arteries', arteries)
    g.add_vertex_property('veins', veins)
    radii[:Vartshape] = 3
    radii[Vartshape:Vartshape+Vveinshape] = 3
    g.add_edge_property('radii', radii)
    g.set_edge_geometry(name='radii', values=radii)
    radii = np.ones(g.n_vertices) * radius
    radii[arteries.astype(bool)] = 3
    radii[veins.astype(bool)] = 3
    g.add_vertex_property('radii', radii)
    startPoint=np.zeros(g.n_vertices)
    startPoint[VID0]=1
    startPoint[Vartshape+np.array(VID02)+1]=1
    startPoint[[0,Vartshape+1] ]=1
    g.add_vertex_property('startPoint', startPoint)
    return(g)


def extractSubGraph(edges_centers, mins, maxs):
    """
    Extracts the subgraph contained in the cube between mins and maxs coordinates
          6-------7
         /|      /|
        4-------5 |
        | |     | |
        | 2-----|-3
        |/      |/
        0-------1
    """
    isOver = (edges_centers > mins).all(axis=1)
    isUnder = (edges_centers < maxs).all(axis=1)
    return(np.asarray(np.logical_and(isOver, isUnder)))#.nonzero()[0])


def get_distances(args):
    n, pt, cell,coordinates,capillaries=args

    print(n, pt, cell)

    mins =cell - 150
    maxs = cell + 150
    close_edges_centers_indices=np.logical_and(capillaries, extractSubGraph(coordinates, mins, maxs)).nonzero()[0]
    print(close_edges_centers_indices)
    close_coordinate = [coordinates[cec] for cec in close_edges_centers_indices]
    try:
        close_coordinate=np.vstack(close_coordinate)
        close_distance=np.array([np.linalg.norm(close_coordinate[i]-cell) for i in range(close_coordinate.shape[0])])
        close_distance[np.argwhere(close_distance == 0)]=1000
        return((pt, close_edges_centers_indices[np.argmin(close_distance)]))
    except:
        print('no vessels found')
        # return((np.nan, np.nan))


def mutualLoopDetection(args):
    res=0
    i, conn = args
    co = conn[i]
    print(i, co)
    similaredges = np.logical_or(np.logical_and(conn[:, 0] == co[0], conn[:, 1] == co[1]),
                                 np.logical_and(conn[:, 1] == co[0], conn[:, 0] == co[1]))
    # print(similaredges)
    similaredges = np.asarray(similaredges).nonzero()[0]
    print(similaredges)
    return similaredges



IDs=0
IDs2=0
NbArt=2
NbVein=2
sigma=20
a=1
neuronposition=generateNeurons(Nb=100, size=[0,100], weightype='gaussian',a=a, sigma=sigma)

veins=generateNeurons(Nb=20, size=[0,100], weightype='random',a=a, sigma=sigma)

vegf=np.zeros(neuronposition.shape[0])
plt.ion()
fig, ax = plt.subplots()
ax = plt.axes(projection='3d')
# plt.plot(neuronposition[:,0], neuronposition[:,1], 'ro')[0]
graph = ax.scatter(neuronposition[:,0], neuronposition[:,1],neuronposition[:,2],c=vegf, alpha=0.05)
plt.pause(0.01)





xs=np.random.choice(np.arange(100), NbArt)
zs=np.random.choice(np.arange(100), NbArt)
ys=15*np.ones(NbArt)

vasccells=np.concatenate((xs, ys, zs), axis=0).astype(int)
vasccells=vasccells.reshape((3,NbArt)).transpose().tolist()
vasccells=[tuple(vasccells[k]) for k in range(len(vasccells))]
# vasccells=[(58.0, 20.0, 8.0),(88.0, 20.0, 90.0),(2.0, 20.0, 3.0),(12.0, 20.0, 19.0),(81.0, 20.0, 82.0),(48.0, 20.0, 30.0),(96.0, 20.0, 54.0),(25.0, 20.0, 44.0)]
# vasccells=[(15, 20 ,47),(33, 20 ,25),(89, 20 ,12),(65, 20 ,78),(42, 20 ,80),(56, 20 ,63)]#[(30,10, 50)]#, (100,0)]
tipscells=vasccells.copy()#[(15, 20 ,47),(33, 20 ,25),(89, 20 ,12),(65, 20 ,78),(42, 20 ,80),(56, 20 ,63)]#[(30,10, 50)]#, (100,0)]
tipscells_age=np.zeros(len(tipscells)).astype(int)





xs=np.random.choice(np.arange(100), NbVein)
zs=np.random.choice(np.arange(100), NbVein)
ys=15*np.ones(NbVein)

vasccells2=np.concatenate((xs, ys, zs), axis=0).astype(int)
vasccells2=vasccells2.reshape((3,NbVein)).transpose().tolist()
vasccells2=[tuple(vasccells2[k]) for k in range(len(vasccells2))]
# vasccells2=[(14.0, 20.0, 58.0),(11.0, 20.0, 95.0),(24.0, 20.0, 82.0),(93.0, 20.0, 67.0),(28.0, 20.0, 20.0),(75.0, 20.0, 6.0)]
# vasccells2=[(41, 20, 39 ),(9, 20 ,85),(69, 20 ,18),(30, 20 ,45)]#[(60,10, 50)]#, (100,0)]
tipscells2=vasccells2.copy()#[(41, 20, 39 ),(9, 20 ,85),(69, 20 ,18),(30, 20 ,45)]#[(60,10, 50)]#, (100,0)]
tipscells_age2=np.zeros(len(tipscells2)).astype(int)





edgestree=[]
vertexID=np.arange(NbArt).tolist()
VID0=vertexID
vertexID2=np.arange(NbVein).tolist()
VID02=vertexID2
edgestree2=[]

tipscells_reco2=[]
tipscells_reco=[]
# ax.colorbar()

rep_coeff=10
vegf_coeff=5#0.4
vein_coeff=10
siga=-1#1.5
sigb=0.01
N=80

a=2
sigma=10#10
# a=10
a_rep=1
sigma_rep=3#3
lr=1
compute_vein=False
N=30
while N>0:
    print(N)
    tipscells,vasccells, vegf, O2, tipscells_reco,tipscells_age,edgestree, vertexID, IDs=computevegfgradient(neuronposition,vasccells,
                                                                                            tipscells,tipscells_reco,siga,
                                                                                            sigb, lr, a_rep, sigma_rep,
                                                                                            tipscells_age,edgestree, vertexID, IDs, sizemax=[0,100])
    # tipscells_reco,vasccells=computerecogradient(vasccells,tipscells_reco,tipscells_age, sizemax=[0,100])
    tipscells, vasccells,IDs,vertexID, edgestree= forkingUpdate(vasccells,tipscells,tipscells_age, vertexID, edgestree, IDs, length=10, proba=100)#50#100
    # graphUpdate(graph, vegf, tipscells, vasccells, 'r')

    tipscells2, vasccells2, vegf, O2, tipscells_reco2, tipscells_age2,edgestree2, vertexID2, IDs2 = computevegfgradient(neuronposition, vasccells2,
                                                                                        tipscells2, tipscells_reco2, siga,
                                                                                        sigb, lr, a_rep, sigma_rep,
                                                                                        tipscells_age2, edgestree2, vertexID2, IDs2, sizemax=[0, 100])
    # tipscells_reco2, vasccells2 = computerecogradient(vasccells2, tipscells_reco2, tipscells_age2, sizemax=[0, 100])
    tipscells2, vasccells2, IDs2,vertexID2,edgestree2 = forkingUpdate(vasccells2, tipscells2, tipscells_age2, vertexID2, edgestree2, IDs2, length=10, proba=200)  # 50 #200
    #

    N=N-1
    # print(tipscells)
    # graphUpdate(graph, vegf, tipscells, vasccells)
    graphUpdate(graph, vegf, tipscells, vasccells, tipscells2, vasccells2, col='r', col2='b')
    # ani = animation.FuncAnimation(fig, graphUpdate, interval=1000)
    # plt.show()




####### plot test
plt.ion()
fig, ax = plt.subplots()
ax = plt.axes(projection='3d')

vertices=np.array(vasccells)
E=np.array(edgestree)
E[0]=(1,2)
E=E-1

vertices2=np.array(vasccells2)
E2=np.array(edgestree2)
E2[0]=(1,2)
E2=E2-1

Etot=np.concatenate((E,E2+np.max(E)+1), axis=0)
verticestot=np.concatenate((vertices,vertices2), axis=0)
# vertices=np.array(vasccells2)
# E=np.array(edgestree2)


vertices=verticestot
E=Etot
plotGraph(vertices, E, mins=[0, 0, 0], maxs=[100, 100, 100])



directory = "V20"

parent_dir = "/data_SSD_2to/SimulationVasculature3D"
path = os.path.join(parent_dir, directory)

os.mkdir(path)
print("Directory '% s' created" % directory)


## saving
np.save(os.path.join(path, 'tipscells.npy'), tipscells)
np.save(os.path.join(path, 'vasccells.npy'), vasccells)
np.save(os.path.join(path, 'vegf.npy'), vegf)
np.save(os.path.join(path, 'O2.npy'), O2)
np.save(os.path.join(path, 'tipscells_reco.npy'), tipscells_reco)
np.save(os.path.join(path, 'tipscells_age.npy'), tipscells_age)

np.save(os.path.join(path, 'tipscells2.npy'), tipscells2)
np.save(os.path.join(path, 'vasccells2.npy'), vasccells2)
np.save(os.path.join(path, 'tipscells_reco2.npy'), tipscells_reco2)
np.save(os.path.join(path, 'tipscells_age2.npy'), tipscells_age2)

np.save(os.path.join(path, 'rep_coeff.npy'), rep_coeff)
np.save(os.path.join(path, 'vegf_coeff.npy'), vegf_coeff)
np.save(os.path.join(path, 'vein_coeff.npy'), vein_coeff)
np.save(os.path.join(path, 'siga.npy'), siga)
np.save(os.path.join(path, 'sigb.npy'), sigb)
np.save(os.path.join(path, 'a.npy'), a)
np.save(os.path.join(path, 'sigma.npy'), sigma)
np.save(os.path.join(path, 'a_rep.npy'), a_rep)
np.save(os.path.join(path, 'sigma_rep.npy'), sigma_rep)
np.save(os.path.join(path, 'lr.npy'), lr)

np.save(os.path.join(path, 'edgestree.npy'), edgestree)
np.save(os.path.join(path, 'edgestree2.npy'), edgestree2)
np.save(os.path.join(path, 'VID0.npy'), VID0)
np.save(os.path.join(path, 'VID02.npy'), VID02)

## loading
#
tipscells=np.load(os.path.join(path, 'tipscells.npy'))
vasccells=np.load(os.path.join(path, 'vasccells.npy'))
vegf=np.load(os.path.join(path, 'vegf.npy'))
O2=np.load(os.path.join(path, 'O2.npy'))
tipscells_reco=np.load(os.path.join(path, 'tipscells_reco.npy'))
tipscells_age=np.load(os.path.join(path, 'tipscells_age.npy'))

tipscells2=np.load(os.path.join(path, 'tipscells2.npy'))
vasccells2=np.load(os.path.join(path, 'vasccells2.npy'))
tipscells_reco2=np.load(os.path.join(path, 'tipscells_reco2.npy'))
tipscells_age2=np.load(os.path.join(path, 'tipscells_age2.npy'))

rep_coeff=np.load(os.path.join(path, 'rep_coeff.npy'))
vegf_coeff=np.load(os.path.join(path, 'vegf_coeff.npy'))
vein_coeff=np.load(os.path.join(path, 'vein_coeff.npy'))
siga=np.load(os.path.join(path, 'siga.npy'))
sigb=np.load(os.path.join(path, 'sigb.npy'))
a=np.load(os.path.join(path, 'a.npy'))
sigma=np.load(os.path.join(path, 'sigma.npy'))
a_rep=np.load(os.path.join(path, 'a_rep.npy'))
sigma_rep=np.load(os.path.join(path, 'sigma_rep.npy'))
lr=np.load(os.path.join(path, 'lr.npy'))

edgestree=np.load(os.path.join(path, 'edgestree.npy'))
edgestree2=np.load(os.path.join(path, 'edgestree2.npy'))
VID0=np.load(os.path.join(path, 'VID0.npy'))
VID02=np.load(os.path.join(path, 'VID02.npy') )

#
# N=10
# compute_vein=True
# veins=vasccells2
# while N>0:
#     print(N)
#     tipscells, vasccells, vegf, O2, tipscells_reco,tipscells_age = computevegfgradient(neuronposition, vasccells, tipscells,
#                                                                          tipscells_reco, siga, sigb, lr, a_rep,
#                                                                          sigma_rep,tipscells_age, sizemax=[0, 100])
#     # tipscells,vasccells=computerecoVeins(vasccells,tipscells,veins, sizemax=[0,100])
#     tipscells2, vasccells2, vegf, O2, tipscells_reco2, tipscells_age2 = computevegfgradient(neuronposition, vasccells2,
#                                                                                             tipscells2,
#                                                                                             tipscells_reco2, siga, sigb,
#                                                                                             lr, a_rep,
#                                                                                             sigma_rep, tipscells_age2,
#                                                                                             sizemax=[0, 100])
#     # tipscells,vasccells=computerecoVeins(vasccells,tipscells,veins, sizemax=[0,100])
#     graphUpdate(graph, vegf, tipscells, vasccells, tipscells2, vasccells2)
#
#     # graphUpdate(graph, vegf,tipscells, vasccells,tipscells2, vasccells2)
#     N = N - 1
# N=10
# compute_vein=True
# veins=vasccells
# while N>0:
#     tipscells2, vasccells2, vegf, O2, tipscells_reco2,tipscells_age2 = computevegfgradient(neuronposition, vasccells2, tipscells2,
#                                                                          tipscells_reco2, siga, sigb, lr, a_rep,
#                                                                          sigma_rep,tipscells_age2, sizemax=[0, 100])
#     # tipscells,vasccells=computerecoVeins(vasccells,tipscells,veins, sizemax=[0,100])
#     graphUpdate(graph, vegf,tipscells, vasccells,tipscells2, vasccells2)
#     N = N - 1


######## capillaries
plt.xlim(0,100)
plt.ylim(0,100)
# ax.set_zlim(0,100)




def generateNeurons(Nb=1000, size=[0,100], weightype='gaussian', a=1, sigma=10):
    xweight = np.ones(size[1]) / size[1]
    zweight = np.ones(size[1]) / size[1]
    if weightype=='random':
        yweight = np.ones(size[1]) / size[1]
        yweight = np.nan_to_num(yweight)

    elif weightype=='gaussian':
        x = np.linspace(0, size[1], size[1])
        yweight = gaussian(x, a, size[1]*30/100, sigma )
        yweight2 = gaussian(x, a/2, size[1] * 50 / 100, sigma-5)
        yweight = np.nan_to_num(np.maximum(yweight,yweight2))

    zposition = np.random.choice(range(size[1]), int(np.round(Nb)), p=zweight / np.sum(zweight), replace=True)[:,np.newaxis]
    yposition=np.random.choice(range(size[1]), int(np.round(Nb)), p=yweight / np.sum(yweight), replace=True)[:, np.newaxis]
    xposition=np.random.choice(range(size[1]), int(np.round(Nb)), p=xweight / np.sum(xweight), replace=True)[:, np.newaxis]

    neuronposition=np.concatenate((xposition,yposition,zposition), axis=1)
    return neuronposition



# neuronpositionTest=generateNeurons(Nb=5000, size=[0,70], weightype='gaussian',a=a, sigma=sigma)#100000
neuronpositionTest=generateNeurons(Nb=5000, size=[0,100], weightype='gaussian',a=a, sigma=sigma)#100000
neuronpositionTest=neuronpositionTest[neuronpositionTest[:,1]>10]
plt.figure()
sns.distplot(neuronpositionTest[:,1], bins=10)
# graph = ax.scatter(neuronposition[:,0], neuronposition[:,1],neuronposition[:,2],c=vegf, alpha=0.05)
ax = plt.axes(projection='3d')
graphUpdate(graph, vegf, tipscells, vasccells, tipscells2, vasccells2, col='r', col2='b')

type='knn'#'knn'#'voronoi

if type=='voronoi':
    vor = Voronoi(neuronpositionTest)
    ridges=vor.ridge_vertices
    vertices=vor.vertices
    E=[]
    for faces in ridges:
        for i, node in enumerate(faces):
            try:
                E.append((node, faces[i+1]))
            except:
                print("reached end of array", faces)
                # break
                E.append((node, faces[0]))
                break
    E=np.array(E).astype('int32')


    # np.save(os.path.join(path, 'cap_vertices.npy'), vertices)
    # np.save(os.path.join(path, 'cap_edges.npy'), E)




    VecStart_x=vertices[E[:,0]][:,0]
    VecEnd_x=vertices[E[:,1]][:,0]

    VecStart_y=vertices[E[:,0]][:,1]
    VecEnd_y=vertices[E[:,1]][:,1]

    VecStart_z=vertices[E[:,0]][:,2]
    VecEnd_z=vertices[E[:,1]][:,2]

    mins=[5,5,5]
    maxs=[100,70,100]

    isOverStart = (np.concatenate((np.concatenate((VecStart_x[:, np.newaxis],VecStart_y[:, np.newaxis]), axis=1),VecStart_z[:, np.newaxis]), axis=1) > mins).all(axis=1)
    isUnderStart = (np.concatenate((np.concatenate((VecStart_x[:, np.newaxis],VecStart_y[:, np.newaxis]), axis=1),VecStart_z[:, np.newaxis]), axis=1) < maxs).all(axis=1)

    isOverEnd = (np.concatenate((np.concatenate((VecEnd_x[:, np.newaxis],VecEnd_y[:, np.newaxis]), axis=1),VecEnd_z[:, np.newaxis]), axis=1) > mins).all(axis=1)
    isUnderEnd = (np.concatenate((np.concatenate((VecEnd_x[:, np.newaxis],VecEnd_y[:, np.newaxis]), axis=1),VecEnd_z[:, np.newaxis]), axis=1) < maxs).all(axis=1)

    mask=isOverStart * isUnderStart *isOverEnd*isUnderEnd


    V_tot_f=vertices
    E_cap=E[mask]
    isOverV=(vertices > mins).all(axis=1)
    isUndeV=(vertices < maxs).all(axis=1)
    V=vertices[isOverV*isUndeV]


    E_cap=E_cap[np.logical_and(E_cap[:,0]>=0, E_cap[:,1]>=0)]

    Etot_f=E#E_cap
    # V_tot_f=np.concatenate((verticestot, vertices), axis=0)
    # Etot_f=np.concatenate((Etot, E_cap+np.max(Etot)+1), axis=0)

    # np.save(os.path.join(path, 'Etot_f.npy'), Etot_f)
    # np.save(os.path.join(path, 'V_tot_f.npy'),V_tot_f)
    # plotGraph(V_tot_f, Etot_f, mins=[0, 0, 0], maxs=[100, 100, 100])
    ratio=10
    graph=fromEVtoGraph(V_tot_f, Etot_f, 0, 0,0,0, radius=1, ratio=ratio)#ratio=30
    # graph=fromEVtoGraph(V_tot_f, Etot_f, len(edgestree), len(edgestree2),VID0,VID02, radius=0.1, ratio=10)#ratio=30
    # p3d.plot_graph_line(graph)

    vertices=graph.vertex_coordinates()
    E=graph.edge_connectivity()

    mins=[5*ratio,5*ratio,5*ratio]
    maxs=[100*ratio,70*ratio,100*ratio]
    isOverV=(vertices > mins).all(axis=1)
    isUndeV=(vertices < maxs).all(axis=1)
    vertex_filter=isOverV*isUndeV
    graph=graph.sub_graph(vertex_filter=vertex_filter)


    # vertex_filter=np.ones(graph.n_vertices)
    # vertex_filter[graph.vertex_property('arteries').astype(bool)]=0
    # vertex_filter[graph.vertex_property('veins').astype(bool)]=0
    # graph = graph.sub_graph(vertex_filter=vertex_filter)
    graph=graph.largest_component()
    import graph_tool.topology as gtt
    import graph_tool
    tree = gtt.min_spanning_tree(graph.base).a
    edge_filter=tree.astype(bool)
    graph = graph.sub_graph(edge_filter=edge_filter)
    p3d.plot_graph_line(graph)


    art_ain_candidates=np.asarray(graph.vertex_coordinates()[:,1]<np.min(graph.vertex_coordinates()[:,1])+10).nonzero()[0]
    art=np.random.choice(art_ain_candidates, 3)

    new_art_candidates = np.delete(art_ain_candidates, np.where(art_ain_candidates == art[1]))
    new_art_candidates = np.delete(art_ain_candidates, np.where(art_ain_candidates == art[0]))
    new_art_candidates = np.delete(art_ain_candidates, np.where(art_ain_candidates == art[2]))

    vein=np.random.choice(new_art_candidates, 2)

    arteries=np.zeros(graph.n_vertices)
    veins=np.zeros(graph.n_vertices)
    arteries[art]=1
    veins[vein]=1


    #tracing
    iter=10
    i=iter
    while i >0:
        print(i, art, vein)
        for j, a in enumerate(art):
            print(np.asarray(arteries==1).nonzero()[0])
            neigh=graph.vertex_neighbours(a)
            print(neigh)
            for r, n in enumerate(neigh):
                print(n)
                if n in np.asarray(arteries==1).nonzero()[0]:
                    neigh=np.delete(neigh, r)
            print(neigh)
            if neigh.shape[0]!=0:
                if neigh.shape[0]==1:
                    arteries[neigh[0]]=1
                    art[j]=neigh[0]
                    print(0,neigh[0])

                else:
                    art_pos=graph.vertex_coordinates()[arteries.astype(bool)]
                    neigh_pos=graph.vertex_coordinates()[neigh]
                    vegf, o2=vegflevel(neuronpositionTest,art_pos, siga, sigb)
                    signal_val=[np.sum([gaussian(neigh_pos[n], vegf[k], neuronpositionTest[k], sigma) for k in range(neuronpositionTest.shape[0])]) for n in range(neigh.shape[0])]
                    arteries[neigh[np.argmax(signal_val)]]=1
                    art[j]=neigh[np.argmax(signal_val)]
                    print(1,neigh[np.argmax(signal_val)])

        for j, a in enumerate(vein):
            neigh=graph.vertex_neighbours(a)
            for r, n in enumerate(neigh):
                if n in np.asarray(veins==1).nonzero()[0]:
                    neigh=np.delete(neigh, r)
            if neigh.shape[0]!=0:
                if neigh.shape[0]==1:
                    veins[neigh[0]]=1
                    vein[j]=neigh[0]
                    print(2,neigh[0])

                else:
                    art_pos=graph.vertex_coordinates()[veins.astype(bool)]
                    neigh_pos=graph.vertex_coordinates()[neigh]
                    vegf, o2=vegflevel(neuronpositionTest,art_pos, siga, sigb)
                    signal_val=[np.sum([gaussian(neigh_pos[n], vegf[k], neuronpositionTest[k], sigma) for k in range(neuronpositionTest.shape[0])]) for n in range(neigh.shape[0])]
                    veins[neigh[np.argmax(signal_val)]]=1
                    vein[j]=neigh[np.argmax(signal_val)]
                    print(3,neigh[np.argmax(signal_val)])

        i=i-1

    graph.add_vertex_property('arteries',arteries)
    graph.add_vertex_property('veins',veins)

    colorVal = np.zeros((graph.n_vertices, 4))
    red_blue_map = {1: [1.0, 0, 0, 1.0], 3: [0.0, 1.0, 0.0, 1.0], 2: [0.0, 0, 1.0, 1.0], 4: [0.0, 0.0, 0.0, 1.0]}
    arteries=graph.vertex_property('arteries')
    veins=graph.vertex_property('veins')
    for i in range(graph.n_vertices):
        # print(j)
        if arteries[i] == 1:
            colorVal[i] = red_blue_map[1]
        elif veins[i] == 1:
            colorVal[i] = red_blue_map[2]
        else:
            colorVal[i] = red_blue_map[3]


    p3d.plot_graph_mesh(graph, vertex_colors=colorVal,  n_tube_points=3)



    conn=graph.edge_connectivity()
    vertices=graph.vertex_coordinates()
    distanxes=np.array([np.linalg.norm(vertices[c[0]]-vertices[c[1]]) for c in conn])
    plt.figure()
    plt.hist(distanxes[distanxes<100], bins=10)


    graphclean=graph.copy()
    deg=graphclean.vertex_degrees()
    deg2=np.asarray(deg<=2).nonzero()[0]
    vertices=graphclean.vertex_coordinates()
    from sklearn.neighbors import NearestNeighbors
    i=0
    old_deg2=100000
    while deg2.shape[0]<old_deg2:
        old_deg2=deg2.shape[0]
        print(i, deg2.shape[0])
        A=kneighbors_graph(vertices[deg2], 1, n_jobs=10, mode='distance', include_self=False).toarray()
        for v in range(A.shape[0]):
            if deg[deg2[v]]<3:

                U=np.asarray(A[v, :]>0).nonzero()[0]
                for u in U:
                    if deg[deg2[u]]<3:
                        conn=graphclean.edge_connectivity()
                        sim_edge=np.logical_or(np.logical_and(conn[:, 0]==deg2[u],conn[:, 1]==deg2[v]), np.logical_and(conn[:, 1]==deg2[u],conn[:, 0]==deg2[v]))
                        if np.sum(sim_edge)==0:
                            print(v, '/', A.shape[0])
                            print(A[v, u])
                            conn=graphclean.edge_connectivity()
                            print(np.linalg.norm(vertices[deg2[v]]-vertices[deg2[u]]))
                            if A[v, u]<=100:
                                graphclean.add_edge([(deg2[v],deg2[u])])
                                deg=graphclean.vertex_degrees()
                                break
                        # else:
                        # print('edge already exists')
                deg=graphclean.vertex_degrees()
        deg2=np.asarray(deg<=2).nonzero()[0]


    plt.figure()
    deg=graphclean.vertex_degrees()
    plt.hist(deg, bins=np.arange(10))


    colorVal = np.zeros((graphclean.n_vertices, 4))
    red_blue_map = {1: [1.0, 0, 0, 1.0], 3: [0.0, 1.0, 0.0, 1.0], 2: [0.0, 0, 1.0, 1.0], 4: [0.0, 0.0, 0.0, 1.0]}
    arteries=graphclean.vertex_property('arteries')
    veins=graphclean.vertex_property('veins')
    for i in range(graphclean.n_vertices):
        # print(j)
        if arteries[i] == 1:
            colorVal[i] = red_blue_map[1]
        elif veins[i] == 1:
            colorVal[i] = red_blue_map[2]
        else:
            colorVal[i] = red_blue_map[3]


    p3d.plot_graph_mesh(graphclean, vertex_colors=colorVal,  n_tube_points=3)
    graphclean.save(os.path.join(path, 'graph_test_reco_cleaned_deg3.gt'))







elif type=='knn':

    from sklearn.neighbors import kneighbors_graph
    vor = Voronoi(neuronpositionTest)
    vertices=vor.vertices
    mins=[5,5,5]
    maxs=[100,70,100]
    isOverV=(vertices > mins).all(axis=1)
    isUndeV=(vertices < maxs).all(axis=1)
    vertices=vertices[isOverV*isUndeV]
    A = kneighbors_graph(vertices, 3, n_jobs=10, mode='connectivity', include_self=False)
    E_cap=np.array([(A>0).nonzero()[0],(A>0).nonzero()[1]]).T
    graph=fromEVtoGraph(vertices,E_cap, 0,0,0,0,radius=0.1, ratio=10)
    # A.toarray()
    # u=np.unique(A.toarray())
    # plt.hist(u, bins=100)
    # graph, distance=gtg.generate_knn(vertices, k=2, r=10, epsilon=0.00001, exact=False)
    # graph=ggtu.Graph(base=graph)
    conn=graph.edge_connectivity()
    # graph=fromEVtoGraph(vertices,conn, 0,0,0,0,radius=0.1, ratio=10)

    degrees=graph.vertex_degrees()
    deg4edges_list=np.logical_and(degrees[conn[:,0].tolist()]>=4,degrees[conn[:,1].tolist()]>4).nonzero()[0]
    deg4edges_list2=np.logical_and(degrees[conn[:,0].tolist()]>4,degrees[conn[:,1].tolist()]>=4).nonzero()[0]
    deg4edges_list3=np.logical_or(degrees[conn[:,0].tolist()]==1,degrees[conn[:,1].tolist()]==1).nonzero()[0]
    edge_filter = np.ones(graph.n_edges)
    edge_filter[deg4edges_list]=0
    edge_filter[deg4edges_list2]=0
    edge_filter[deg4edges_list3]=0
    graph = graph.sub_graph(edge_filter=edge_filter)
    # graph=graph.largest_component()
    plt.figure()
    deg=graph.vertex_degrees()
    plt.hist(deg, bins=np.arange(10))
    p3d.plot_graph_line(graph)

    # remove mutual edges
    conn=graph.edge_connectivity()


    p = Pool(20)
    start = time.time()
    e2rm = np.array(
        [p.map(mutualLoopDetection, [(i, conn) for  i in range(graph.n_edges)])])
    end = time.time()
    print(end - start)
    e2rm=e2rm[0][[e2rm[0][i].shape[0]>1 for i in range(e2rm[0].shape[0])]]
    e2rm_list=[e2rm[i].tolist() for i in range(e2rm.shape[0])]
    L=[]
    for i, e in enumerate(e2rm_list):
        print(i)
        if e not in L:
            print('not in list')
            L.append(e)

    edge_filter = np.ones(graph.n_edges)
    for e in L:
        # print(e)
        try:
            edge_filter[e[1:]] = 0
        except:
            print('no edge to remove')
            print(e)
    graphclean = graph.sub_graph(edge_filter=edge_filter)



    deg=graphclean.vertex_degrees()
    deg2=np.asarray(deg<=2).nonzero()[0]
    from sklearn.neighbors import NearestNeighbors
    i=0
    while deg2.shape[0]>10:
        print(i, deg2.shape[0])
        A=kneighbors_graph(vertices[deg2], 10, n_jobs=10, mode='connectivity', include_self=False).toarray()
        for v in range(A.shape[0]):
            if deg[deg2[v]]<3:

                U=np.asarray(A[v, :]==1).nonzero()[0]
                for u in U:
                    if deg[deg2[u]]<3:
                        conn=graphclean.edge_connectivity()
                        sim_edge=np.logical_or(np.logical_and(conn[:, 0]==deg2[u],conn[:, 1]==deg2[v]), np.logical_and(conn[:, 1]==deg2[u],conn[:, 0]==deg2[v]))
                        if np.sum(sim_edge)==0:
                            print(v, '/', A.shape[0])
                            graphclean.add_edge([(deg2[v],deg2[u])])
                            deg=graphclean.vertex_degrees()
                        # else:
                            # print('edge already exists')
                deg=graphclean.vertex_degrees()
        deg2=np.asarray(deg<=2).nonzero()[0]


    plt.figure()
    deg=graphclean.vertex_degrees()
    plt.hist(deg, bins=np.arange(10))
    p3d.plot_graph_line(graphclean)



    E_cap=graph.edge_connectivity()
    V_tot_f=np.concatenate((verticestot, neuronpositionTest), axis=0)
    Etot_f=np.concatenate((Etot, E_cap+np.max(Etot)+1), axis=0)

    np.save(os.path.join(path, 'Etot_f.npy'), Etot_f)
    np.save(os.path.join(path, 'V_tot_f.npy'),V_tot_f)
    # plotGraph(V_tot_f, Etot_f, mins=[0, 0, 0], maxs=[100, 100, 100])
    graph=fromEVtoGraph(V_tot_f, Etot_f, len(edgestree), len(edgestree2),VID0,VID02, radius=1, ratio=10)#ratio=30

    plt.figure()
    deg=graph.vertex_degrees()
    plt.hist(deg, bins=np.arange(10))
    # p3d.plot_graph_line(graph)


    conn=graphclean.edge_connectivity()
    degrees=graphclean.vertex_degrees()
    deg2vertex=degrees==2
    for v in deg2vertex:

    deg4edges_list=np.logical_and(degrees[conn[:,0].tolist()]==4,degrees[conn[:,1].tolist()]==4).nonzero()[0]
    edge_filter = np.ones(graphclean.n_edges)
    edge_filter[deg4edges_list]=0


graph=graphclean.copy()
colorVal = np.zeros((graph.n_vertices, 4))
red_blue_map = {1: [1.0, 0, 0, 1.0], 3: [0.0, 1.0, 0.0, 1.0], 2: [0.0, 0, 1.0, 1.0], 4: [0.0, 0.0, 0.0, 1.0]}
arteries=graph.vertex_property('arteries')
veins=graph.vertex_property('veins')
for i in range(graph.n_vertices):
    # print(j)
    if arteries[i] == 1:
        colorVal[i] = red_blue_map[1]
    elif veins[i] == 1:
        colorVal[i] = red_blue_map[2]
    else:
        colorVal[i] = red_blue_map[3]


p3d.plot_graph_mesh(graph, vertex_colors=colorVal,  n_tube_points=3)
graph.save(os.path.join(path, 'graph_test.gt'))


graph=ggt.load(os.path.join(path, 'graph_test.gt'))


vf=graph.vertex_degrees()!=0
graph=graph.sub_graph(vertex_filter=vf)
coordinates = graph.vertex_property('coordinates')
# pt2reconnect=np.logical_and(np.logical_and(graph.vertex_degrees()==1, np.logical_not(graph.vertex_property('startPoint'))),np.logical_or(graph.vertex_property('arteries'),graph.vertex_property('veins')))
pt2reconnect=np.logical_and(graph.vertex_degrees()==1, np.logical_not(graph.vertex_property('startPoint')))

pt2reconnectind=pt2reconnect.nonzero()[0]

artVein=np.logical_or(graph.vertex_property('arteries'),graph.vertex_property('veins'))#.nonzero()[0]
artVein[graph.vertex_property('startPoint').nonzero()[0]]=0

bp2reco=np.random.choice(artVein.nonzero()[0], 20)
pt2reconnectind=np.concatenate((pt2reconnectind, bp2reco))

capillaries=np.logical_not(np.logical_or(graph.vertex_property('arteries'),graph.vertex_property('veins')))

##reconnexcion
from multiprocessing import Pool
p = Pool(20)
import time
start = time.time()

minDistances = np.array(
    [p.map(get_distances, [(n, pt, coordinates[pt],coordinates,capillaries) for n, pt in enumerate(pt2reconnectind)])])

end = time.time()
print(end - start)

edges2add=minDistances[0]

if type=='knn':
    graph.add_edge(edges2add)
    graph.save(os.path.join(path, 'graph_test_reco_cleaned.gt'))


elif type=='voronoi':
    ## remove multiple connexion to a similkar vertex

    edges2add_bis=[]
    elem=[]
    for e in edges2add:
        if edges2add_bis==[]:
            edges2add_bis.append(e)
            elem.append(e[0])
            elem.append(e[1])
        else:
            if e[0] in elem:
                print('vertex already used')
            elif e[1] in elem:
                print('vertex already used')
            else:
                edges2add_bis.append(e)
                elem.append(e[0])
                elem.append(e[1])

    edges2add=edges2add_bis.copy()
    # graph.save(os.path.join(path, 'graph_test_reco.gt'))

    ##remove mutual loops
    conn=graph.edge_connectivity()


    p = Pool(20)


    start = time.time()

    e2rm = np.array(
        [p.map(mutualLoopDetection, [(i, conn) for  i in range(graph.n_edges)])])

    end = time.time()
    print(end - start)

    e2rm=e2rm[0][[e2rm[0][i].shape[0]>1 for i in range(e2rm[0].shape[0])]]
    e2rm_list=[e2rm[i].tolist() for i in range(e2rm.shape[0])]
    L=[]
    for i, e in enumerate(e2rm_list):
        print(i)
        if e not in L:
            print('not in list')
            L.append(e)

    edge_filter = np.ones(graph.n_edges)
    for e in L:
        # print(e)
        try:
            edge_filter[e[1:]] = 0
        except:
            print('no edge to remove')
            print(e)
    graphclean = graph.sub_graph(edge_filter=edge_filter)

    try:
        graphclean.add_edge(edges2add)
    except:
        edges2add=edges2add[[edges2add[i]!=None for i in range(edges2add.shape[0])]]
        edges2add=np.array(edges2add.tolist())
        # edges2add=edges2add.reshape((int(edges2add.shape[0]/2) ,2))
        graphclean.add_edge(edges2add)

    graphclean.save(os.path.join(path, 'graph_test_reco_cleaned.gt'))


    ## remove random edges to get deg 3 config

    graph_deg3=graphclean.copy()
    ratio=graph_deg3.n_vertices/graph_deg3.n_edges
    N=0
    while graph_deg3.n_vertices/graph_deg3.n_edges<=0.66:
        conn=graph_deg3.edge_connectivity()
        degrees=graph_deg3.vertex_degrees()
        deg4edges_list=np.logical_and(degrees[conn[:,0].tolist()]==4,degrees[conn[:,1].tolist()]==4).nonzero()[0]
        e2rm=np.random.choice(deg4edges_list, 1)[0]
        print(e2rm)
        # e2rm=deg4edges_list[e2rm][0]
        v1=conn[e2rm][0]
        v2 = conn[e2rm][1]
        print(degrees[v1],degrees[v2])
        edge_filter=np.ones(graph_deg3.n_edges)
        edge_filter[e2rm]=0
        graph_deg3 = graph_deg3.sub_graph(edge_filter=edge_filter)
        degrees = graph_deg3.vertex_degrees()
        print(degrees[v1], degrees[v2])
        ratio = graph_deg3.n_vertices / graph_deg3.n_edges
        print(graph_deg3, ratio)
        N=N+1
    print(N)

    print(graph_deg3)
    graph_deg3.save(os.path.join(path, 'graph_test_reco_cleaned_deg3.gt'))

    colorVal = np.zeros((graphclean.n_vertices, 4))
    red_blue_map = {1: [1.0, 0, 0, 1.0], 3: [0.0, 1.0, 0.0, 1.0], 2: [0.0, 0, 1.0, 1.0], 4: [0.0, 0.0, 0.0, 1.0]}
    arteries=graphclean.vertex_property('arteries')
    veins=graphclean.vertex_property('veins')
    for i in range(graphclean.n_vertices):
        # print(j)
        if arteries[i] == 1:
            colorVal[i] = red_blue_map[1]
        elif veins[i] == 1:
            colorVal[i] = red_blue_map[2]
        else:
            colorVal[i] = red_blue_map[3]


    p3d.plot_graph_mesh(graphclean, vertex_colors=colorVal,  n_tube_points=3)




    degrees=graph_deg3.vertex_degrees()
    plt.figure()
    plt.hist(degrees, bins=100)



base=graph_deg3.base
state_sbm = gti.minimize_blockmodel_dl(base)
modules = state_sbm.get_blocks().a
graph_deg3.add_vertex_property('blocks_sbm', modules)
# gss4.add_vertex_property('indices', indices)
Q, Qs = modularity_measure(modules, graph_deg3, 'blocks_sbm')
#
# b=state_sbm.b
#
# bn= b.get_array()
# new_cmap, randRGBcolors = rand_cmap(np.unique(bn).shape[0], type='bright', first_color_black=False,
#                                     last_color_black=False, verbose=True)
# n = graphclean.n_vertices
# colorval = np.zeros((n, 3));
# for i in range(bn.size):
#     colorval[i] = randRGBcolors[int(bn[i])]
# colorval=np.array(colorval)
#
# graph_deg3.add_vertex_property('vertex_fill_color',colorval)#colorval
#
# p3d.plot_graph_mesh(graph_deg3, vertex_colors=graph_deg3.vertex_property('vertex_fill_color'),  n_tube_points=3)
#



# assign edge geormetry
import scipy.spatial.transform.rotation as R
from scipy.spatial.transform.rotation import Rotation
from scipy.linalg import norm
refg=ggt.load('/data_SSD_2to/191122Otof/5R/data_graph_correctedIsocortex.gt')
graph_deg3_test=ggt.load(os.path.join(path, 'graph_test_reco_cleaned_deg3.gt'))
# p3d.plot_graph_mesh(graph_deg3_test)

def assign_edge_greom_from_graph(graph_deg3_test, refg):
    coordinates=graph_deg3_test.vertex_property('coordinates')
    conn=graph_deg3_test.edge_connectivity()
    artery=from_v_prop2_eprop(graph_deg3_test, 'arteries')
    veins=from_v_prop2_eprop(graph_deg3_test, 'veins')
    art_vein=np.logical_or(artery, veins)
    capillaries_filter=np.logical_and(np.logical_not(refg.edge_property('artery')), np.logical_not(refg.edge_property('vein')))

    refg=refg.sub_graph(edge_filter=capillaries_filter)

    ref_coordinates=refg.vertex_property('coordinates')
    ref_indices=refg._edge_geometry_indices_graph()
    ref_conn=refg.edge_connectivity()
    ref_geometry_coordinates=refg.edge_geometry_property('coordinates')
    ref_geometry_radii = refg.edge_geometry_property('radii')

    res_edge_geom=[]
    res_edge_ind = []
    res_edge_rad=[]
    N=0
    for i, e in enumerate(conn):
        # A=coordinates[e[1]]-coordinates[e[0]]
        j=0
        lenghth=0
        while lenghth<20:#40
            j = random.choice(range(refg.n_edges))
            lenghth=ref_indices[j][1]-ref_indices[j][0]
        print(lenghth)
        print(i, j, '/', graph_deg3_test.n_edges)
        # B=ref_coordinates[ref_indices[j][1]]-ref_coordinates[ref_indices[j][0]]
        P=np.array([coordinates[e[0]],coordinates[e[1]]]).transpose()
        pnorm=np.linalg.norm(P)
        Pn=P/pnorm
        Q = np.array([ref_coordinates[ref_conn[j][0]], ref_coordinates[ref_conn[j][1]]]).transpose()
        qnorm =np.linalg.norm(Q)
        Qn=Q/qnorm
        M = P.dot(np.linalg.pinv(Q))
        # M=M*pnorm/qnorm
        res_edge_ind.append([N, N+ref_indices[j][1]-ref_indices[j][0]])
        N=N+ref_indices[j][1]-ref_indices[j][0]
        # print(ref_indices[j][1]-ref_indices[j][0])
        e_geom=M.dot(ref_geometry_coordinates[ref_indices[j][0]:ref_indices[j][1]].transpose()).transpose()
        e_rad=ref_geometry_radii[ref_indices[j][0]:ref_indices[j][1]]
        # ratio=pnorm/qnorm
        # e_geom=ratio*e_geom
        # e_geom=e_geom-e_geom[0]
        # e_geom=e_geom+coordinates[e[0]]
        print(e_rad)
        print(e_geom.shape,e_geom.shape==(ref_indices[j][1]-ref_indices[j][0]))
        for k in range(e_geom.shape[0]):
            if k==0:
                res_edge_geom.append(e_geom[k])
            else:
                if np.linalg.norm(e_geom[k]-res_edge_geom[-1])<=30:
                    res_edge_geom.append(e_geom[k])
                else:
                    print('too long')
                    res_edge_geom.append(res_edge_geom[-1]+2*(e_geom[k]-res_edge_geom[-1])/np.linalg.norm(e_geom[k]-res_edge_geom[-1]))
            if art_vein[i]==0:
                res_edge_rad.append(e_rad[k])
            else:
                res_edge_rad.append(10)
        print(len(res_edge_geom))
    return(res_edge_geom, res_edge_ind,res_edge_rad)


def assign_edge_geom_from_interpolation(graph_deg3_test):
    coordinates=graph_deg3_test.vertex_property('coordinates')
    conn=graph_deg3_test.edge_connectivity()
    res_edge_geom=[]
    res_edge_ind = []
    res_edge_rad=[]
    N=0
    for i, e in enumerate(conn):
        # A=coordinates[e[1]]-coordinates[e[0]]
        j=0
        lenghth=0




res_edge_geom, res_edge_ind,res_edge_rad=assign_edge_greom_from_graph(graph_deg3_test, refg)

#
# def from_v_prop2_eprop(graph, vprop):
#     # vprop = graph.vertex_property(property)
#     # e_prop=np.zeros(graph.n_edges)
#     connectivity = graph.edge_connectivity()
#     e_prop = np.logical_and(vprop[connectivity[:, 0]], vprop[connectivity[:, 1]])
#     return e_prop
#
#
#
#
#
#
# def from_v_prop2_eprop(graph, property):
#     vprop = graph.vertex_property(property)
#     # e_prop=np.zeros(graph.n_edges)
#     connectivity = graph.edge_connectivity()
#     e_prop = np.logical_and(vprop[connectivity[:, 0]], vprop[connectivity[:, 1]])
#     return e_prop


# ##geometry_radii=2*np.ones(len(res_edge_geom))
# art_edges=from_v_prop2_eprop(graph_deg3_test, 'arteries')
# vein_edges=from_v_prop2_eprop(graph_deg3_test, 'veins')
#
# art_vein_indices=res_edge_ind[np.max(np.asarray(np.logical_or(art_edges,vein_edges)).nonzero()[0])]
res_edge_rad=np.array(res_edge_rad)
# res_edge_rad[:np.max(art_vein_indices)]=10

np.save(os.path.join(path, 'graph_test_edge_geom_v11.npy'),res_edge_geom)
np.save(os.path.join(path, 'graph_test_edge_ind_v11.npy'),res_edge_ind)
np.save(os.path.join(path, 'graph_test_edge_rad_v11.npy'),res_edge_rad)

graph_deg3_test.set_edge_geometry(coordinates=res_edge_geom, indices=res_edge_ind, radii=res_edge_rad)
# graph_deg3.add_edge_property('geometry_indices',res_edge_ind)
# graph_deg3.set_edge_geometry_property('geometry_coordinates',res_edge_geom)

# ref_rad=refg.edge_property('radii')
# rad=ref_rad[np.random.choice(range(refg.n_edges), graph_deg3_test.n_edges)]
# graph_deg3_test.add_edge_property('radii', rad)


graph_deg3_test.save(os.path.join(path, 'graph_test_reco_cleaned_deg3_goemetry_v11.gt'))

p3d.plot_graph_mesh(graph_deg3_test)
######
# graph_deg3_test=ggt.load(os.path.join(path, 'graph_test_reco_cleaned_deg3_v11.gt'))
# res_edge_geom=np.load(os.path.join(path, 'graph_test_edge_geom_v11.npy'))
# res_edge_ind=np.load(os.path.join(path, 'graph_test_edge_ind_v11.npy'))
#
# graph_deg3_test.set_edge_geometry(coordinates=res_edge_geom, indices=res_edge_ind)
# # graph_deg3.add_edge_property('geometry_indices',res_edge_ind)
# # graph_deg3.set_edge_geometry_property('geometry_coordinates',res_edge_geom)
# graph_deg3_test.save(os.path.join(path, 'graph_test_reco_cleaned_deg3_goemetry_v11.gt'))
#
#
#
#
# graph_deg3_test=ggtn.load(os.path.join(path, 'graph_test_reco_cleaned_deg3_goemetry_v11.gt'))

# graph_deg3_test.set_edge_radii_from_vertex_radii()
# graph_deg3_test.edge_geometry_from_edge_property('radii')

# g2p=graph_deg3_test.sub_graph(vertex_filter=np.logical_not(np.logical_or(graph_deg3_test.vertex_property('arteries'),graph_deg3_test.vertex_property('veins'))))


coord=graph_deg3_test.vertex_property('coordinates')
conn=graph_deg3_test.edge_connectivity()

length=np.array([np.linalg.norm(coord[conn[i][1]]-coord[conn[i][0]]) for i in range(graph_deg3_test.n_edges)])
ef=length<=300

mins=np.array([15,15, 15])*10
maxs=np.array([100, 70, 100])*10

g2p = graph_deg3_test.sub_graph(edge_filter=ef)
# vf=np.logical_and(graph_deg3_test.vertex_property('coordinates')[:,0]>15*30,graph_deg3_test.vertex_property('coordinates')[:,0]<70*30)
# vf=extractSubGraph(g2p.vertex_property('coordinates'), mins, maxs)
# g2p = g2p.sub_graph(vertex_filter=vf)
g2p=g2p.largest_component()


colorVal = np.zeros((g2p.n_vertices, 4))
red_blue_map = {1: [1.0, 0, 0, 1.0], 3: [0.0, 1.0, 0.0, 1.0], 2: [0.0, 0, 1.0, 1.0], 4: [0.0, 0.0, 0.0, 1.0]}
arteries=g2p.vertex_property('arteries')
veins=g2p.vertex_property('veins')
for i in range(g2p.n_vertices):
    # print(j)
    if arteries[i] == 1:
        colorVal[i] = red_blue_map[1]
    elif veins[i] == 1:
        colorVal[i] = red_blue_map[2]
    else:
        colorVal[i] = red_blue_map[3]

p = p3d.plot_graph_mesh(g2p,vertex_colors=colorVal, n_tube_points=3)# vertex_colors=graph_deg3_test.vertex_property('vertex_fill_color'),



g2p.save(os.path.join(path, 'graph_test_reco_cleaned_deg3_goemetry_g2p.gt'))




stl_file=p.mesh_data.save()
import pickle
pickle.dump(stl_file, open(os.path.join(path, 'graph_test_reco_cleaned_deg3_goemetry_mesh.stl'), 'wb'))




base=g2p.base
state_sbm = gti.minimize_blockmodel_dl(base)
modules = state_sbm.get_blocks().a
g2p.add_vertex_property('blocks_sbm', modules)
# gss4.add_vertex_property('indices', indices)
Q, Qs = modularity_measure(modules, g2p, 'blocks_sbm')

b=state_sbm.b

bn= b.get_array()
new_cmap, randRGBcolors = rand_cmap(np.unique(bn).shape[0], type='bright', first_color_black=False,
                                    last_color_black=False, verbose=True)
n = g2p.n_vertices
colorval = np.zeros((n, 3));
for i in range(bn.size):
    colorval[i] = randRGBcolors[int(bn[i])]
colorval=np.array(colorval)

g2p.add_vertex_property('vertex_fill_color',colorval)#colorval

p3d.plot_graph_mesh(g2p, vertex_colors=g2p.vertex_property('vertex_fill_color'),  n_tube_points=3)











np.logical_not(np.logical_or(graph_deg3_test.vertex_property('arteries'),graph_deg3_test.vertex_property('veins')))





plt.figure()
plt.hist(g2p.vertex_degrees(), bins=10)

plt.figure()
plt.hist(g2p.vertex_degrees(), bins=10)
# indices = graph_deg3_test._edge_geometry_indices_graph();
#
# indices_new = np.diff(indices, axis=1)[:, 0];
# indices_new = np.cumsum(indices_new);
# indices_new = np.array([np.hstack([0, indices_new[:-1]]), indices_new]).T;
# graph_deg3_test._set_edge_geometry_indices_graph(indices_new);
#
# # reduce arrays
# n = indices_new[-1, -1];
# for prop_name in graph_deg3_test.edge_geometry_properties:
#     print(prop_name)
#     prop = graph_deg3_test.graph_property(prop_name);
#     shape_new = (n,) + prop.shape[1:];
#     prop_new = np.zeros(shape_new, prop.dtype);
#     for i, j in zip(indices, indices_new):
#         si, ei = i;
#         print(i, si, ei)
#         sj, ej = j;
#         print(j, sj, ej)
#         prop_new[sj:ej] = prop[si:ei];
#     graph_deg3_test.set_graph_property(prop_name, prop_new)
#

graph=ggtn.load(os.path.join(path, 'graph_test_reco_cleaned_deg3_goemetry_g2p.gt'))
# graph_deg3_test=graph_deg3_test.largest_component()
# graph=graph_deg3_test.largest_component().copy()


graph=graph_deg3_test.copy()
connectivity=graph.edge_connectivity()
arteries=graph.vertex_property('arteries')
veins=graph.vertex_property('veins')

av=np.logical_or(np.logical_and(arteries[connectivity[:,0]],veins[connectivity[:,1]]),np.logical_and(veins[connectivity[:,0]],arteries[connectivity[:,1]]))
av=av.nonzero()[0]

for v in connectivity[av].flatten():
    if veins[v]==1:
        arteries[v]=1
        veins[v]=0



def from_v_prop2_eprop(graph, vprop):
    # vprop = graph.vertex_property(property)
    # e_prop=np.zeros(graph.n_edges)
    connectivity = graph.edge_connectivity()
    e_prop = np.logical_and(vprop[connectivity[:, 0]], vprop[connectivity[:, 1]])
    return e_prop


artery=from_v_prop2_eprop(graph, arteries)
graph.add_edge_property('artery', artery)

vein=from_v_prop2_eprop(graph, veins)
graph.add_edge_property('vein', vein)

artery=from_e_prop2_vprop(graph, 'arteries')
graph.add_vertex_property('arteries', artery)

vein=from_e_prop2_vprop(graph, 'veins')
graph.add_vertex_property('veins', vein)


graph=g2p.copy()
ref_rad=refg.edge_property('radii')
rad=ref_rad[np.random.choice(range(refg.n_edges), graph.n_edges)]
graph.add_edge_property('radii', rad)
radius=graph.edge_property('radii')

plt.figure()
plt.hist(radius)
#
# veins=np.zeros(graph.n_edges)
# arteries=np.zeros(graph.n_edges)
# rart=np.random.choice(np.arange(graph.n_edges), 1)
# rvein=np.random.choice(np.arange(graph.n_edges), 1)
# arteries[rart]=1
# veins[rvein]=1
#
# graph.add_edge_property('vein', veins.astype(bool))
# graph.add_edge_property('artery', arteries.astype(bool))


colorVal = np.zeros((graph.n_edges, 4))
red_blue_map = {1: [1.0, 0, 0, 1.0], 3: [0.0, 1.0, 0.0, 1.0], 2: [0.0, 0, 1.0, 1.0], 4: [0.0, 0.0, 0.0, 1.0]}
arteries=graph.edge_property('arteries')
veins=graph.edge_property('veins')
for i in range(graph.n_edges):
    # print(j)
    if arteries[i] == 1:
        colorVal[i] = red_blue_map[1]
    elif veins[i] == 1:
        colorVal[i] = red_blue_map[2]
    else:
        colorVal[i] = red_blue_map[3]
#
p3d.plot_graph_mesh(graph,edge_colors=colorVal, n_tube_points=3)
# colorVal=np.zeros(graph.n_vertices)
# arteries=graph.vertex_property('arteries')
# veins=graph.vertex_property('veins')
# colorVal[np.asarray(arteries==1).nonzero()[0]]=1
# colorVal[np.asarray(veins==1).nonzero()[0]]=2
# graph.add_vertex_property('temp', colorVal)
import graph_tool.draw as gtd
pos = gtd.sfdp_layout(graph.base)
gtd.graph_draw(graph.base, pos=pos, vertex_fill_color=graph.base.vertex_properties['temp'],output="/data_2to/graphtest.pdf")

import pickle
try:
    with open(path+'/test_v11/sampledicttest_v11.pkl', 'rb') as fp:
        dictio = pickle.load(fp)
        f=dictio['flow']
        v=dictio['v']
        p=dictio['pressure']

except:
    print('no dictionary found')
    f, v=computeFlowFranca(path, graph, '')
    with open(path+'/sampledict.pkl', 'rb') as fp:
        dictio = pickle.load(fp)
        f=dictio['flow']
        v=dictio['v']
        p=dictio['pressure']


graph.add_edge_property('flow', f)
graph.add_edge_property('veloc', v)
ps=7.3
# f=np.array(f)
# e = 1 - np.exp(-(ps / abs(f[0])))#abs(f)
# graph.add_edge_property('extracted_frac', e)
import seaborn as sns
plt.figure()
sns.set_style('white')
sns.despine()
plt.hist(p, bins=100)
plt.title('pressure')
plt.yscale('log')

plt.figure()
sns.set_style('white')
sns.despine()
plt.hist(f, bins=100)
plt.title('flow')
plt.yscale('log')

plt.figure()
sns.set_style('white')
sns.despine()
plt.hist(v, bins=100)
plt.title('v')
plt.yscale('log')

e = 1 - np.exp(-(7.3/ abs(np.asarray(f)[0])))#abs(f)
plt.figure()
sns.set_style('white')
sns.despine()
histp, binsp=np.histogram(e, bins=100, normed=True)
plt.hist(e, bins=100)
# plt.bar(binsp[:-1],histp)
plt.title('extracted fraction')
plt.yscale('log')


VecStart_x=VecStart_x[mask]
VecEnd_x=VecEnd_x[mask]
VecStart_y=VecStart_y[mask]
VecEnd_y=VecEnd_y[mask]
VecStart_z=VecStart_z[mask]
VecEnd_z=VecEnd_z[mask]

for i in range(VecStart_x.shape[0]):
    # if (abs(VecStart_x[i])<100 and abs(VecEnd_x[i]<100) and abs(VecStart_y[i]<100)  and abs(VecEnd_y[i]<100)  and abs(VecStart_z[i]<100)  and VecEnd_z[i]<100 ):
    plt.plot([VecStart_x[i] ,VecEnd_x[i]],[VecStart_y[i],VecEnd_y[i]],[VecStart_z[i],VecEnd_z[i]], c='g')

plt.xlim(0,100)
plt.ylim(0,100)
ax.set_zlim(0,100)



controls=['2R','3R','5R', '8R']
mutants=['1R','7R', '6R', '4R']
work_dir='/data_SSD_2to/191122Otof'
states=[controls]
for state in states:
    for a, control in enumerate(state):
        with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
            sampledict = pickle.load(fp)
            f = sampledict['flow']
            v = sampledict['v']
            p = sampledict['pressure']
            e = 1 - np.exp(-(7.3 / abs(np.asarray(f)[0])))  # abs(f)





            if a==0:
                print(a, control)
                histp, binsp = np.histogram(p, bins=30, normed=True)
                histf, binsf = np.histogram(v, bins=np.arange(0,50,5), normed=True)
                histv, binsv = np.histogram(f, bins=np.arange(0,7, 7/10), normed=True)
                histe, binse = np.histogram(e, bins=30, normed=True)

                histp = np.expand_dims(histp, 1)
                histv = np.expand_dims(histv, 1)
                histf = np.expand_dims(histf, 1)
                histe = np.expand_dims(histe, 1)

                pressures=histp
                velocities=histv
                flows=histf
                extracted_frac=histe
            else:
                print(a, control)
                histp, binsp = np.histogram(p, bins=binsp, normed=True)
                histv, binsv = np.histogram(v, bins=binsv, normed=True)
                histf, binsf = np.histogram(f, bins=binsf, normed=True)
                histe, binse = np.histogram(e, bins=binse, normed=True)

                histp = np.expand_dims(histp, 1)
                histv = np.expand_dims(histv, 1)
                histf = np.expand_dims(histf, 1)
                histe = np.expand_dims(histe, 1)

                pressures=np.concatenate((pressures, histp), axis=1)
                velocities = np.concatenate((velocities, histv), axis=1)
                flows = np.concatenate((flows, histf), axis=1)
                extracted_frac = np.concatenate((extracted_frac, histe), axis=1)

import pandas as pd
from sklearn.preprocessing import normalize
from scipy import stats
# with open('/data_SSD_2to/SimulationVasculature3D/V17/sampledict.pkl', 'rb') as fp:
with open('/data_SSD_2to/SimulationVasculature3D/090821/sampledict.pkl', 'rb') as fp:
    dictio = pickle.load(fp)
    f = dictio['flow']
    v = dictio['v']
    p = dictio['pressure']
    e = 1 - np.exp(-(7.3 / abs(np.asarray(f)[0])))  # abs(f)

plt.figure()
sns.set_style('white')
sns.despine()
# P = pd.DataFrame(normalize(pressures, axis=0).transpose()).melt()
# sns.lineplot(x="variable", y="value", err_style='bars', data=P, color='cadetblue')
# histp, binsp = np.histogram(p, bins=binsp, normed=True)
# histp = np.expand_dims(histp, 1)
# P = pd.DataFrame(histp).melt()
density = stats.kde.gaussian_kde(p)
xs=binsp[1:]#np.arange(0, 400, 40)
# sns.lineplot(data=normalize(np.expand_dims(density(xs), 1),axis=0)[:,0], color='red')
sns.kdeplot(pd.DataFrame(pres))
sns.kdeplot(pd.DataFrame(p))
# plt.hist(p, bins=100)
plt.title('pressure')
# plt.yscale('log')
sns.despine()


plt.figure()
sns.set_style('white')
sns.despine()
P = pd.DataFrame(normalize(velocities, axis=0).transpose()).melt()
sns.lineplot(x="variable", y="value", err_style='bars', data=P, color='cadetblue')
# histv, binsv = np.histogram(v, bins=binsp, normed=True)
# histp = np.expand_dims(histp, 1)
# P = pd.DataFrame(histp).melt()
density = stats.gaussian_kde(v,bw_method=5)
xs=binsv[:-1]#np.arange(0, 400, 40)
sns.lineplot(data=normalize(np.expand_dims(density(xs), 1),axis=0)[:,0], color='red')
# plt.hist(p, bins=100)
plt.title('velocities')
# plt.yscale('log')
sns.despine()

plt.figure()
sns.set_style('white')
sns.despine()
P = pd.DataFrame(normalize(flows, axis=0).transpose()).melt()
sns.lineplot(x="variable", y="value", err_style='bars', data=P, color='cadetblue')
# histv, binsv = np.histogram(v, bins=binsp, normed=True)
# histp = np.expand_dims(histp, 1)
# P = pd.DataFrame(histp).melt()
density = stats.kde.gaussian_kde(f,bw_method=0.2)
xs=binsf[:-1]#np.arange(0, 400, 40)
sns.lineplot(data=normalize(np.expand_dims(density(xs), 1),axis=0)[:,0], color='red')
# plt.hist(p, bins=100)
plt.title('flows')
# plt.yscale('log')
sns.despine()


plt.figure()
sns.set_style('white')
sns.despine()
P = pd.DataFrame(normalize(extracted_frac, axis=0).transpose()).melt()
sns.lineplot(x="variable", y="value", err_style='bars', data=P, color='cadetblue')
# histv, binsv = np.histogram(v, bins=binsp, normed=True)
# histp = np.expand_dims(histp, 1)
# P = pd.DataFrame(histp).melt()
density = stats.kde.gaussian_kde(e)
xs=binse[1:]#np.arange(0, 400, 40)
sns.lineplot(data=normalize(np.expand_dims(density(xs), 1),axis=0)[:,0], color='red')
# plt.hist(p, bins=100)
plt.title('extracted_frac')
# plt.yscale('log')
sns.despine()



## orientation
graph=ggtn.load(os.path.join(path, 'graph_test_reco_cleaned_deg3_goemetry_g2p.gt'))
limit_angle=40
pi=math.pi
import math

coordinates=graph.vertex_coordinates()
top2bot=np.array([0,1,0])
x = graph.vertex_coordinates()[:, 0]
y = graph.vertex_coordinates()[:, 1]
z = graph.vertex_coordinates()[:, 2]

connectivity = graph.edge_connectivity()
# lengths = graph.edge_property('length')
edge_vect = np.array(
    [x[connectivity[:, 1]] - x[connectivity[:, 0]], y[connectivity[:, 1]] - y[connectivity[:, 0]],
     z[connectivity[:, 1]] - z[connectivity[:, 0]]]).T

normed_edge_vect=np.array([edge_vect[i] / np.linalg.norm(edge_vect[i]) for i in range(edge_vect.shape[0])])
# N=np.linalg.norm(edge_vect[i])
# print(N)
# normed_edge_vect=normed_edge_vect[~np.isnan(normed_edge_vect)]
rad=np.array([np.dot(top2bot.transpose(), normed_edge_vect[i]) for i in range(edge_vect.shape[0])])
plan=np.sqrt(1-rad**2)

r=abs(rad)
p=abs(plan)

r = r[~np.isnan(r)]
p = p[~np.isnan(r)]

edges_centers = np.array(
    [(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in range(connectivity.shape[0])])
dist = edges_centers[:,1][~np.isnan(r)]

# radiality = (r / (r + p)) > 0.5
# planarity = (p / (r + p)) > 0.6
# neutral = np.logical_not(np.logical_or(radiality, planarity))
angle = np.array([math.acos(r[i]) for i in range(r.shape[0])]) * 180 / pi

radiality = angle <  limit_angle#40
planarity = angle >  90-limit_angle#60
neutral = np.logical_not(np.logical_or(radiality, planarity))


plt.figure()
sns.distplot(dist[radiality], bins=10)
sns.distplot(dist[planarity], bins=10)
sns.distplot(dist[neutral], bins=10)
plt.legend(['in-flow', 'cross-flow', 'neutral'])

plt.figure()
radhist, radbins=np.histogram(dist[radiality], bins=10)
planhist, planbins=np.histogram(dist[planarity], bins=radbins)
# neuthist, neutbins=np.histogram(dist[neutral], bins=radbins)

sns.lineplot(radbins[:-1], radhist/(radhist+planhist+neuthist), color='cadetblue')
sns.lineplot(radbins[:-1], planhist/(radhist+planhist+neuthist),color='indianred')
sns.lineplot(radbins[:-1], neuthist/(radhist+planhist+neuthist),color='forestgreen')
plt.twinx()
sns.distplot(y, hist=False, kde=True, bins=10)

plt.legend(['in-flow', 'cross-flow', 'neutral'])

plt.figure()
sns.distplot(y, bins=10)
#
# Nb=10
#
# sigma=20
# a=1
# cap_tipscells=generateCapillariesTC(Nb, vasccells,size=[0,100], weightype='gaussian',a=a, sigma=sigma)
# cap_vasccells=cap_tipscells
# cap_tipscells_age=np.zeros(len(cap_tipscells)).astype(int)
# cap_tipscells_reco=[]
#
# cap_tipscells2=generateCapillariesTC(Nb, vasccells2,size=[0,100], weightype='gaussian',a=a, sigma=sigma)
# cap_vasccells2=cap_tipscells2
# cap_tipscells_age2=np.zeros(len(cap_tipscells2)).astype(int)
# cap_tipscells_reco2=[]
# graphUpdate(graph, vegf,cap_tipscells, cap_vasccells,cap_tipscells2, cap_vasccells2, col='g', col2='g')
#
# a=2
# sigma=10#10
# # a=10
# a_rep=1
# sigma_rep=3
# lr=1
#
# N=5
# compute_vein=False
# while N>0:
#     print(N)
#     cap_tipscells,cap_vasccells, vegf, O2, cap_tipscells_reco,cap_tipscells_age= computevegfgradient(neuronposition, cap_vasccells,
#                                                                                         cap_tipscells, cap_tipscells_reco, siga,
#                                                                                         sigb, lr, a_rep, sigma_rep,
#                                                                                         cap_tipscells_age, sizemax=[0, 100])
#     cap_tipscells2, cap_vasccells2, vegf, O2, cap_tipscells_reco2,cap_tipscells_age2 = computevegfgradient(neuronposition, cap_vasccells2,
#                                                                                      cap_tipscells2,
#                                                                                      cap_tipscells_reco2, siga, sigb, lr,
#                                                                                      a_rep,
#                                                                                      sigma_rep, cap_tipscells_age2,
#                                                                                      sizemax=[0, 100])
#     N=N-1
#     # print(tipscells)
#     graphUpdate(graph, vegf,cap_tipscells, cap_vasccells,cap_tipscells2, cap_vasccells2, col='g', col2='g')
#     # ani = animation.FuncAnimation(fig, graphUpdate, interval=1000)
#     # plt.show()



# plt.figure()
# Z=np.arange(0, 100, 0.1)
# plt.plot(Z, reverted_sigmoid(Z, siga, sigb))
#
# var = -1.5 # change this to -1.5 to get f2(x)
# x = np.arange(-8, 8, 0.1)
# y = np.exp(var*x) / (1+ np.exp(var*(x-3)))
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


# ani = animation.FuncAnimation(fig, graphUpdate, interval=1000)
# plt.show()


X = np.linspace(0,2,1000)
Y = X**2 + np.random.random(X.shape)

plt.ion()
graph = plt.plot(X,Y)[0]

    # Y = X**2 + np.random.random(X.shape)
