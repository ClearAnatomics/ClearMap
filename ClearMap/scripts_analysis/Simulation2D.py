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

def gaussian(x, a, x0, sigma):
    return a * (1. / (sigma*math.sqrt(2 * math.pi))) * np.exp(-(1./(2*(sigma**2))) * np.power((x - x0), 2.))

def Xgaussianderivative2D(x,y, a, x0, y0, sigma):
    # print(x, x0, a, sigma)
    res= - (x - x0)*a * (1. / 2 * math.pi*sigma**4) * np.exp(-(1./(2*(sigma**2))) * (np.power((x - x0), 2.)+np.power((y - y0), 2.)))
    # print(- x*a * (1. / 2 * math.pi*sigma**4) )
    # print(np.exp(-(1./(2*(sigma**2))) * (np.power((x - x0), 2.)+np.power((y - y0), 2.))))
    return res

def Ygaussianderivative2D(x,y, a, x0, y0, sigma):
    # print(y, y0, a, sigma)
    # print(- y * a * (1. / 2 * math.pi * sigma ** 4))
    # print(np.exp(-(1. / (2 * (sigma ** 2))) * (np.power((x - x0), 2.) + np.power((y - y0), 2.))))
    res=- (y - y0)*a * (1. / 2 * math.pi*sigma**4) * np.exp(-(1./(2*(sigma**2))) * (np.power((x - x0), 2.)+np.power((y - y0), 2.)))
    return res


def sigmoid(Z,a,b):
    return [a/(b + np.exp(-z)) for z in Z]


def reverted_sigmoid(Z,a,b):
    return np.exp(a*(Z-b)) / (1 + np.exp(a*(Z-b)))


def generateNeurons(Nb=1000, size=[0,100], weightype=gaussian, a=1, sigma=10):
    xweight = np.ones(size[1]) / size[1]
    if weightype=='random':
        yweight = np.ones(size[1]) / size[1]
        yweight = np.nan_to_num(yweight)
    elif weightype=='gaussian':
        x = np.linspace(0, size[1], size[1])
        yweight = gaussian(x, a, size[1]*40/100, sigma )
        yweight = np.nan_to_num(yweight)

    yposition=np.random.choice(range(size[1]), int(np.round(Nb)), p=yweight / np.sum(yweight), replace=True)[:, np.newaxis]
    xposition=np.random.choice(range(size[1]), int(np.round(Nb)), p=xweight / np.sum(xweight), replace=True)[:, np.newaxis]

    neuronposition=np.concatenate((xposition,yposition), axis=1)
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
    O2 = [1. / (1+(math.pow(math.sqrt( ((pos[0]-vascposition[i][0])**2)+((pos[1]-vascposition[i][1])**2)), 2))) for i in range(len(vascposition))]
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
        [Xgaussianderivative2D(t[0], t[1], a, tipscells[i][0], tipscells[i][1], sigma) for i in
         range(len(tipscells))]) - Xgaussianderivative2D(t[0], t[1], a, t[0], t[1], sigma)
    Yvegftc = np.sum(
        [Ygaussianderivative2D(t[0], t[1], a, tipscells[i][0], tipscells[i][1], sigma) for i in
         range(len(tipscells))]) - Ygaussianderivative2D(t[0], t[1], a, t[0], t[1], sigma)
    norm = LA.norm(np.array([Xvegftc, Yvegftc]))
    print(Xvegftc,Yvegftc,norm)
    if norm==0:
        return np.array([0.0, 0.0])
    else:
        repulsioncontib = np.array([Xvegftc / norm, Yvegftc / norm])
    return repulsioncontib


def computevegfgradient(neuronposition,vasccells,tipscells, tipscells_reco,siga, sigb, lr, a_rep, sigma_rep,tipscells_age,sizemax=100):
    vegf, O2=vegflevel(neuronposition,vasccells, siga, sigb)
    tipscells_new=[]
    tipscells_new_age=[]
    for n, tc in enumerate(tipscells):
        Xvegftc=np.sum([Xgaussianderivative2D(tc[0], tc[1], vegf[i], neuronposition[i, 0], neuronposition[i, 1], sigma) for i in range(neuronposition.shape[0])])
        Yvegftc=np.sum([Ygaussianderivative2D(tc[0], tc[1], vegf[i], neuronposition[i, 0], neuronposition[i, 1], sigma) for i in range(neuronposition.shape[0])])
        norm=LA.norm(np.array([Xvegftc,Yvegftc]))

        # preprocessing.normalize(np.array([Xvegftc,Yvegftc]).reshape(1, -1), norm='l2').reshape((2))
        print(norm)
        if norm<=400000:
            vasccells.append(tc)

        if norm<=500000:
            tipscells_reco.append(tc)
        else:
            vasccells.append(tc)
            repulsion=computeRepulsion(tc, vasccells, a_rep, sigma_rep)
            # Xvegftc=Xvegftc/repulsion[0]
            # Yvegftc = Yvegftc / repulsion[1]
            # norm = LA.norm(np.array([Xvegftc, Yvegftc]))
            vector = np.array([Xvegftc / norm, Yvegftc / norm])
            new_tc=(tc[0]+(vegf_coeff*vector[0]), tc[1]+(vegf_coeff*vector[1]))
            print(new_tc)
            if compute_vein==False:
                vect_f=((vegf_coeff*vector[0]) - (rep_coeff*repulsion[0]),
                        (vegf_coeff*vector[1])-(rep_coeff*repulsion[1]))
                norm = LA.norm(np.array([vect_f[0], vect_f[1]]))
                vect_f = np.array([vect_f[0] / norm, vect_f[1] / norm])
                new_tc = (lr * (tc[0] +vect_f[0]), lr *  (tc[1] +vect_f[1]))
                # new_tc=(lr*(tc[0]+(vegf_coeff*vector[0]) - (rep_coeff*repulsion[0])), lr*(tc[1]+(vegf_coeff*vector[1])-(rep_coeff*repulsion[1])))
            else:
                Xvegftc = np.sum(
                    [Xgaussianderivative2D(tc[0], tc[1], 1, veins[i][0], veins[i][1], 20) for i in
                     range(len(veins))])
                Yvegftc = np.sum(
                    [Ygaussianderivative2D(tc[0], tc[1], 1, veins[i][0], veins[i][1], 20) for i in
                     range(len(veins))])
                norm = LA.norm(np.array([Xvegftc, Yvegftc]))
                vectorvein = np.array([Xvegftc / norm, Yvegftc / norm])

                vect_f = ((vegf_coeff * vector[0]) - (rep_coeff*repulsion[0])+ (vein_coeff * vectorvein[0]),
                           (vegf_coeff * vector[1]) - (rep_coeff * repulsion[1]) + (vein_coeff * vectorvein[1]))
                norm = LA.norm(np.array([vect_f[0], vect_f[1]]))
                vect_f = np.array([vect_f[0] / norm, vect_f[1] / norm])
                new_tc = (lr * (tc[0] +vect_f[0]), lr *  (tc[1] +vect_f[1]))

                # new_tc = (lr*(tc[0] + (vegf_coeff * vector[0]) - (rep_coeff*repulsion[0]))+ (vein_coeff * vectorvein[0]),
                #           lr * (tc[1] + (vegf_coeff * vector[1]) - (rep_coeff * repulsion[1])) + (vein_coeff * vectorvein[1]))
            if compute_vein == False:
                if ((new_tc[0]<0) or (new_tc[1]<0)):
                    vasccells.append(tc)
                elif ((new_tc[0] > sizemax[1]) or (new_tc[1] > sizemax[1])):
                    vasccells.append(tc)
                else:
                    tipscells_new.append(new_tc)
                    tipscells_new_age.append(tipscells_age[n]+1)
            else :
                if new_tc in veins:
                    vasccells.append(tc)
                else:
                    tipscells_new.append(new_tc)
                    tipscells_new_age.append(tipscells_age[n] + 1)

    # tipscells_age=np.array(tipscells_new_age)
    print(tipscells_new_age, tipscells_new)
    return tipscells_new, vasccells, vegf, O2,tipscells_reco,tipscells_new_age


def computerecogradient(vasccells,tipscells_reco,tipscells_age, sizemax=100):
    tipscells_new = []
    tipscells_new_age=[]
    for n,tc in enumerate(tipscells_reco):
        Xvegftc = np.sum(
            [Xgaussianderivative2D(tc[0], tc[1], vegf[i], vasccells[i][0], vasccells[i][1], sigma) for i in
             range(len(vasccells))])
        Yvegftc = np.sum(
            [Ygaussianderivative2D(tc[0], tc[1], vegf[i], vasccells[i][0], vasccells[i][1], sigma) for i in
             range(len(vasccells))])
        norm = LA.norm(np.array([Xvegftc, Yvegftc]))

        # preprocessing.normalize(np.array([Xvegftc,Yvegftc]).reshape(1, -1), norm='l2').reshape((2))
        print(norm)

        vasccells.append(tc)

        vector = np.array([Xvegftc / norm, Yvegftc / norm])
        new_tc=(tc[0]+(vegf_coeff*vector[0]), tc[1]+(vegf_coeff*vector[1]))
        print(new_tc)

        if ((new_tc[0]<0) or (new_tc[1]<0)):
            vasccells.append(tc)
        elif ((new_tc[0] > sizemax[1]) or (new_tc[1] > sizemax[1])):
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
            [Xgaussianderivative2D(tc[0], tc[1], vegf[i], neuronposition[i, 0], neuronposition[i, 1], sigma) for i in
             range(neuronposition.shape[0])])
        vegfY = np.sum(
            [Ygaussianderivative2D(tc[0], tc[1], vegf[i], neuronposition[i, 0], neuronposition[i, 1], sigma) for i in
             range(neuronposition.shape[0])])
        normvegf = LA.norm(np.array([vegfX, vegfY]))

        # compute vein contribution
        Xvegftc = np.sum(
            [Xgaussianderivative2D(tc[0], tc[1], 1, veins[i][0], veins[i][1], 20) for i in
             range(len(veins))])
        Yvegftc = np.sum(
            [Ygaussianderivative2D(tc[0], tc[1], 1, veins[i][0], veins[i][1], 20) for i in
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

def forkingUpdate(vasccells,tipscells, tipscells_age, length=10, proba=30):
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
                tipscells.append((tc[0]-1, tc[1]-1))
                tipscells_age[n]=0
                tipscells_age.append(0)

    return(tipscells,vasccells)





def graphUpdate(graph, vegf,tipscells, vasccells, tipscells2, vasccells2, color='r', color2='b'):
    # fig.clear()

    plt.scatter(neuronposition[:, 0], neuronposition[:, 1],c=vegf)
    T=np.stack(tipscells)
    V=np.stack(vasccells)

    T2 = np.stack(tipscells2)
    V2 = np.stack(vasccells2)
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
    plt.scatter(V[:, 0], V[:, 1], c=color)
    plt.scatter(T[:, 0], T[:, 1], c='g')

    plt.scatter(V2[:, 0], V2[:, 1], c=color2)
    plt.scatter(T2[:, 0], T2[:, 1], c='g')

    # plt.scatter(Veins[:, 0], Veins[:, 1], c='b', s=100)
    plt.draw()
    plt.pause(0.01)
    # plt.colorbar()



sigma=20
a=1
neuronposition=generateNeurons(Nb=1000, size=[0,100], weightype='gaussian',a=a, sigma=sigma)

veins=generateNeurons(Nb=20, size=[0,100], weightype='random',a=a, sigma=sigma)

vegf=np.zeros(neuronposition.shape[0])
plt.ion()
fig, ax = plt.subplots()
# plt.plot(neuronposition[:,0], neuronposition[:,1], 'ro')[0]
graph = plt.scatter(neuronposition[:,0], neuronposition[:,1],c=vegf)
plt.pause(0.01)
vasccells=[(0,0)]#, (100,0)]
tipscells=[(0,0)]#, (100,0)]
tipscells_age=np.zeros(len(tipscells)).astype(int)


vasccells2=[(99,99)]#, (100,0)]
tipscells2=[(99,99)]#, (100,0)]
tipscells_age2=np.zeros(len(tipscells2)).astype(int)
tipscells_reco2=[]

tipscells_reco=[]
plt.colorbar()

rep_coeff=2
vegf_coeff=0.4
vein_coeff=2
siga=-1#1.5
sigb=0.01
N=80

a=2
sigma=10#10
# a=10
a_rep=1
sigma_rep=3
lr=1
compute_vein=False
while N>0:
    print(N)
    tipscells,vasccells, vegf, O2, tipscells_reco,tipscells_age=computevegfgradient(neuronposition,vasccells,tipscells,tipscells_reco,siga, sigb, lr, a_rep, sigma_rep, tipscells_age, sizemax=[0,100])
    tipscells_reco,vasccells=computerecogradient(vasccells,tipscells_reco,tipscells_age, sizemax=[0,100])
    tipscells, vasccells=forkingUpdate(vasccells,tipscells,tipscells_age, length=10, proba=100)#50
    # graphUpdate(graph, vegf, tipscells, vasccells, 'r')

    tipscells2, vasccells2, vegf, O2, tipscells_reco2, tipscells_age2 = computevegfgradient(neuronposition, vasccells2,
                                                                                        tipscells2, tipscells_reco2, siga,
                                                                                        sigb, lr, a_rep, sigma_rep,
                                                                                        tipscells_age2, sizemax=[0, 100])
    tipscells_reco2, vasccells2 = computerecogradient(vasccells2, tipscells_reco2, tipscells_age2, sizemax=[0, 100])
    tipscells2, vasccells2 = forkingUpdate(vasccells2, tipscells2, tipscells_age2, length=10, proba=200)  # 50


    N=N-1
    # print(tipscells)

    graphUpdate(graph, vegf, tipscells, vasccells, tipscells2, vasccells2)
    # ani = animation.FuncAnimation(fig, graphUpdate, interval=1000)
    # plt.show()
N=10
compute_vein=True
veins=vasccells2
while N>0:
    tipscells, vasccells, vegf, O2, tipscells_reco,tipscells_age = computevegfgradient(neuronposition, vasccells, tipscells,
                                                                         tipscells_reco, siga, sigb, lr, a_rep,
                                                                         sigma_rep,tipscells_age, sizemax=[0, 100])
    # tipscells,vasccells=computerecoVeins(vasccells,tipscells,veins, sizemax=[0,100])
    graphUpdate(graph, vegf,tipscells, vasccells,tipscells2, vasccells2)
    N = N - 1
N=10
compute_vein=True
veins=vasccells
while N>0:
    tipscells2, vasccells2, vegf, O2, tipscells_reco2,tipscells_age2 = computevegfgradient(neuronposition, vasccells2, tipscells2,
                                                                         tipscells_reco2, siga, sigb, lr, a_rep,
                                                                         sigma_rep,tipscells_age2, sizemax=[0, 100])
    # tipscells,vasccells=computerecoVeins(vasccells,tipscells,veins, sizemax=[0,100])
    graphUpdate(graph, vegf,tipscells, vasccells,tipscells2, vasccells2)
    N = N - 1




Nb=10

sigma=20
a=1
tipscells=generateCapillariesTC(Nb, vasccells,size=[0,100], weightype='gaussian',a=a, sigma=sigma)
a=2
sigma=10#10
# a=10
a_rep=1
sigma_rep=3
lr=1
tipscells_reco=[]
N=5
while N>0:
    print(N)
    tipscells,vasccells, vegf, O2, tipscells_reco=computevegfgradient(neuronposition,vasccells,tipscells,tipscells_reco,siga, sigb, lr, a_rep, sigma_rep, sizemax=[0,100])
    # tipscells_reco,vasccells=computerecogradient(vasccells,tipscells_reco,sizemax=[0,100])
    # tipscells, vasccells=forkingUpdate(vasccells,tipscells,length=10, proba=20)#40
    N=N-1
    print(tipscells)
    graphUpdate(graph, vegf)
    # ani = animation.FuncAnimation(fig, graphUpdate, interval=1000)
    # plt.show()


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
