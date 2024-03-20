import numpy as np
import matplotlib.pyplot as plt

def fit_for_pressureBC():
    #Available Data --> see Schmid et al. 2017 for references: https://doi.org/10.1371/journal.pcbi.1005392
    #Werber 1984, Pial Vessels Rat (normotensive Wistar Kyoto rat) 
    aP=[[60.],[3.0]] #[[list of mean pressures],[list of associated stds]]
    aD=[[77.],[4.]] #[[list of mean diameters],[list of associated stds]]
    vP=[[8.],[1.0]]
    vD=[[180.],[16.]]
    
    #Harper 1984, Pial Rat (normotensive Wistar Kyoto rat)
    aP1=[[55.6,51.8,34.6,27.2],[5.0,3.3,3.1,1.0]]
    aD1=[[51.2,34.3,21.6,12.5],[3.3,2.2,1.3,0.6]]
    vP1=[[12.2,13.8,15.8],[1.,1.,1.2]]
    vD1=[[160.6,76.2,34.2],[10.7,8.1,2.5]]
    
    #Hudetz 1987 Rat Pial (normotensive)
    aP2=[[79.7,72.7,63.7,60.7,57],[0,0,0,0,0]]
    aD2=[[191,168,106,71,50],[7,22,22,19,12]]
    
    #Shapiro 1971 Cat pial
    aP3=[[40.6,47.5,50.0,52,52.8],[0,0,0,0,0]]
    aD3=[[25,30,35,40,50],[0,0,0,0,0]]
    
    #estimate value for D = 0 for the fit based on the smallest arteriole and venule diameter
    #min(vD)=34.2, vP=15.8 --> 15.8 = m*(34.2) + b
    #min(aD)=-12.5, aP=27.2 --> 27.2 = m*(-12.5) + b
    m = - 0.244; b = 24.145
    f_x = lambda x,m,b: x*m+b
    d0=0.; p0=f_x(0,m,b)
    
    #Data to fit:
    #Artery
    aPall=np.concatenate([aP[0],aP1[0],aP2[0],aP3[0],[p0]])
    aDall=np.concatenate([aD[0],aD1[0],aD2[0],aD3[0],[d0]])
    z=np.polyfit(aDall,aPall,3)
    #Vein
    vPall=np.concatenate([vP[0],vP1[0],[p0]])
    vDall=np.concatenate([vD[0],vD1[0],[d0]])
    z2=np.polyfit(vDall,vPall,3)
    
    polynomial_a = np.poly1d(z)
    polynomial_v = np.poly1d(z2) #NOTE I don't use the venule fit but assign a constant value of 10 mmHg
    
    return polynomial_a, polynomial_v

def assign_pressureBC(G,polynomial_a):
    G.vs['degree'] = G.degree()
    G.vs['pBC'] = [None]*G.vcount()
    G.vs(degree_eq=1,nkind_eq=3)['pBC'] = 10. #NOTE nkind = 3 --> label for venules 
    for v in G.vs(degree_eq=1,nkind_ne=3): #NOTE nkind != 3, i.e. nkind = 2 --> label for arterioles and nkind = 4 for capillaries/everything else 
        v['pBC'] = np.polyval(polynomial_a,G.es[G.incident(v)[0]]['diameter'])

    return G

