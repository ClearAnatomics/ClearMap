from __future__ import division, print_function
from copy import deepcopy
import pyamg
import igraph as ig
import numpy as np
from pyamg import smoothed_aggregation_solver, rootnode_solver, util

from scipy.sparse import lil_matrix, linalg, coo_matrix
from scipy.sparse.linalg import gmres
from ClearMap.Flow.physiology import Physiology
import pickle
import ClearMap.Flow.units as units
# import g_output
# import vascularGraph
import pdb
import time as ttime
# import vgm

__all__ = ['LinearSystem']
defaultUnits={'length': 'um', 'mass': 'ug', 'time': 'ms'}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class LinearSystem(object):
    def __init__(self, G, withRBC = 0, invivo = 0, dMin_empirical = 3.5, htdMax_empirical = 0.6, verbose = True,**kwargs):
        """
        Computes the flow and pressure field of a vascular graph without RBC tracking.
        It can be chosen between pure plasma flow, constant hematocrit or a given htt/htd
        distribution.
        The pressure boundary conditions (pBC) should be given in mmHg and pressure will be output in mmHg

        INPUT: G: Vascular graph in iGraph format.
               invivo: boolean if the invivo or invitro empirical functions are used (default = 0, i.e. in vitro)
               withRBC: = 0: no RBCs, pure plasma Flow (default)
                        0 < withRBC < 1 & 'htt' not in edgeAttributes: the given value is assigned as htt to all edges.
                        0 < withRBC < 1 & 'htt' in edgeAttributes: the given value is assigned as htt to all edges where htt = None.
                dMin_empiricial: lower limit for the diameter that is used to compute nurel (effective viscosity). The aim of the limit
                        is to avoid using the empirical equations in a range where no experimental data is available (default = 3.5).
                htdMax_empirical: upper limit for htd that is used to compute nurel (effective viscosity). The aim of the limit
                        is to avoid using the empirical equations in a range where no experimental data is available (default = 0.6). Maximum has to be 1.
                verbose: Bool if WARNINGS and setup information is printed
        OUTPUT: None, the edge properties htt is assgined and the function update is executed (see description for more details)
        """
        self._G = G
        self._eps = np.finfo(float).eps
        print(self._eps)
        self._eps=1e-5
        self._P = Physiology(defaultUnits)
        self._muPlasma = self._P.dynamic_plasma_viscosity()
        self._withRBC = withRBC
        self._invivo = invivo
        self._verbose = verbose
        self._dMin_empirical = dMin_empirical
        self._htdMax_empirical = htdMax_empirical

        if self._verbose:
            print('INFO: The limits for the compuation of the effective viscosity are set to')
            print('Minimum diameter %.2f' %self._dMin_empirical)
            print('Maximum discharge %.2f' %self._htdMax_empirical)

        if self._withRBC != 0:
            if self._withRBC < 1.:
                if 'htt' not in G.es.attribute_names():
                    G.es['htt']=[self._withRBC]*G.ecount() #G.ecount() --> total number of edges
                else:
                    httNone = G.es(htt_eq=None).indices 
                    if len(httNone) > 0:
                        G.es[httNone]['htt']=[self._withRBC]*len(httNone)
                    else:
                        if self._verbose:
                            print('WARNING: htt is already an edge attribute. \n Existing values are not overwritten!'+\
                                    '\n If new values should be assigned htt has to be deleted beforehand!')
            else:
                print('ERROR: 0 < withRBC < 1')

        if 'rBC' not in G.vs.attribute_names(): #instead of pressure boundary conditions, in/outflow boundary conditions can be assigned (vertex attributed rBC)
            G.vs['rBC'] = [None]*G.vcount()

        if 'pBC' not in G.vs.attribute_names():
            G.vs['pBC'] = [None]*G.vcount()

        self.update()
        
    #--------------------------------------------------------------------------    
        
    def update(self):
        """Constructs the linear system A x = b where the matrix A contains the 
        conductance information of the vascular graph, the vector b specifies 
        the boundary conditions and the vector x holds the pressures at the 
        vertices (for which the system needs to be solved). 
    
        OUTPUT: matrix A and vector b
        """
        htt2htd = self._P.tube_to_discharge_hematocrit
        nurel = self._P.relative_apparent_blood_viscosity
        G = self._G

        #Convert 'pBC' ['mmHG'] to default Units
        # for v in G.vs(pBC_ne=None):
        #     v['pBC']=v['pBC']*vgm.units.scaling_factor_du('mmHg',G['defaultUnits'])
        
        nVertices = G.vcount()
        b = np.zeros(nVertices)            
        A = lil_matrix((nVertices,nVertices),dtype=float)

        # Compute nominal and specific resistance:
        self._update_nominal_and_specific_resistance()

        #if with RBCs compute effective resistance
        if self._withRBC:
            dischargeHt = [min(htt2htd(htt, d, self._invivo), 1.0) for htt,d in zip(G.es['htt'],G.es['diameter'])]
            G.es['htd'] = dischargeHt

            G.es['effResistance'] =[ res * nurel(max(self._dMin_empirical,d),min(dHt,self._htdMax_empirical),self._invivo) \
                    for res,dHt,d in zip(G.es['resistance'], dischargeHt,G.es['diameter'])]
            G.es['conductance']=1/np.array(G.es['effResistance'])
        else:
            G.es['conductance'] = [1/e['resistance'] for e in G.es]
        
        self._conductance = G.es['conductance']

        for vertex in G.vs:        
            i = vertex.index
            A.data[i] = []
            A.rows[i] = []
            b[i] = 0.0
            if vertex['pBC'] is not None:                   
                A[i,i] = 1.0
                b[i] = vertex['pBC']
            else: 
                aDummy=0
                k=0
                neighbors=[]
                for edge in G.incident(i,'all'): #G.incident --> list of edges attached to vertex i
                    if G.is_loop(edge):
                        continue
                    j=G.neighbors(i)[k] #G.neighbors --> list of vertices neighboring vertex i
                    k += 1
                    conductance = G.es[edge]['conductance']
                    neighbor = G.vs[j]
                    # +=, -= account for multiedges
                    aDummy += conductance
                    if neighbor['pBC'] is not None:
                        b[i] = b[i] + neighbor['pBC'] * conductance
                    else:
                        if j not in neighbors:
                            A[i,j] = - conductance
                        else:
                            A[i,j] = A[i,j] - conductance
                    neighbors.append(j)
                    if vertex['rBC'] is not None:
                        b[i] += vertex['rBC']
                A[i,i]=aDummy
      
        self._A = A
        self._b = b
    
    #--------------------------------------------------------------------------

    def solve(self, method, **kwargs):
        """Solves the linear system A x = b for the vector of unknown pressures
        x, either using a direct solver (obsolete) or an iterative GMRES solver. From the
        pressures, the flow field is computed.
        INPUT: method: This can be either 'direct' or 'iterative2'
        OUTPUT: None - G is modified in place.
                G_final.pkl & G_final.vtp: are save as output
                sampledict.pkl: is saved as output
        """
        b = self._b
        G = self._G
        htt2htd = self._P.tube_to_discharge_hematocrit
        
        A = self._A.tocsr()
        if method == 'direct':
            linalg.use_solver(useUmfpack=True)
            x = linalg.spsolve(A, b)
        elif method == 'iterative2':
             ml = rootnode_solver(A, smooth=('energy', {'degree':2}), strength='evolution' )
             M = ml.aspreconditioner(cycle='V')
             # Solve pressure system
             #x,info = gmres(A, self._b, tol=self._eps, maxiter=1000, M=M)
             x,info = gmres(A, self._b, tol=10*self._eps,M=M)
             if info != 0:
                 print('ERROR in Solving the Matrix')
                 print(info)

        G.vs['pressure'] = x
        self._x = x
        conductance = self._conductance
        G.es['flow'] = [abs(G.vs[edge.source]['pressure'] - G.vs[edge.target]['pressure']) *  \
                        conductance[i] for i, edge in enumerate(G.es)]

        #Default Units - mmHg for pressure
        for v in G.vs:
            v['pressure']=v['pressure']/units.scaling_factor_du('mmHg',G['defaultUnits'])

        #Fahraeus effect --> RBC flow velocity larger than bulk flow velocity
        if self._withRBC: 
	        G.es['v']=[e['htd']/e['htt']*e['flow']/(0.25*np.pi*e['diameter']**2) for e in G.es]
        else:
	        G.es['v']=[e['flow']/(0.25*np.pi*e['diameter']**2) for e in G.es]
        
        #Convert 'pBC' from default Units to mmHg
        pBCneNone=G.vs(pBC_ne=None).indices
        G.vs[pBCneNone]['pBC']=np.array(G.vs[pBCneNone]['pBC'])*(1/units.scaling_factor_du('mmHg',G['defaultUnits']))

        # vgm.write_pkl(G, 'G_final.pkl')
        G.write_pickle( 'G_final.pkl')
        # vgm.write_vtp(G, 'G_final.vtp',False)

        #Write Output
        sampledict={}
        for eprop in ['flow', 'v']:
            if not eprop in sampledict.keys():
                sampledict[eprop] = []
            sampledict[eprop].append(G.es[eprop])
        for vprop in ['pressure']:
            if not vprop in sampledict.keys():
                sampledict[vprop] = []
            sampledict[vprop].append(G.vs[vprop])

	    # g_output.write_pkl(sampledict, 'sampledict.pkl')
        with open('sampledict.pkl', 'wb') as fp:
                pickle.dump(sampledict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    #--------------------------------------------------------------------------
    def _update_nominal_and_specific_resistance(self, esequence=None):
        """Updates the nominal and specific resistance of a given edge 
        sequence.
        INPUT: es: Sequence of edge indices as tuple. If not provided, all 
                   edges are updated.
        OUTPUT: None, the edge properties 'resistance' and 'specificResistance'
                are updated (or created).
        """
        G = self._G

        if esequence is None:
            es = G.es
        else:
            es = G.es(esequence)

        G.es['specificResistance'] = [128 * self._muPlasma / (np.pi * d**4)
                                        for d in G.es['diameter']]

        G.es['resistance'] = [l * sr for l, sr in zip(G.es['length'],
                                                G.es['specificResistance'])]

    #-----------------------------------------------------------------------------
