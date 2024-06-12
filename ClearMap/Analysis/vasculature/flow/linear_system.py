from __future__ import division, print_function
import pickle
from itertools import chain

import numpy as np

from pyamg import rootnode_solver

from scipy.sparse import lil_matrix, linalg
from scipy.sparse.linalg import gmres

from pyximport import pyximport


import ClearMap.Analysis.vasculature.flow.units as units
pyximport.install(setup_args={"include_dirs": [np.get_include()]}, reload_support=True)
from ClearMap.Analysis.vasculature.flow.physiology import Physiology  # dot import for pyx

__all__ = ['LinearSystem']
defaultUnits = {'length': 'um', 'mass': 'ug', 'time': 'ms'}


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class LinearSystem(object):
    def __init__(self, G, withRBC=0, invivo=0, dMin_empirical=3.5, htdMax_empirical=0.6, verbose=True, **kwargs):
        """
        Computes the flow and pressure field of a vascular graph without RBC tracking.
        It can be chosen between pure plasma flow, constant hematocrit or a given htt/htd
        distribution.
        The pressure boundary conditions (pBC) should be given in mmHg and pressure will be output in mmHg

        Parameters
        ----------
        G: igraph.Graph
            Vascular graph in iGraph format.
        withRBC: int
            0: no RBCs, pure plasma Flow (default)
            0 < withRBC < 1 & 'htt' not in edgeAttributes: the given value is assigned as htt to all edges.
            0 < withRBC < 1 & 'htt' in edgeAttributes: the given value is assigned as htt to all edges where htt = None.
        invivo: boolean
            whether the invivo or invitro empirical functions are used (default = 0, i.e. in vitro)
        dMin_empiricial: float
            lower limit for the diameter that is used to compute nurel (effective viscosity). The aim of the limit
            is to avoid using the empirical equations in a range where no experimental data is available (default = 3.5).
        htdMax_empirical: float
            upper limit for htd that is used to compute nurel (effective viscosity). The aim of the limit
            is to avoid using the empirical equations in a range where no experimental data is available (default = 0.6). Maximum has to be 1.
        verbose: bool
            if WARNINGS and setup information is printed

        Returns
        -------
        None, the edge properties htt is assigned and the function update is executed (see description for more details)
        """
        self._G = G
        self._eps = np.finfo(float).eps
        print(self._eps)
        self._eps = 1e-5
        self._P = Physiology(defaultUnits)
        self._muPlasma = self._P.dynamic_plasma_viscosity()
        self._withRBC = withRBC
        self._invivo = invivo
        self._verbose = verbose
        self._dMin_empirical = dMin_empirical
        self._htdMax_empirical = htdMax_empirical

        if self._verbose:
            print('INFO: The limits for the compuation of the effective viscosity are set to')
            print(f'Minimum diameter {self._dMin_empirical:.2f}')
            print(f'Maximum discharge {self._htdMax_empirical:.2f}')

        if self._withRBC != 0:
            if self._withRBC < 1.:
                if 'htt' not in G.es.attribute_names():
                    G.es['htt'] = [self._withRBC] * G.ecount()  # G.ecount() --> total number of edges
                else:
                    httNone = G.es(htt_eq=None).indices
                    if len(httNone) > 0:
                        G.es[httNone]['htt'] = [self._withRBC] * len(httNone)
                    else:
                        if self._verbose:
                            print('WARNING: htt is already an edge attribute. \n Existing values are not overwritten!' + \
                                  '\n If new values should be assigned htt has to be deleted beforehand!')
            else:
                print('ERROR: 0 < withRBC < 1')

        if 'rBC' not in G.vs.attribute_names():  # instead of pressure boundary conditions, in/outflow boundary conditions can be assigned (vertex attributed rBC)
            G.vs['rBC'] = [None] * G.vcount()

        if 'pBC' not in G.vs.attribute_names():
            G.vs['pBC'] = [None] * G.vcount()

        self.update()

    # --------------------------------------------------------------------------

    def update(self):
        """
        Constructs the linear system A x = b where the matrix A contains the
        conductance information of the vascular graph, the vector b specifies
        the boundary conditions and the vector x holds the pressures at the
        vertices (for which the system needs to be solved).

        Returns
        -------
        matrix A and vector b
        """
        htt2htd = self._P.tube_to_discharge_hematocrit
        nurel = self._P.relative_apparent_blood_viscosity
        G = self._G

        # Convert 'pBC' ['mmHG'] to default Units
        # for v in G.vs(pBC_ne=None):
        #     v['pBC']=v['pBC']*vgm.units.scaling_factor_du('mmHg',G['defaultUnits'])

        nVertices = G.vcount()
        b = np.zeros(nVertices)
        A = lil_matrix((nVertices, nVertices), dtype=float)

        # Compute nominal and specific resistance:
        self._update_nominal_and_specific_resistance()

        # if with RBCs compute effective resistance
        if self._withRBC:
            # using numpy notation:
            # dischargeHt = np.minimum(htt2htd(G.es['htt'], G.es['diameter'], self._invivo), 1.0)
            dischargeHt = [min(htt2htd(htt, d, self._invivo), 1.0) for htt, d in zip(G.es['htt'], G.es['diameter'])]
            G.es['htd'] = dischargeHt

            G.es['effResistance'] = [
                res * nurel(max(self._dMin_empirical, d), min(dHt, self._htdMax_empirical), self._invivo) \
                for res, dHt, d in zip(G.es['resistance'], dischargeHt, G.es['diameter'])]
            G.es['conductance'] = 1 / np.array(G.es['effResistance'])
        else:
            G.es['conductance'] = [1 / e['resistance'] for e in G.es]

        self._conductance = G.es['conductance']

        for vertex in G.vs:
            i = vertex.index
            A.data[i] = []
            A.rows[i] = []
            b[i] = 0.0
            if vertex['pBC'] is not None:
                A[i, i] = 1.0
                b[i] = vertex['pBC']
            else:
                aDummy = 0
                k = 0
                neighbors = []
                for edge in G.incident(i, 'all'):  # G.incident --> list of edges attached to vertex i
                    if G.is_loop(edge):
                        continue
                    j = G.neighbors(i)[k]  # G.neighbors --> list of vertices neighboring vertex i
                    k += 1
                    conductance = G.es[edge]['conductance']
                    neighbor = G.vs[j]
                    # +=, -= account for multiedges
                    aDummy += conductance
                    if neighbor['pBC'] is not None:
                        b[i] = b[i] + neighbor['pBC'] * conductance
                    else:
                        if j not in neighbors:
                            A[i, j] = - conductance
                        else:
                            A[i, j] = A[i, j] - conductance
                    neighbors.append(j)
                    if vertex['rBC'] is not None:
                        b[i] += vertex['rBC']
                A[i, i] = aDummy

        self._A = A
        self._b = b

    # --------------------------------------------------------------------------

    def solve(self, method, dest_file_path='sampledict.pkl', **kwargs):
        """
        Solves the linear system A x = b for the vector of unknown pressures
        x, either using a direct solver (obsolete) or an iterative GMRES solver. From the
        pressures, the flow field is computed.

        Parameters
        ----------
        method: str
            This can be either 'direct' or 'iterative2'
        dest_file_path: str
            The path to save the output file

        Returns
        -------
        None - G is modified in place.
        G_final.pkl & G_final.vtp: are save as output
        sampledict.pkl: is saved as output
        """
        # htt2htd = self._P.tube_to_discharge_hematocrit

        A = self._A.tocsr()
        if method == 'direct':
            linalg.use_solver(useUmfpack=True)
            x = linalg.spsolve(A, self._b)
        elif method == 'iterative2':
            ml = rootnode_solver(A, smooth=('energy', {'degree': 2}), strength='evolution')
            M = ml.aspreconditioner(cycle='V')
            # Solve pressure system
            # x,info = gmres(A, self._b, tol=self._eps, maxiter=1000, M=M)
            x, info = gmres(A, self._b, tol=10 * self._eps, M=M)
            if info != 0:
                print('ERROR in Solving the Matrix')
                print(info)
        else:
            raise ValueError(f'ERROR: Unknown method {method}! Choose either "direct" or "iterative2"!')

        G = self._G
        edges = np.array(G.get_edgelist())
        G.vs['pressure'] = x
        self._x = x
        pressure = np.array(x)
        conductance = self._conductance
        G.es['flow'] = abs(pressure[edges[:, 0]] - pressure[edges[:, 1]]) * conductance

        # Default Units - mmHg for pressure
        skl = units.scaling_factor_du('mmHg', G['defaultUnits'])
        G.vs['pressure'] = pressure / skl

        # Fahraeus effect --> RBC flow velocity larger than bulk flow velocity
        if self._withRBC:  # TODO: test
            from time import time
            start = time()
            e_htd = G.es['htd']
            e_htt = G.es['htt']
            e_flow = G.es['flow']
            e_diam = G.es['diameter']
            G.es['v'] = e_htd / e_htt * e_flow / (0.25 * np.pi * e_diam ** 2)
            end1 = time()
            d1 = end1 - start

            G.es['v'] = [e['htd'] / e['htt'] * e['flow'] / (0.25 * np.pi * e['diameter'] ** 2) for e in G.es]
            d2 = time() - end1
        else:
            G.es['v'] = np.array(G.es['flow']) / (0.25 * np.pi * np.array(G.es['diameter']) ** 2)

        # Convert 'pBC' from default Units to mmHg
        pbc_not_none = G.vs(pBC_ne=None).indices
        G.vs[pbc_not_none]['pBC'] = np.array(G.vs[pbc_not_none]['pBC']) * (
                    1 / units.scaling_factor_du('mmHg', G['defaultUnits']))

        G.write_pickle('G_final.pkl')
        # vgm.write_pkl(G, 'G_final.pkl')
        # vgm.write_vtp(G, 'G_final.vtp',False)

        # Write Output
        sample_dict = {
            'flow': G.es['flow'],
            'v': G.es['v'],
            'pressure': G.vs['pressure']
        }
        with open(dest_file_path, 'wb') as fp:
            pickle.dump(sample_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
        return sample_dict

    # --------------------------------------------------------------------------
    def _update_nominal_and_specific_resistance(self, esequence=None):
        """Updates the nominal and specific resistance of a given edge
        sequence.
        Arguments
        ---------
        esequence:
            Sequence of edge indices as tuple. If not provided, all
            edges are updated.
        Returns
        -------
        None, the edge properties 'resistance' and 'specificResistance'
        are updated (or created).
        """
        G = self._G

        if esequence is None:
            es = G.es
        else:
            es = G.es(esequence)

        G.es['specificResistance'] = [128 * self._muPlasma / (np.pi * d ** 4)
                                      for d in G.es['diameter']]

        G.es['resistance'] = [l * sr for l, sr in zip(G.es['length'],
                                                      G.es['specificResistance'])]
