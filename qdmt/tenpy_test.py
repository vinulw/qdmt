import numpy as np
import matplotlib.pyplot as plt
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg, tebd
from tenpy.networks.site import SpinHalfSite
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.models.lattice import Chain

from analyse import exact_overlap
from loschmidt import loschmidt_paper

from tqdm import tqdm

class TFIModel(CouplingMPOModel):
    r"""Transverse field Ising model on a general lattice. This is modified from the `TFIModel` defined in tenpy to swap the coupling and interaction Pauli basis.

    The Hamiltonian reads:

    .. math ::
        H = - \sum_{\langle i,j\rangle, i < j} \mathtt{J} \sigma^z_i \sigma^z_{j}
            - \sum_{i} \mathtt{g} \sigma^x_i

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs, each pair appearing
    exactly once.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`TFIModel` below.

    Options
    -------
    .. cfg:config :: TFIModel
        :include: CouplingMPOModel

        conserve : None | 'parity'
            What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
        sort_charge : bool | None
            Whether to sort by charges of physical legs.
            See change comment in :class:`~tenpy.networks.site.Site`.
        J, g : float | array
            Coupling as defined for the Hamiltonian above.

    """
    def init_sites(self, model_params):
        sort_charge = model_params.get('sort_charge', None)
        site = SpinHalfSite(conserve='None', sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1.))
        g = np.asarray(model_params.get('g', 1.))
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, 'Sigmax')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'Sigmaz', u2, 'Sigmaz', dx)
        # done


class TFIChain(TFIModel, NearestNeighborModel):
    """The :class:`TFIModel` on a Chain, suitable for TEBD.

    See the :class:`TFIModel` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True

if __name__=="__main__":
    # ######################################################################
    # # Example using iDMRG for ground state sweep
    # ######################################################################
    # exactFile = '../data/exact_gs_energy.csv'
    # g_exact, e_exact = np.loadtxt(exactFile, delimiter=',')

    # D_max = 2
    # eDMRG = []

    # print('iDMRG ground state search...')
    # for g in tqdm(g_exact):
    #     M = TFIChain({"L": 2, "J": 1., "g": g, "bc_MPS": "infinite"})
    #     psi = MPS.from_product_state(M.lat.mps_sites(), [0]*2, "infinite")
    #     dmrg_params = {"trunc_params": {"chi_max": D_max, "svd_min": 1.e-10}}

    #     dmrg.run(psi, M, dmrg_params) # Find the ground state

    #     e = sum(psi.expectation_value(M.H_bond))/psi.L
    #     eDMRG.append(e)

    # print('Error in energy')
    # diff = [np.abs(ex-dmrg) for ex, dmrg in zip(e_exact, eDMRG)]
    # for g, d in zip(g_exact, diff):
    #     print(f'\t{g} : {d}')

    # plt.semilogy(g_exact, diff, 'x--')
    # plt.show()

    ######################################################################
    # Perform time evolution
    ######################################################################
    g1 = 1.5
    g2 = 0.2

    dt = 0.1
    maxTime = 1.3
    stepsPerDt = 10
    dtTEBD = dt / stepsPerDt

    # Save generated As
    savePath = None

    D = 4
    # Prepare the ground state
    print('Preparing ground state')
    M1 = TFIChain({"L": 2, "J": 1., "g": g1, "bc_MPS": "infinite"})
    psi = MPS.from_product_state(M1.lat.mps_sites(), [0]*2, "infinite")
    dmrg_params = {"trunc_params": {"chi_max": D, "svd_min": 1.e-10}}

    dmrg.run(psi, M1, dmrg_params)  # Find the ground state
    e = sum(psi.expectation_value(M1.H_bond))/psi.L
    print(f'\tGround state energy: {e}')

    # Run the evolution
    psi0 = psi.copy()
    A0 = psi.get_B(0, copy=True).to_ndarray()

    M2 = TFIChain({"L": 2, "J": 1., "g": g2, "bc_MPS": "infinite"})
    tebd_params = {
        'order' : 1,
        'dt' : dtTEBD,
        'N_steps': stepsPerDt,
        'trunc_params': {
            'chi_max': D,
            'chi_min': D
        }
    }
    tebdEngine = tebd.TEBDEngine(psi, M2, tebd_params)
    Ats = [A0]
    ts = np.arange(0, maxTime, dt)
    print('Performing evolution...')
    for t in tqdm(ts[1:]):
        tebdEngine.run()
        At = tebdEngine.psi.get_B(0, copy=True).to_ndarray()
        Ats.append(At)

    # Save the generated As
    if savePath is not None:
        np.save(Ats, savePath)

    # Analyse the results
    loschmidt = []
    for A in tqdm(Ats):
        print(A.shape)
        ls = exact_overlap(A0, A)
        ls = -1*np.log(ls**2)
        loschmidt.append(ls)

    analyticsts = np.linspace(0, maxTime, 250)
    analyticLoschmidt = [np.real(loschmidt_paper(t, g1, g2)) for t in analyticsts]

    plt.plot(analyticsts, analyticLoschmidt, '--', label='analytic')
    plt.plot(ts, loschmidt, 'x', label='iTEBD')
    plt.legend()
    plt.show()
