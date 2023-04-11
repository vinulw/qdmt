import numpy as np
import matplotlib.pyplot as plt
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import dmrg

if __name__=="__main__":
    M = TFIChain({"L": 2, "J": 1., "g": 1.5, "bc_MPS": "infinite"})
    print(M)
    psi = MPS.from_product_state(M.lat.mps_sites(), [0]*2, "infinite")
    dmrg_params = {"trunc_params": {"chi_max": 30, "svd_min": 1.e-10}}

    dmrg.run(psi, M, dmrg_params)
    print("E =", sum(psi.expectation_value(M.H_bond))/psi.L)
    print("final bond dimensions: ", psi.chi)
