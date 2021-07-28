from qutip import *
import numpy as np
import time
import fire


def build_operators(L: int, N_bos: int) -> list:
    """Generate all the annihilation operators needed to build the hamiltonian

    Args:
        L (int): the cutoff of the single site Fock space
        N_bos (int): the number of bosonic sites

    Returns:
        list: a list of annihilation operators, length=N_bos 
    """
    ### our basis operators are the annihilation and the identity for bosons
    a_b = destroy(L)
    i_b = identity(L)

    # generically speaking, we construct the list of bosons and then take the outer product
    product_list = [i_b] * N_bos  # only the identity for bosons repeated N_bos times
    a_b_list = []  # this will contain a1...a6
    for i in np.arange(0, N_bos):  # loop over all bosonic operators
        operator_list = product_list.copy()  # all elements are the identity operator
        operator_list[
            i
        ] = a_b  # the i^th element is now the annihilation operator for a single boson
        a_b_list.append(
            tensor(operator_list)
        )  # do the outer product, add .unit() to tensor if you want it normalized but it will take a long
 
    return a_b_list


def build_gauge_generators(L: int, N_bos: int) -> list:
    """Generate the gauge generators operators

    Args:
        L (int): the single site cutoff of the Fock space
        N_bos (int): the number of bosonic sites

    Returns:
        list : 3 generators (for SU(2))
    """
    # generate the annihilation operators
    bosons = build_operators(L, N_bos)
    # define the generator list for SU(2)
    g_list = [0] * 3
    g_list[0] = 1j * (
        bosons[1].dag() * bosons[2]
        - bosons[2].dag() * bosons[1]
        + bosons[4].dag() * bosons[5]
        - bosons[5].dag() * bosons[4]
    )
    g_list[1] = 1j * (
        bosons[2].dag() * bosons[0]
        - bosons[0].dag() * bosons[2]
        + bosons[5].dag() * bosons[3]
        - bosons[3].dag() * bosons[5]
    )
    g_list[2] = 1j * (
        bosons[0].dag() * bosons[1]
        - bosons[1].dag() * bosons[0]
        + bosons[3].dag() * bosons[4]
        - bosons[4].dag() * bosons[3]
    )

    return g_list


def build_hamiltonian(L: int, N_bos: int, g2N: float) -> Qobj:
    """Build the bosonic BMN2 SU(2) hamiltonian with interaction strength lambda=g2N
    and N_bos bosons

    Args:
        L (int): the cutoff of the bosonic Fock space
        N_bos (int): the number of bosonic sites
        g2N (float): the 'tHooft coupling

    Returns:
        Qobj: the Hamiltoninan operator
    """
    # generate the annihilation operators
    bosons = build_operators(L, N_bos)
    # bosonic position operators
    x_list = []
    for op in bosons:
        x_list.append(1 / np.sqrt(2) * (op.dag() + op))

    ### Harmonic oscillator
    # Create the simple quartic Hamiltonian.
    H_q = 0

    for a in bosons:
        H_q = H_q + a.dag() * a
    # vacuum energy
    H_q = H_q + 0.25 * (2 * N_bos)

    ### Quartic Interaction for bosons ONLY
    V_b = (
        x_list[2] * x_list[2] * x_list[3] * x_list[3]
        + x_list[2] * x_list[2] * x_list[4] * x_list[4]
        + x_list[1] * x_list[1] * x_list[3] * x_list[3]
        + x_list[1] * x_list[1] * x_list[5] * x_list[5]
        + x_list[0] * x_list[0] * x_list[4] * x_list[4]
        + x_list[0] * x_list[0] * x_list[5] * x_list[5]
        - 2 * x_list[0] * x_list[2] * x_list[3] * x_list[5]
        - 2 * x_list[0] * x_list[1] * x_list[3] * x_list[4]
        - 2 * x_list[1] * x_list[2] * x_list[4] * x_list[5]
    )
    # g^2 = g2N/2
    return H_q + (g2N / 2) * V_b


def build_penalty_hamiltonian(h0: Qobj, g_list: list, penalty_L: float) -> Qobj:
    """Build the modified Hamiltonian with a penalty term to suppress gauge non-singlet states

    Args:
        h0 (Qobj): The original Hamiltonian
        g_list (list): the list of 3 gauge generator operators
        penalty_L (float): the coefficient of the penalty term

    Returns:
        Qobj: the modified Hamiltonian
    """
    g_sum = sum([g * g for g in g_list])
    return h0 + penalty_L * g_sum


def print_out(Ls, Es, name):
    names = [f"{name}{i}" for i in np.arange(Es.shape[-1])]
    print("Lambda", *names, sep=",")
    for a, b in zip(Ls, Es):
        print(a, *b, sep=",")


def diagonalize_hamiltonian(N, Nmat, g2, num=1, sparse=True, tolerance=1e-06):
    """Build the bosonic BMN2 SU(2) hamiltonian with interaction strength g
    and find the ground state energy. Should use the sparse solutions when N is larger than 4"""
    if (not sparse) and (N > 4):
        print("Do not recommend dense eigensolvers unless you have very large memory!")
        return
    H = build_hamiltonian(N, Nmat, g2)
    # if tolerance is = 0 it is done at machine precision
    return H.eigenenergies(sparse=sparse, tol=tolerance, eigvals=num, maxiter=1000000)


def main(num_eigs: int, L_range: list, l_range: list, penalty: bool):
    """Run the eigensolver (sparse) for the Hamiltonian of 6 bosons with penalty terms.
    The cutoff for each boson is given by the L_range list and the coupling constant ('t Hooft) is given by
    the l_range list.
    The eigensolver returns num_eigs eigenvalues and eigenvectors from the lowest energy.
    The code then measure the expectation value of the Hamiltonian without penalties and of the gauge generators.

    Args:
        num_eigs (int): number of lowest eigenvectors and eigenenergies
        L_range (list): list of cutoff values to use for each boson
        l_range (list): list of 't Hooft coupling constants
        penalty (bool): if the eigenvectors should come from the Hamiltonian with a penalty term
    """
    Nbos = 6  # fixed for SU(2) with 2 matrices: number of bosons
    num_eig = num_eigs  # how many eigen states to consider
    cutoff_range = L_range  # range of cutoffs to study for the Fock space
    g2N = l_range  # 'tHooft coupling lambda
    start_time = time.time()
    for g in g2N:
        print(f"----- Coupling={g}")
        gs = []
        gv = []
        for cutoff in cutoff_range:
            penalty_L = float(cutoff)
            hamiltonian_orig = build_hamiltonian(cutoff, Nbos, g)
            G2_ops = build_gauge_generators(cutoff, Nbos)
            hamiltonian_p = hamiltonian_orig.copy()
            if penalty:
                hamiltonian_p = build_penalty_hamiltonian(
                    hamiltonian_orig, G2_ops, penalty_L
                )
            print(f"--- Computing eigenvalues at cutoff: {cutoff}")
            _, eigk = hamiltonian_p.eigenstates(
                sparse=True, sort="low", eigvals=num_eig, tol=0
            )
            gs.append(expect(hamiltonian_orig, eigk))
            gv.append(expect(sum([g * g for g in G2_ops]), eigk))
            print(f"Finished computing expectation values.")
        gs = np.array(gs).reshape(-1, num_eig).real
        gv = np.array(gv).reshape(-1, num_eig).real
        print_out(cutoff_range, gs, "Energy")
        print_out(cutoff_range, gv, "GaugeViolation")

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Program runtime: {runtime} seconds")


if __name__ == "__main__":
    fire.Fire(main)
