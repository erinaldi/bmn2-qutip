from qutip import *
import numpy as np
import time
import fire


def build_operators(L: int, N_bos: int, N_fer: int) -> list:
    """Generate all the annihilation operators needed to build the hamiltonian

    Args:
        L (int): the cutoff of the single site Fock space
        N_bos (int): the number of bosonic sites
        N_fer (int): the number of fermionic sites

    Returns:
        list: a list of annihilation operators, N_bos followed by N_fer
    """
    ### our basis operators are the annihilation and the identity for bosons and fermions
    a_b = destroy(L)
    i_b = identity(L)
    a_f = destroy(2)
    i_f = identity(2)
    sz = sigmaz()
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
    # same for the fermions
    product_list = [i_f] * N_fer  # only the identity for fermions repeated N_f times
    a_f_list = []  # this will contain f1...f3
    for i in np.arange(0, N_fer):  # loop over all bosonic operators
        operator_list = product_list.copy()  # all elements are the identity operator
        operator_list[
            i
        ] = a_f  # the i^th element is now the annihilation operator for a single fermion
        for j in np.arange(0, i):  # the 0:(i-1) elements are replaced by sigma_Z
            operator_list[j] = sz
        a_f_list.append(
            tensor(operator_list)
        )  # do the outer product, add .unit() to tensor if you want it normalized but it will take a long
    # need the Identity for bosonic space (dimension will be $L^{N_{bos}} \times L^{N_{bos}}$)
    i_b_tot = tensor([identity(L)] * N_bos)
    # and the Identity for fermionic space (dimension will be $2^{N_f} \times 2^{N_f}$)
    i_f_tot = tensor([identity(2)] * N_fer)
    # build the operators in the global Hilbert space
    op_list = []
    for a in a_b_list:
        op_list.append(tensor(a, i_f_tot))
    for a in a_f_list:
        op_list.append(tensor(i_b_tot, a))

    return op_list


def build_gauge_generators(L: int, N_bos: int, N_f: int) -> list:
    """Generate the gauge generators operators

    Args:
        L (int): the single site cutoff of the Fock space
        N_bos (int): the number of bosonic sites
        N_f (int): the number of fermionic sites

    Returns:
        list : 3 generators (for SU(2))
    """
    # generate the annihilation operators
    op_list = build_operators(L, N_bos, N_f)
    bosons = op_list[:N_bos]
    fermions = op_list[-N_f:]
    # define the generator list for SU(2)
    g_list = [0] * 3
    g_list[0] = 1j * (
        bosons[1].dag() * bosons[2]
        - bosons[2].dag() * bosons[1]
        + bosons[4].dag() * bosons[5]
        - bosons[5].dag() * bosons[4]
        + fermions[1].dag() * fermions[2]
        - fermions[2].dag() * fermions[1]
    )
    g_list[1] = 1j * (
        bosons[2].dag() * bosons[0]
        - bosons[0].dag() * bosons[2]
        + bosons[5].dag() * bosons[3]
        - bosons[3].dag() * bosons[5]
        + fermions[2].dag() * fermions[0]
        - fermions[0].dag() * fermions[2]
    )
    g_list[2] = 1j * (
        bosons[0].dag() * bosons[1]
        - bosons[1].dag() * bosons[0]
        + bosons[3].dag() * bosons[4]
        - bosons[4].dag() * bosons[3]
        + fermions[0].dag() * fermions[1]
        - fermions[1].dag() * fermions[0]
    )

    return g_list


def build_rotation_generators(L: int, N_bos: int, N_f: int) -> list:
    """Generate the gauge generators operators

    Args:
        L (int): the single site cutoff of the Fock space
        N_bos (int): the number of bosonic sites
        N_f (int): the number of fermionic sites

    Returns:
        List : 3 generators (for SO(2))
    """
    # generate the annihilation operators
    op_list = build_operators(L, N_bos, N_f)
    bosons = op_list[:N_bos]
    fermions = op_list[-N_f:]
    # build momentum and position operators
    p_list = []
    x_list = []
    for op in bosons:
        p_list.append(1j / np.sqrt(2) * (op.dag() - op))
        x_list.append(1 / np.sqrt(2) * (op.dag() + op))
    # start from z and pz
    z_list = [0] * 3
    z_list[0] = 1 / np.sqrt(2) * (x_list[0] - 1j * x_list[3])
    z_list[1] = 1 / np.sqrt(2) * (x_list[1] - 1j * x_list[4])
    z_list[2] = 1 / np.sqrt(2) * (x_list[2] - 1j * x_list[5])
    pz_list = [0] * 3
    pz_list[0] = 1 / np.sqrt(2) * (p_list[0] - 1j * p_list[3])
    pz_list[1] = 1 / np.sqrt(2) * (p_list[1] - 1j * p_list[4])
    pz_list[2] = 1 / np.sqrt(2) * (p_list[2] - 1j * p_list[5])
    # define the generator list for SO(2)
    m_list = [0] * 3
    m_list[0] = (
        1j * (z_list[0] * pz_list[0].dag() - z_list[0].dag() * pz_list[0])
        - 0.5 * fermions[0].dag() * fermions[0]
    )
    m_list[1] = (
        1j * (z_list[1] * pz_list[1].dag() - z_list[1].dag() * pz_list[1])
        - 0.5 * fermions[1].dag() * fermions[1]
    )
    m_list[2] = (
        1j * (z_list[2] * pz_list[2].dag() - z_list[2].dag() * pz_list[2])
        - 0.5 * fermions[2].dag() * fermions[2]
    )

    return m_list


def build_hamiltonian(L, N_bos, N_f, g2N):
    """Build the supersymmetric BMN2 SU(2) hamiltonian with interaction strength lambda=g2N
    and N_bos bosons and N_f fermions
    """
    # generate the annihilation operators
    op_list = build_operators(L, N_bos, N_f)
    bosons = op_list[:N_bos]
    fermions = op_list[-N_f:]
    # bosonic position operators
    x_list = []
    for op in bosons:
        x_list.append(1 / np.sqrt(2) * (op.dag() + op))

    ### Harmonic oscillator
    # Create the simple quartic Hamiltonian.
    H_q = 0

    for a in bosons:
        H_q = H_q + a.dag() * a

    for a in fermions:
        H_q = H_q + (3.0 / 2) * a.dag() * a

    # vacuum energy
    H_q = H_q + 0.25 * (2 * N_bos - 3 * N_f - 3)

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
    ### Quartic interactions for bosons and fermions
    V_bf = (2j / np.sqrt(2)) * (
        (x_list[0] - 1j * x_list[3]) * fermions[1].dag() * fermions[2].dag()
        + (x_list[1] - 1j * x_list[4]) * fermions[2].dag() * fermions[0].dag()
        + (x_list[2] - 1j * x_list[5]) * fermions[0].dag() * fermions[1].dag()
        - (x_list[0] + 1j * x_list[3]) * fermions[2] * fermions[1]
        - (x_list[1] + 1j * x_list[4]) * fermions[0] * fermions[2]
        - (x_list[2] + 1j * x_list[5]) * fermions[1] * fermions[0]
    )
    # g^2 = g2N/2
    return H_q + (g2N / 2) * V_b + np.sqrt((g2N / 2)) * V_bf


def build_penalty_hamiltonian(h0, g_list, m_list, penalty_L, penalty_M, J):
    g_sum = sum([g * g for g in g_list])
    m_sum = sum([m for m in m_list])
    return h0 + penalty_L * g_sum + penalty_M * (m_sum - J) ** 2


def print_out(Ls, Es, name):
    names = [f"{name}{i}" for i in np.arange(Es.shape[-1])]
    print("Lambda", *names, sep=",")
    for a, b in zip(Ls, Es):
        print(a, *b, sep=",")


def main(num_eigs: int, L_range: list, l_range: list, j: float, penalty: bool = True):
    """Run the eigensolver (sparse) for the Hamiltonian of 6 bosons and 3 fermions with penalty terms.
    The cutoff for each boson is given by the L_range list and the coupling constant ('t Hooft) is given by
    the l_range list. The eigensolver returns num_eigs eigenvalues and eigenvectors from the lowest energy.
    The code then measure the expectation value of the Hamiltonian without penalties, of the gauge generators
    and of the angular momentum operator.

    Args:
        num_eigs (int): number of lowest eigenvectors and eigenenergies
        L_range (list): list of cutoff values to use for each boson
        l_range (list): list of 't Hooft coupling constants
        j (float): the angular momentum sector
        penalty (bool): if the eigenvectors should come from the Hamiltonian with penalty terms
    """
    Nbos = 6  # fixed for SU(2) with 2 matrices: number of bosons
    Nf = 3  # fixed for SU(2) with 2 matrices in the mini-BMN model
    num_eig = num_eigs  # how many eigen states to consider
    cutoff_range = L_range  # range of cutoffs to study for the Fock space
    g2N = l_range  # 'tHooft coupling lambda
    ang_mom = j  # J value
    start_time = time.time()
    for g in g2N:
        print(f"----- Coupling={g}")
        gs = []
        gv = []
        gm = []
        for cutoff in cutoff_range:
            penalty_L = cutoff  # coefficient of G^2
            penalty_m = 10.0 * cutoff  # coefficient of (M-J)^2
            hamiltonian_orig = build_hamiltonian(cutoff, Nbos, Nf, g)
            G2_ops = build_gauge_generators(cutoff, Nbos, Nf)
            M_ops = build_rotation_generators(cutoff, Nbos, Nf)
            hamiltonian_p = hamiltonian_orig.copy()
            if penalty:
                hamiltonian_p = build_penalty_hamiltonian(
                    hamiltonian_orig, G2_ops, M_ops, penalty_L, penalty_m, ang_mom
                )
            print(f"--- Computing eigenvalues at cutoff: {cutoff}")
            _, eigk = hamiltonian_p.eigenstates(
                sparse=True, sort="low", eigvals=num_eig, tol=0
            )
            gs.append(expect(hamiltonian_orig, eigk))
            gv.append(expect(sum([g * g for g in G2_ops]), eigk))
            gm.append(expect(sum([m for m in M_ops]), eigk))
            print(f"Finished computing expectation values.")
        gs = np.array(gs).reshape(-1, num_eig).real
        gv = np.array(gv).reshape(-1, num_eig).real
        gm = np.array(gm).reshape(-1, num_eig).real
        print_out(cutoff_range, gs, "Energy")
        print_out(cutoff_range, gv, "GaugeViolation")
        print_out(cutoff_range, gm, "AngMomentum")

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Program runtime: {runtime} seconds")


if __name__ == "__main__":
    fire.Fire(main)
