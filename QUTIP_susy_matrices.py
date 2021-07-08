# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # SU(2) mini-BMN Hamiltonians
#
# The following cell defines a function which creates the bosons, then it creates the SU(2) mini-BMN Hamiltonian.
# The function receives the number of bosons ($n_b$), the number of fermions ($n_f$), and the size of the bosons ($n$) as arguments.
# For example, for 6 bosons that are $2 \times 2$ and 3 fermions, the function call would look like `bosonHamiltonians(6, 3, 8)`.
# The function returns four Hamiltonians.
# %% [markdown]
# - First, an annihilation operator and identity matrix are created. Each of the 6 bosons are constructed by taking the Kronecker product of the annihilation operator with the identity matrix, in a specific order. For the $n^{th}$ boson, the annhiliation operator will be the $n^{th}$ term of the Kronecker product. Note that for an $n \times n$ boson, the identity matrix and annhilation operator are $n \times n$. The last term in the Kronecker product is an identity matrix of size $2^{n_f} \times 2^{n_f}$.
# - For example, the first boson's Kronecker product would look like
# $$\hat{a}_n \otimes I_n \otimes I_n \otimes I_n \otimes I_n \otimes I_n \otimes I_{2^{n_f}}$$
# where $\hat{a}$ is the annihilation operator and $I$ is the identity matrix and the subscript denotes the size of the matrix (subscript n means it has size $n \times n$ and subscript $2^{n_f}$ means it has size $2^{n_f} \times 2^{n_f}$).
# - The 2nd boson would look like
# $$I_n \otimes \hat{a}_n \otimes I_n \otimes I_n \otimes I_n \otimes I_n \otimes I_{2^{n_f}}$$
# - Each of the 3 fermions are constructed by taking the Kronecker product of the annihilation operator with the identity matrix and Pauli `Z` matrix, in a specific order. Note that the annihilation operator and identity matrix are always size $2 \times 2$. The first term in the Kronecker product is an identity matrix of size $n^{n_b} \times n^{nb}$, where again, $n \times n$ is the boson size. For the $n^{th}$ fermion, the annhiliation operator will be the $n+1$ term of the Kronecker product and it will be followed by identity matrices and preceded by Pauli Z matrices.
# - For example, the first fermion will look like
# $$I_{n^{n_b}} \otimes \hat{a}_2 \otimes I_2 \otimes I_2$$
# where, the subscript $2$ denotes a size of $2 \times 2$.
# - The second fermion will look like
# $$I_{n^{n_b}} \otimes Z \otimes \hat{a}_2 \otimes I_2$$
# where $Z$ is the Pauli `Z` matrix.
# - The third fermion will look like
# $$I_{n^{n_b}} \otimes Z \otimes Z \otimes \hat{a}_2$$
# %% [markdown]
# ## Use `qutip` to build the operators

# %%
# check versioning
from qutip.ipynbtools import version_table

version_table()


# %%
from qutip import *

# %% [markdown]
# ### Precalculate bosonic and fermionic annihilation /operators

# %%
# cutoff of the modes for the bosonic space
L = 3

# %% [markdown]
# - Annihilation operator for bosons

# %%
a_b = destroy(L)


# %%
a_b

# %% [markdown]
# - Identity for single boson site

# %%
i_b = qeye(L)


# %%
i_b

# %% [markdown]
# - Annihilation for fermions (they are all $2 \times 2$ matrices)

# %%
a_f = destroy(2)


# %%
a_f

# %% [markdown]
# - Pauli `Z` matrix: $\sigma_Z$

# %%
sz = sigmaz()


# %%
sz

# %% [markdown]
# - Identity for fermionic space

# %%
i_f = qeye(2)


# %%
i_f

# %% [markdown]
# ### Bosonic Hilbert space operators

# %%
N_bos = 6  # number of boson sites


# %%
import numpy as np

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


# %%
len(a_b_list)


# %%
a_b_list[0]

# %% [markdown]
# To make it work in the combined space of fermionic and bosonic sites, we need to take a final outer product with the fermionic space identity operator. We will do it later.
# %% [markdown]
# ### Fermionic Hilbert space operators

# %%
N_f = 3  # number of fermion sites

# %% [markdown]
# Create a list of operators just for the fermions

# %%
product_list = [i_f] * N_f  # only the identity for fermions repeated N_f times
a_f_list = []  # this will contain f1...f3
for i in np.arange(0, N_f):  # loop over all bosonic operators
    operator_list = product_list.copy()  # all elements are the identity operator
    operator_list[
        i
    ] = a_f  # the i^th element is now the annihilation operator for a single fermion
    for j in np.arange(0, i):  # the 0:(i-1) elements are replaced by sigma_Z
        operator_list[j] = sz
    a_f_list.append(
        tensor(operator_list)
    )  # do the outer product, add .unit() to tensor if you want it normalized but it will take a long


# %%
len(a_f_list)


# %%
a_f_list[0]

# %% [markdown]
# These fermionic operators need to be preceded by the identity operator for the bosonic space, before taking another outer product. We will do that later.
# %% [markdown]
# ### Combine the two spaces
# %% [markdown]
# - Identity for bosonic space (dimension will be $L^{N_{bos}} \times L^{N_{bos}}$)

# %%
i_b_tot = tensor([qeye(L)] * N_bos)


# %%
i_b_tot.shape


# %%
i_b_tot.dims

# %% [markdown]
# - Identity for fermionic space (dimension will be $2^{N_f} \times 2^{N_f}$)

# %%
i_f_tot = tensor([qeye(2)] * N_f)


# %%
i_f_tot.shape


# %%
i_f_tot.dims

# %% [markdown]
# The $N_{bos}$ total bosonic operators and $N_f$ fermionic operators are constructed in a new list

# %%
op_list = []
for a in a_b_list:
    op_list.append(tensor(a, i_f_tot))
for a in a_f_list:
    op_list.append(tensor(i_b_tot, a))


# %%
len(op_list)


# %%
print([o.dims for o in op_list])

# %% [markdown]
# ### Precompute the position operators for the bosons, needed in the interaction term

# %%
x_list = []
for op in op_list[:N_bos]:
    x_list.append(1 / np.sqrt(2) * (op.dag() + op))


# %%
len(x_list) == N_bos

# %% [markdown]
# ## The full mini-BMN hamiltonian
# %% [markdown]
# - The quadratic terms

# %%
# Create the simple quartic Hamiltonian.
H_q = 0

for a in op_list[:N_bos]:
    H_q = H_q + a.dag() * a

for a in op_list[-N_f:]:
    H_q = H_q + (3.0 / 2) * a.dag() * a

# vacuum energy
H_q = H_q + 0.25 * (2 * N_bos - 3 * N_f - 3)


# %%
H_q.shape


# %%
H_q.dims

# %% [markdown]
# - The interaction term for just bosons

# %%
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


# %%
V_b

# %% [markdown]
# - The interaction term mixing bosons and fermions

# %%
fermions = op_list[-N_f:]
V_bf = (2j / np.sqrt(2)) * (
    (x_list[0] - 1j * x_list[3]) * fermions[1].dag() * fermions[2].dag()
    + (x_list[1] - 1j * x_list[4]) * fermions[2].dag() * fermions[0].dag()
    + (x_list[2] - 1j * x_list[5]) * fermions[0].dag() * fermions[1].dag()
    - (x_list[0] + 1j * x_list[3]) * fermions[2] * fermions[1]
    - (x_list[1] + 1j * x_list[4]) * fermions[0] * fermions[2]
    - (x_list[2] + 1j * x_list[5]) * fermions[1] * fermions[0]
)


# %%
V_bf

# %% [markdown]
# ### Combining the terms and adding a coupling

# %%
g2N = 2.0  # this is the 't hooft coupling
N = 2  # we fix N=2 for SU(2)
g2 = g2N / N
H = H_q + g2 * V_b + np.sqrt(g2) * V_bf

# %% [markdown]
# ## Getting the eigenstates
# %% [markdown]
# * Ground state of the quadratic free Hamiltonian should be zero

# %%
# print("The ground state energy of the free Hamiltonian:")
# if L<4:
#     print(H_q.eigenenergies(eigvals=10))
# else:
#     print(H_q.eigenenergies(eigvals=10,sparse=True))


# %%
if L == 2:
    eig = H.eigenenergies()
else:
    eig = H.eigenenergies(eigvals=20, sparse=True)


# %%
print(f"The ground state energy (H): {eig[0]}")


# %%
print(f"The 10 lowest eigen energies at lambda={g2N}: {eig}")


# %%
# import matplotlib.pyplot as plt

# plt.plot(eig)

# %% [markdown]
# ## The gauge generator operators
# %% [markdown]
# These are similar to the bosonic BMN case, but with the addition of the fermionic operators:
#
# $$ \hat{G}_\alpha = i\sum_{\beta,\gamma}\epsilon_{\alpha\beta\gamma} \left( \hat{a}_{1\beta}^\dagger\hat{a}_{1\gamma} + \hat{a}_{2\beta}^\dagger\hat{a}_{2\gamma} + \hat{\xi}_\beta^\dagger\hat{\xi}_\gamma \right).  $$
# %% [markdown]
# * Save the bosonic operators in a separate list for convenience

# %%
bosons = op_list[:N_bos]

# %% [markdown]
# * Make a list with the 3 generators

# %%
g_list = [0] * 3
g_list[0] = 1j * (
    bosons[1].dag() * bosons[2]
    - bosons[2].dag() * bosons[1]
    + bosons[4].dag() * bosons[5]
    - bosons[5].dag() * bosons[4]
    + fermions[1].dag() * fermions[2]
    - fermions[2].dag() * fermions[1]
)


# %%
g_list[1] = 1j * (
    bosons[2].dag() * bosons[0]
    - bosons[0].dag() * bosons[2]
    + bosons[5].dag() * bosons[3]
    - bosons[3].dag() * bosons[5]
    + fermions[2].dag() * fermions[0]
    - fermions[0].dag() * fermions[2]
)


# %%
g_list[2] = 1j * (
    bosons[0].dag() * bosons[1]
    - bosons[1].dag() * bosons[0]
    + bosons[3].dag() * bosons[4]
    - bosons[4].dag() * bosons[3]
    + fermions[0].dag() * fermions[1]
    - fermions[1].dag() * fermions[0]
)


# %%
g_sum = sum([g * g for g in g_list])
g_sum

# %% [markdown]
# ### Measure the sum of the square of the operators on the groundstate

# %%
eigv0_H, eigk0_H = H.groundstate(sparse=True)


# %%
print(f"The ground state energy (H): {eigv0_H}")


# %%
gs = expect(g_sum, eigk0_H)


# %%
print(f"Ground state gauge singlet violation (H): {gs}")

# %% [markdown]
# ## Modified Hamiltonian: adding an energy penalty term
#
# We can change the Hamiltonian to add a term proportional to the gauge singlet constraint violation in order to penalize (with a higher energy) those states which are not gauge singlets.
# $$ \tilde{H} = H + \Lambda \sum_\alpha \hat{G}^2_\alpha $$

# %%
# we choose the penalty coefficient equal to the cutoff for the bosons
penalty = L
H_l = H + penalty * g_sum

# %% [markdown]
# The ground state will not change much if it is gauge invariant as expected

# %%
eigv0_H_l, eigk0_H_l = H_l.groundstate(sparse=True)


# %%
print(f"The ground state energy (H_l): {eigv0_H_l}")

# %% [markdown]
# Its gauge-singlet violation should be smaller

# %%
gs = expect(g_sum, eigk0_H_l)


# %%
print(f"Ground state gauge singlet violation (H_l): {gs}")

# %% [markdown]
# ## The rotation generator operators
# %% [markdown]
# We need to combine the position and momentum operators of the two bosonic matrices (6 total degrees of freedom):
#
# $$ \hat{Z} = \frac{X_1-iX_2}{\sqrt{2}} $$
#
# and
#
# $$\hat{P}_Z = \frac{\hat{P}_1-i\hat{P}_2}{\sqrt{2}}$$
# %% [markdown]
# ### Precompute the momentum operators for the bosons, needed for the generators

# %%
p_list = []
for op in op_list[:N_bos]:
    p_list.append(1j / np.sqrt(2) * (op.dag() - op))

# %% [markdown]
# ### Precompute the combinations of position and momentum operators to manifest the SO(2) symmetry

# %%
z_list = [0] * 3
z_list[0] = 1 / np.sqrt(2) * (x_list[0] - 1j * x_list[3])
z_list[1] = 1 / np.sqrt(2) * (x_list[1] - 1j * x_list[4])
z_list[2] = 1 / np.sqrt(2) * (x_list[2] - 1j * x_list[5])


# %%
pz_list = [0] * 3
pz_list[0] = 1 / np.sqrt(2) * (p_list[0] - 1j * p_list[3])
pz_list[1] = 1 / np.sqrt(2) * (p_list[1] - 1j * p_list[4])
pz_list[2] = 1 / np.sqrt(2) * (p_list[2] - 1j * p_list[5])

# %% [markdown]
# ### Compute the generators for the SO(2) rotations
#
# $$\hat{M} = \sum_\alpha \big[ i(\hat{Z}_\alpha\hat{P}_{Z\alpha}^\dagger-\hat{Z}_\alpha^\dagger\hat{P}_{Z\alpha}) -\frac{1}{2}\hat{\xi}_\alpha^\dagger\hat{\xi}_\alpha \big]$$

# %%
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


# %%
m_sum = sum([m for m in m_list])
m_sum

# %% [markdown]
# ### Measure the expectation value on the ground state of the unperturbed Hamiltonian

# %%
gm = expect(m_sum, eigk0_H)


# %%
print(f"Ground state rotation generator value (H): {gm}")

# %% [markdown]
# ### Measure $\hat{G}^2$ and $\hat{M}$ for the different Hamiltonians

# %%
print(f"--- Unperturbed Hamiltonian H --- coupling lambda={g2N}")
print("Eigenenergies:", eigv0_H)
print("Expectation values of H:", expect(H, eigk0_H))
print("Expectation values G^2:", expect(g_sum, eigk0_H))
print("Expectation values M:", expect(m_sum, eigk0_H))


# %%
print(f"--- Perturbed Hamiltonian H_l = H+L*G^2 --- coupling lambda={g2N}")
print("Eigenenergies:", eigv0_H_l)
print("Expectation values of H:", expect(H, eigk0_H_l))
print("Expectation values G^2:", expect(g_sum, eigk0_H_l))
print("Expectation values M:", expect(m_sum, eigk0_H_l))

# %% [markdown]
# Look at a few lowest eigenstates for the unperturbed Hamiltonian

# %%
eigv_H, eigk_H = H.eigenstates(sparse=True, sort="low", eigvals=4)


# %%
print(f"--- Unperturbed Hamiltonian H --- coupling lambda={g2N}")
print("Eigenenergies:", eigv_H)
print("Expectation values of H:", expect(H, eigk_H))
print("Expectation values G^2:", expect(g_sum, eigk_H))
print("Expectation values M:", expect(m_sum, eigk_H))

# %% [markdown]
# Look at a few lowest eigenstates for the perturbed Hamiltonian with $\hat{G}^2$

# %%
eigv_H_l, eigk_H_l = H_l.eigenstates(sparse=True, sort="low", eigvals=4)


# %%
print(f"--- Perturbed Hamiltonian H_l = H+L*G^2 --- coupling lambda={g2N}")
print("Eigenenergies:", eigv_H_l)
print("Expectation values of H:", expect(H, eigk_H_l))
print("Expectation values G^2:", expect(g_sum, eigk_H_l))
print("Expectation values M:", expect(m_sum, eigk_H_l))

# %% [markdown]
# ## Modified Hamiltonian (part II): adding a penalty term for angular momentum
# %% [markdown]
# We will add a new term to the Hamiltonian:
# $$ \hat{H}' = \hat{H} + c\sum_\alpha\hat{G}_\alpha^2 + c' (\hat{M}-J)^2. $$

# %%
# we choose the penalty coefficient equal to the cutoff for the bosons
penalty_l = L
penalty_m = 1
J = 0  # {0,0.5}
H_ml = H + penalty_l * g_sum + penalty_m * (m_sum - J) ** 2


# %%
H_ml

# %% [markdown]
# ### Compute the ground state energy
#
# Find the lowest eigenstates and compute the expectation values of $\hat{G}^2$ and $\hat{M}$
# %% [markdown]
# Only the groundstate

# %%
eigv0_H_ml, eigk0_H_ml = H_ml.groundstate(sparse=True)


# %%
print(f"--- Perturbed Hamiltonian H_ml = H+L*G^2+M^2 --- coupling lambda={g2N}")
print("Eigenenergies:", eigv0_H_ml)
print("Expectation values of H:", expect(H, eigk0_H_ml))
print("Expectation values G^2:", expect(g_sum, eigk0_H_ml))
print("Expectation values M:", expect(m_sum, eigk0_H_ml))

# %% [markdown]
# The 4 lowest eigenstates

# %%
eigv_H_ml, eigk_H_ml = H_ml.eigenstates(sparse=True, sort="low", eigvals=4, tol=1e-8)


# %%
print(f"--- Perturbed Hamiltonian H_ml = H+L*G^2+M^2 --- coupling lambda={g2N}")
print("Eigenenergies:", eigv_H_ml)
print("Expectation values of H:", expect(H, eigk_H_ml))
print("Expectation values G^2:", expect(g_sum, eigk_H_ml))
print("Expectation values M:", expect(m_sum, eigk_H_ml))
