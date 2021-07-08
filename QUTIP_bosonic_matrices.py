# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
# # Bosonic Matrices

# %%
# check versioning
from qutip.ipynbtools import version_table

version_table()

# %% [markdown]
# ## The Fock basis

# %%
from qutip import *


# %%
import numpy as np

# %% [markdown]
# Define the dimension of the Fock space. This should be small enough to use existing quantum hardware. The dimension of the Fock space (or the cutoff) is equal to the number of qubits we can use.
#
# We call the cutoff $\Lambda$ in the notes. Here we use $L$ instead

# %%
# for L=5 we will have each boson represented as a 5x5 matrix
# qutip stores operators as sparse matrices
L = 4

# %% [markdown]
# For the *bosonic* mini-BMN model with group SU(2) we have $2^2-1=3$ bosonic degrees of freedom per matrix. We consider the 2 matrix case, so we will have 6 bosons in total.
#
# In the future it would be best to disentangle the number of matrices of the model (2) from the generators of the gauge group (3).

# %%
# for Nmat=6 and the SU(2) case, that means we are considering the BMN model with 2 matrices
Nmat = 6

# %% [markdown]
# To construct each bosonic annihilation operator we start from the $L \times L$ annihiliation operator in the Fock basis of one boson and then use the outer product with the identity.
#
# * annihilation operator for one harmonic oscillator in a space truncated to L levels

# %%
a = destroy(L)
print(a)


# %%
# this takes time as the matrix size grows because it requires the dense form of the matrix
a.norm()

# %% [markdown]
# If we want to normalize it to unit norm we can simply call `a.unit()`
# %% [markdown]
# * identity operator in the Fock space of N levels

# %%
id = identity(L)  # same as qeye(N)
print(id)


# %%
id.norm()

# %% [markdown]
# Same for the identity: `id.unit()`
# %% [markdown]
# For the the 6 bosons, the annihilation operators are tensor products (outer products) between the Hilbert spaces of each individual boson:
#
# - $\hat{a}_0 = \hat{a} \otimes \mathcal{I} \otimes \mathcal{I} \otimes \mathcal{I} \otimes \mathcal{I} \otimes \mathcal{I}$
# - $\hat{a}_1 = \mathcal{I} \otimes \hat{a} \otimes \mathcal{I} \otimes \mathcal{I} \otimes \mathcal{I} \otimes \mathcal{I}$
# - $\hat{a}_2 = \mathcal{I} \otimes \mathcal{I} \otimes \hat{a} \otimes \mathcal{I} \otimes \mathcal{I} \otimes \mathcal{I}$
# - $\hat{a}_3 = \mathcal{I} \otimes \mathcal{I} \otimes \mathcal{I} \otimes \hat{a} \otimes \mathcal{I} \otimes \mathcal{I}$
# - $\hat{a}_4 = \mathcal{I} \otimes \mathcal{I} \otimes \mathcal{I} \otimes \mathcal{I} \otimes \hat{a} \otimes \mathcal{I}$
# - $\hat{a}_5 = \mathcal{I} \otimes \mathcal{I} \otimes \mathcal{I} \otimes \mathcal{I} \otimes \mathcal{I} \otimes \hat{a}$
# %% [markdown]
# The next 2 cells create the annihilation operators for the first 2 bosons. The outer product will transform the matrices from size $L$ to size $L^\textrm{Nmat}$.
#
# For $L=5$ and $\textrm{Nmat}=6$ that is $15625$!!!
#
# Since we need 6 of them, this requires more than 10GB of RAM

# %%
a0 = tensor(
    [a, id, id, id, id, id]
)  # add .unit() if you want it normalized but it will take a long time to run!
print(a0)

# %% [markdown]
# You can see from the print line that the representation is sparse and only the non-zero elements are shown with the corresponding indices $(i,j)$ in the matrix representation.

# %%
a1 = tensor(
    [id, a, id, id, id, id]
)  # add .unit() if you want it normalized to unit norm
print(a1)

# %% [markdown]
# We can repeat the tensor product above for each boson by passing a new list for each boson.

# %%
# generically speaking, we construct the list of bosons and then take the outer product
product_list = [id] * Nmat  # only the identity repeated Nmat times
a_list = []  # this will contain a1...a6
for i in np.arange(0, Nmat):  # loop over all operators
    operator_list = product_list.copy()  # all elements are the identity operator
    operator_list[
        i
    ] = a  # the i^th element is now the annihilation operator for a single boson
    a_list.append(
        tensor(operator_list)
    )  # do the outer product, add .unit() to tensor if you want it normalized but it will take a long time to run


# %%
id_tensor = tensor(product_list)


# %%
print(a_list[0])

# %% [markdown]
# From the output above you can see the shape of the matrix and also the fact that the original individual Hilbert space have a smaller dimension $L \times L$

# %%
# this takes too long on Colaboratory
# a_list[0].norm()

# %% [markdown]
# The `norm` operation can run in parallel (if `qutip` is compiled with openMP support, and it uses half of the available cores) but it requires allocating all the memory needed for the operator (for `N=5` it allocates ~6Gb of memory).

# %%
# check that the operators are created as expected
a_list[0] == a0

# %% [markdown]
# Generate the above list of operators (plus the identity operator) using a single function defined in `qutip`: `enr_destroy` where the first argument is a list of the dimensions of each subsystem of a composite quantum system and the second argument is the highest number of excitations that are to be included in the state space.
# %%
new_a_list = enr_destroy([L] * Nmat, excitations=L * Nmat)
new_id = enr_identity([L] * Nmat, excitations=L * Nmat)
# %% [markdown]
# Check that the objects are the same

# %%
[np.allclose(x[0], x[1]) for x in zip(a_list, new_a_list)]

# %% [markdown]
# ### Cross-check
#
# Check if the procedure in `qutip` agrees with the one done in `numpy` (following the Tutorial by Mohammad and the PDF by Yuan in Mathematica)

# %%
annOp = np.array(np.diagflat(np.sqrt(np.linspace(1, L - 1, L - 1)), k=1))
with np.printoptions(
    precision=3, suppress=True, linewidth=120, threshold=100
):  # print array lines up to character 100 and floats using 3 digits
    print(annOp)


# %%
# this is the dense (full) matrix form of the annihilation operator in qutip (for one particle)
with np.printoptions(precision=3, suppress=True, linewidth=100, threshold=100):
    print(a.full())


# %%
np.allclose(annOp, a.full())


# %%
iden = np.identity(L)
with np.printoptions(
    precision=3, suppress=True, linewidth=120, threshold=100
):  # print array lines up to character 100 and floats using 3 digits
    print(iden)


# %%
bosonList = [annOp]
for bosons in range(0, Nmat - 1):
    bosonList.append(iden)
with np.printoptions(precision=3, suppress=True, linewidth=120, threshold=100):
    for i in bosonList:
        print(f"{i}\n")

# %% [markdown]
# **ATTENTION**
#
# Do not run the cell below unless $N < 5$.
# If higher it will run out of memory because it is using matrices that are 15625x15625 and it needs to store 6 of them in memory (>10GB)

# %%
# This for loop takes the appropriate Kronecker products for each boson.
for i in range(0, Nmat):
    for j in range(0, Nmat - 1):
        # For the nth boson, the nth Kronecker product is with the annihilation operator.
        if j == i - 1 and i != 0:
            bosonList[i] = np.kron(bosonList[i], annOp)
        # Else, the nth Kronecker product is with the identity matrix.
        else:
            bosonList[i] = np.kron(bosonList[i], iden)


# %%
[x.shape for x in bosonList]


# %%
np.allclose(bosonList[0], a_list[0].full())


# %%
np.allclose(bosonList[1], a_list[1].full())


# %%
np.allclose(bosonList[-1], a_list[-1].full())

# %% [markdown]
# The operators created with the tensor product in `qutip` are different from the ones created from the Kronecker product in `numpy` only because the latter are not normalized to unit norm. If we do not normalize in `qutip` then they are the same.
# %% [markdown]
# ## Position and momentum operators
#
# We can create these annihilation operators to be normalized to unit norm and can be used to construct the Hamiltonian.
#
# For the potential term of the Hamiltonian, it is somewhat convenient to use the position operators. Of course, formally, it is just a redefinition.
#
# First we construct the $\hat{x}$ and $\hat{p}$ operators for each boson in the list starting from the creation and annihilation operators.

# %%
# example position operator
x0 = a0.dag() + a0  # do not normalize because it takes too long .unit()
print(x0)


# %%
# example momentum operator
p0 = 1j * (a0.dag() - a0)  # normalization takes too long .unit()
print(p0)


# %%
x_list = []
p_list = []
for op in a_list:
    x_list.append(1 / np.sqrt(2) * (op.dag() - op))
    p_list.append(1 / np.sqrt(2) * 1j * (op.dag() - op))


# %%
type(np.sqrt(2))

# %% [markdown]
# ## Mini-BMN Bosonic Hamiltonian
# %% [markdown]
# The full Hamiltonian can be seen as two separate terms:
# - the quadratic part which represents the harmonic oscillator
# - the quartic part which represents the interaction potential
#
# The first can be easily written using the number operators $\hat{a}^{\dagger}\hat{a}$  for each boson (and subtracting the zero-point energy) and the second can be written using the position operators.
#
# We can also write everything with the annihilation and creation operators as shown in the notes by Masanori.

# %%
# the harmonic oscillator without potential for boson 0
H_osc0 = a0.dag() * a0 + 0.5
print(H_osc0)

# %% [markdown]
# This piece of the Hamiltonian is hermitean and has non-zero elements only on the diagonal.

# %%
H_osc1 = a1.dag() * a1 + 0.5
H_osc1


# %%
# this should be summed over all the bosons (Nmat)
H_osc = a_list[0].dag() * a_list[0] + 0.5
for i in np.arange(len(a_list) - 1):
    H_osc = H_osc + a_list[i + 1].dag() * a_list[i + 1] + 0.5


# %%
print(H_osc)


# %%
H_osc.eigenenergies(eigvals=5)  # select the lowest 5 only: it is faster


# %%
H_osc.eigenenergies()


# %%
eigv, eigk = H_osc.groundstate(sparse=True, tol=1e-06)


# %%
print(eigv)

# %% [markdown]
# ### Cross-check

# %%
# Create the simple quartic Hamiltonian.
H2MM = 0

for i in range(0, Nmat):
    # The @ symbol is a shorthand for matrix multiplication. It's equivalent to using np.matmul().
    H2MM = H2MM + (np.transpose(np.conjugate(bosonList[i])) @ bosonList[i])

H2MM = H2MM + 0.5 * Nmat * np.identity(L ** (Nmat))


# %%
np.identity(L ** (Nmat)) * 0.5 * 6


# %%
import sys

with np.printoptions(
    precision=5, suppress=False, linewidth=2000, threshold=sys.maxsize
):  # print array lines up to character 120 and floats using 3 digits
    print(H2MM)


# %%
np.allclose(H2MM, H_osc.full())

# %% [markdown]
# ## Quartic potential interaction
# %% [markdown]
# We then add the potential term of the Hamiltonian which includes the quartic interactions
# $$
# V_{4}=\frac{\lambda}{2}\left(x_{3}^{2} x_{4}^{2}+x_{3}^{2} x_{5}^{2}+x_{2}^{2} x_{4}^{2}+x_{2}^{2} x_{6}^{2}+x_{1}^{2} x_{5}^{2}+x_{1}^{2} x_{6}^{2}-2 x_{1} x_{3} x_{4} x_{6}-2 x_{1} x_{2} x_{4} x_{5}-2 x_{2} x_{3} x_{5} x_{6}\right)
# $$

# %%
# coupling lambda
coupling = 0.2
V = (
    0.5
    * coupling
    * (
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
)
print(V)


# %%
H = H_osc + V


# %%
print(H)


# %%
# import sys
# with np.printoptions(precision=5, suppress=False, linewidth=2000, threshold=sys.maxsize): # print array lines up to character 120 and floats using 3 digits
#   print(H.full())


# %%
eigs = H.eigenenergies(eigvals=10, sparse=True, tol=1e-06)


# %%
print(eigs)


# %%
eigv, eigk = H.groundstate(sparse=True, tol=1e-08)
print(eigv)

# %% [markdown]
# ## Try very large matrices
#
# With `qutip` we can go to large matrices, that is large cutoff $\Lambda$ (or $N$ here)
# %% [markdown]
# Create a function to construct the Hamiltonian and return the ground state energy for various values of $N$, $N_{\rm mat}$ and $g^2$, where $g^2$ is the 't Hooft coupling

# %%
L = 8  # this is the cutoff
Nmat = 6  # this is fixed for the bosonic BMN2 with SU(2)
g2 = 1  # this is the t'hooft coupling


# %%
def diagonalize_hamiltonian(N, Nmat, g2, num=1, sparse=True, tolerance=1e-06):
    """Build the bosonic BMN2 SU(2) hamiltonian with interaction strength g
  and find the ground state energy. Should use the sparse solutions when N is larger than 4"""
    if (not sparse) and (N > 4):
        print("Do not recommend dense eigensolvers unless you have very large memory!")
        return
    ### our basis operators are the annihilation and the identity
    a = destroy(N)
    id = identity(N)
    # generically speaking, we construct the list of bosons and then take the outer product
    product_list = [id] * Nmat  # only the identity repeated Nmat times
    a_list = []  # this will contain a1...a6
    for i in np.arange(0, Nmat):  # loop over all operators
        operator_list = product_list.copy()  # all elements are the identity operator
        operator_list[
            i
        ] = a  # the i^th element is now the annihilation operator for a single boson
        a_list.append(
            tensor(operator_list)
        )  # do the outer product, add .unit() to tensor if you want it normalized but it will take a long time to run
    x_list = []  # position operators
    for op in a_list:
        x_list.append(1 / np.sqrt(2) * (op.dag() - op))  # normalized as in the notes

    ### Harmonic oscillator
    # this should be summed over all the bosons (Nmat)
    H_osc = 0
    for i in np.arange(0, Nmat):
        H_osc = H_osc + a_list[i].dag() * a_list[i] + 0.5
    ### Quartic Interaction
    V = (
        0.5
        * g2
        * (
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
    )
    H = H_osc + V
    return H.eigenenergies(sparse=sparse, tol=tolerance, eigvals=num)


#  return H.groundstate(sparse=sparse, tol=tolerance)[0]


# %%
diagonalize_hamiltonian(L, Nmat, g2, sparse=True)


# %%
diagonalize_hamiltonian(4, 6, 0.000001, 10, sparse=False)

# %% [markdown]
# ### Get groundstate energy for many cutoff values at fixed coupling and tolerance

# %%
import time

start_time = time.time()
g2 = 0.2
gs = []
for cutoff in np.arange(2, 12):
    gs.append(diagonalize_hamiltonian(cutoff, Nmat, g2, tolerance=1e-08))
print(gs)
gs = np.array(gs).flatten()
end_time = time.time()
runtime = end_time - start_time
print("Program runtime: ", runtime)


# %%
import matplotlib.pyplot as plt

plt.plot(np.arange(2, 12), gs, "bo", label=r"Sparse diag. $\lambda$=0.2")
plt.title(r"Groundstate energy with $c=0$")
plt.xlabel(r"$\Lambda$")
plt.ylabel(r"$E_0$")
plt.legend(loc="best")


# %%
print("lambda,energy")
for a, b in zip(np.arange(2, 12), gs):
    print(f"{a:2},{b-3:.8f}")


# %%
plt.plot(np.arange(2, 12), gs - 3, "bo", label=r"Sparse diag. $\lambda$=0.2")
plt.title(r"Groundstate energy with $c=0$")
plt.xlabel(r"$\Lambda$")
plt.ylabel(r"$E_0-3$")
# plt.yscale('log')
plt.ylim([0.132, 0.135])
plt.xlim([2.5, 12])
plt.grid()
plt.legend(loc="best")


# %% [markdown]
# ## The gauge generators
# Write the gauge generator as $$\hat{G}_{\alpha} = i \sum_{\beta, \gamma, I} f_{\alpha\beta\gamma} \hat{a}^{\dagger\beta}_I \hat{a}^{\gamma}_I $$

# %%
g_list = [0] * 3
g_list[0] = 1j * (
    a_list[1].dag() * a_list[2]
    - a_list[2].dag() * a_list[1]
    + a_list[4].dag() * a_list[5]
    - a_list[5].dag() * a_list[4]
)
g_list[1] = 1j * (
    a_list[2].dag() * a_list[0]
    - a_list[0].dag() * a_list[2]
    + a_list[5].dag() * a_list[3]
    - a_list[3].dag() * a_list[5]
)
g_list[2] = 1j * (
    a_list[0].dag() * a_list[1]
    - a_list[1].dag() * a_list[0]
    + a_list[3].dag() * a_list[3]
    - a_list[4].dag() * a_list[3]
)
# %% [markdown]
# Get the ground state eigenfunction and measure the expectation value of the sum of the generators squared $$\sum_{\alpha} \langle E_0 | \hat{G}^2_{\alpha} | E_0 \rangle$$

# %%
g_violation = sum([expect(g * g, eigk) for g in g_list])
# %% [markdown]
# Find the ground state for different cutoffs at fixed coupling constant

# %%
def gauge_violation(N, Nmat, g2, sparse=True, tolerance=1e-06):
    """Build the bosonic BMN2 SU(2) hamiltonian with interaction strength g
  and find the ground state energy. Should use the sparse solutions when N is larger than 4"""
    if (not sparse) and (N > 4):
        print("Do not recommend dense eigensolvers unless you have very large memory!")
        return
    ### our basis operators are the annihilation and the identity
    a = destroy(N)
    id = identity(N)
    # generically speaking, we construct the list of bosons and then take the outer product
    product_list = [id] * Nmat  # only the identity repeated Nmat times
    a_list = []  # this will contain a1...a6
    for i in np.arange(0, Nmat):  # loop over all operators
        operator_list = product_list.copy()  # all elements are the identity operator
        operator_list[
            i
        ] = a  # the i^th element is now the annihilation operator for a single boson
        a_list.append(
            tensor(operator_list)
        )  # do the outer product, add .unit() to tensor if you want it normalized but it will take a long time to run
    x_list = []  # position operators
    for op in a_list:
        x_list.append(1 / np.sqrt(2) * (op.dag() - op))  # normalized as in the notes

    ### Harmonic oscillator
    # this should be summed over all the bosons (Nmat)
    H_osc = 0
    for i in np.arange(0, Nmat):
        H_osc = H_osc + a_list[i].dag() * a_list[i] + 0.5
    ### Quartic Interaction
    V = (
        0.5
        * g2
        * (
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
    )
    H = H_osc + V
    eigv, eigk = H.groundstate(sparse=sparse, tol=tolerance)
    # define the genrator list for SU(2)
    g_list = [0] * 3
    g_list[0] = 1j * (
        a_list[1].dag() * a_list[2]
        - a_list[2].dag() * a_list[1]
        + a_list[4].dag() * a_list[5]
        - a_list[5].dag() * a_list[4]
    )
    g_list[1] = 1j * (
        a_list[2].dag() * a_list[0]
        - a_list[0].dag() * a_list[2]
        + a_list[5].dag() * a_list[3]
        - a_list[3].dag() * a_list[5]
    )
    g_list[2] = 1j * (
        a_list[0].dag() * a_list[1]
        - a_list[1].dag() * a_list[0]
        + a_list[3].dag() * a_list[4]
        - a_list[4].dag() * a_list[3]
    )
    print(f"Eigenvalue of the ground state at cutoff {N}: {eigv}")
    return sum([expect(g * g, eigk) for g in g_list])


# %%
import time

start_time = time.time()
g2 = 1.0
gs = []
for cutoff in np.arange(3, 13):
    gs.append(gauge_violation(cutoff, Nmat, g2, tolerance=1e-08))
print(gs)
gs = np.array(gs).flatten().real
end_time = time.time()
runtime = end_time - start_time
print("Program runtime: ", runtime)
# %%
plt.plot(np.arange(3, 13), gs, "bo", label=r"Sparse diag. $\lambda$=1.0")
plt.yscale("log")
plt.grid()
plt.title(r"Gauge singlet violation of the ground state")
plt.xlabel(r"$\Lambda$")
plt.ylabel(r"$\sum\langle E_0 | \hat{G}^2_\alpha| E_0 \rangle$")
plt.legend(loc="best")

# %% [markdown]
# ## Plotting data saved on disk

# %% [markdown]
# ### Ground state energy
# %%
import pandas as pd
import numpy as np

l = "0.5"
ls = l.replace(".", "")
data = pd.read_csv(f"l{ls}_gs.csv", header=0, dtype={"Lambda": int, "Energy": float})
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
data.plot(x="Lambda", y="Energy", marker="o", label=rf"$\lambda$={l}", ax=ax)
ax.set_ylabel(r"$E_0$", rotation=90)
ax.set_xlabel(r"$\Lambda$")

# %%
data.plot(x="Lambda", y="Energy", marker="o", label=rf"$\lambda$={l}", loglog=True)
# %%
# plot first and every 2: odd cutoffs
data[::2].plot(
    x="Lambda", y="Energy", marker="o", label=rf"$\lambda$={l} odd", logy=True
)
# %%
fig, ax = plt.subplots()
data[::2].plot(
    x="Lambda", y="Energy", marker="o", label=rf"$\lambda$={l} odd", logy=True, ax=ax
)
data[1::2].plot(
    x="Lambda", y="Energy", marker="o", label=rf"$\lambda$={l} even", logy=True, ax=ax
)
ax.set_ylabel(r"$E_0$", rotation=90)
ax.set_xlabel(r"$\Lambda$")
ax.legend(loc="lower right")
# %%
# different between successive cutoffs: goes down exponentially
data.diff().abs().plot(y="Energy", marker="o", label=rf"$\lambda$={l} DIFF", logy=True)
# %%
fig, ax = plt.subplots()
data.diff()[::2].abs().plot(
    y="Energy", marker="o", label=rf"$\lambda$={l} odd", logy=True, ax=ax
)
data.diff()[1::2].abs().plot(
    y="Energy", marker="o", label=rf"$\lambda$={l} even", logy=True, ax=ax
)
ax.set_ylabel(r"$E_0^{diff}$", rotation=90)
ax.set_xlabel(r"$\Lambda-3$")
ax.legend(loc="upper right")
plt.savefig(f"l{ls}_gs.pdf")

# %% [markdown]
# ### Gauge Violation term
# %%
import pandas as pd
import numpy as np

l = "0.2"
ls = l.replace(".", "")
data = pd.read_csv(
    f"l{ls}_gv.csv", header=0, dtype={"Lambda": int, "GaugeViolation": float}
)
#
# %%
data.plot(
    x="Lambda", y="GaugeViolation", marker="o", label=fr"$\lambda$={l}", logy=True
)
# %%
fig, ax = plt.subplots()
data[::2].plot(
    x="Lambda",
    y="GaugeViolation",
    marker="o",
    label=fr"$\lambda$={l} odd",
    logy=True,
    ax=ax,
)
data[1::2].plot(
    x="Lambda",
    y="GaugeViolation",
    marker="o",
    label=fr"$\lambda$={l} even",
    logy=True,
    ax=ax,
)
ax.set_ylabel(r"$\langle G^2 \rangle$", rotation=90)
ax.set_xlabel(r"$\Lambda$")
plt.savefig(f"l{ls}_gv.pdf")
# %%

# %%
