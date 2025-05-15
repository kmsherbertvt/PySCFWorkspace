""" Example of how to use pyscf and the toolkit to export Hubbard models. """

import numpy
import pyscf
import openfermion
import toolkit

# SELECT PARAMETERS
L = 6           # Number of sites.
M = 3           # Number of α electrons.
N = 6           # Number of electrons.

t = 1.0         # Hopping energy.
U = 4.0         # Coulomb interaction.
μ = 0.5         # Chemical potential.

##########################################################################################

# CONSTRUCT FERMI OPERATOR

##########################################################################################

# BUILD OUT INTEGRAL ARRAYS
nuc = 0
nuc += U*L/4                # PARTICLE-HOLE TRANSFORMATION

obi = numpy.zeros((L,L))
eri = numpy.zeros((L,L,L,L))
for i in range(L):
    obi[i,i] = -μ
    obi[i,i] += -U/2        # PARTICLE-HOLE TRANSFORMATION

    j = (i+1)%L             # PERIODIC BOUNDARY CONDITIONS
    obi[i,j] = obi[j,i] = -t

    eri[i,i,i,i] = U

# TEMP for verifying eigenspectrum
fermiop_ = toolkit.fermiop_from_integrals(nuc, obi, eri)
fermiop_ = openfermion.normal_ordered(fermiop_)

fermiop__ = openfermion.hamiltonians.fermi_hubbard(
    L,          # Length of chain.
    1,          # Length in y-dimension, if we were doing 2d.
    t, U, μ,    # All the numerical parameters.
    0.0,        # One more numerical parameter, `h`, describing a magnetic field.
    True,       # Use periodic boundary conditions.
    False,      # Use a "spinless" Hubbard model? I never did bother to learn this.
    True,       # Use particle-hole symmetry.
)
fermiop__ = openfermion.normal_ordered(fermiop__)

""" If we want to use the "atomic orbital" basis, we can simply do:

fermiop = toolkit.fermiop_from_integrals(nuc, obi, eri)

    This isn't a bad choice for the Hubbard model,
        since the Hamiltonian is explicitly sparse in this basis.
    Depending on who you ask, it's the standard choice.
    In fact, we can get the same Hamiltonian (up to normal ordering)
        with a built-in openfermion function:

fermiop = openfermion.hamiltonians.fermi_hubbard(
    L,          # Length of chain.
    1,          # Length in y-dimension, if we were doing 2d.
    t, U, μ,    # All the numerical parameters.
    0.0,        # One more numerical parameter, `h`, describing a magnetic field.
    True,       # Use periodic boundary conditions.
    False,      # Use a "spinless" Hubbard model? I never did bother to learn this.
    True,       # Use particle-hole symmetry.
)

    But, suppose you want the mean-field reference state.
    Then we need to use pyscf to help us solve Hartree-Fock.

"""

##########################################################################################

# CONSTRUCT A FAUX MOLECULE TO RUN PYSCF
molecule = pyscf.gto.M()
molecule.nelectron = N
molecule.spin = M - (N-M)
molecule.incore_anyway = True           # Guarantees SCF uses custom `_eri`.

scf = pyscf.scf.RHF(molecule)
scf.get_hcore = lambda *args: obi
scf.get_ovlp = lambda *args: numpy.eye(L)
scf._eri = pyscf.ao2mo.restore(8, eri, L)
scf.init_guess = "hcore"
scf.max_cycles = 1000
scf.run()

""" Careful here. Hartree-Fock doesn't always converge.
    E.g. switching to (L,M,N)=(4,2,4), something breaks.
    I haven't tested thoroughly to see WHAT breaks,
        or if the intermediate result it gives is "good enough",
        so be careful!
"""

# TRANSFORM TO MOLECULAR ORBITAL BASIS
mo_obi = openfermion.general_basis_change(obi, scf.mo_coeff, (1,0))
mo_eri = openfermion.general_basis_change(eri, scf.mo_coeff, (1,1,0,0))
mo_eri = mo_eri.transpose(0,2,3,1)  # Switch to openfermion tensor convention.

# CONSTRUCT FERMI OPERATOR
fermiop = toolkit.fermiop_from_integrals(nuc, mo_obi, mo_eri)
fermiop = openfermion.normal_ordered(fermiop)

##########################################################################################

# CONSTRUCT REFERENCE STATE
reference = toolkit.referencevector(2*L, M, N-M)

# CONSTRUCT CODE
code = toolkit.taperedcode(2*L, M, N-M, codefn=openfermion.parity_code, taper=2)

# TRANSFORM OPERATOR AND REFERENCE STATE
qubitop = toolkit.encode_operator(fermiop, code)
ket = toolkit.encode_vector(reference, code)

# SAVE THEM
toolkit.save_system("systems/hubbard.npz", qubitop, ket)
