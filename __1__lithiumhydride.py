""" Example of how to use pyscf and the toolkit to export lithium hydride systems. """

import pyscf
import openfermion
import geometries
import toolkit

# BUILD MOLECULE
molecule = pyscf.gto.M(
    atom=geometries.LiH(3.0),
    charge=0,
    spin=0,
    basis="sto-3g",
    symmetry=True,
)

# RUN PYSCF
scf = pyscf.scf.RHF(molecule)
scf.run()

# SELECT ORBITALS
freeze_orbitals = [0]
active_orbitals = [1,2,5]

# FETCH QUANTUM NUMBERS
_, n, _, ηα, ηβ = toolkit.quantum_numbers(
    molecule,
    freeze_orbitals=freeze_orbitals,
    active_orbitals=active_orbitals,
)

# CONSTRUCT FERMI OPERATOR
fermiop = toolkit.fermiop_from_molecule(
    molecule, scf.mo_coeff,
    freeze_orbitals=freeze_orbitals,
    active_orbitals=active_orbitals,
)

# CONSTRUCT REFERENCE STATE
reference = toolkit.referencevector(n, ηα, ηβ)

# CONSTRUCT CODE
code = toolkit.taperedcode(n, ηα, ηβ, codefn=openfermion.parity_code, taper=2)

# TRANSFORM OPERATOR AND REFERENCE STATE
qubitop = toolkit.encode_operator(fermiop, code)
ket = toolkit.encode_vector(reference, code)

# SAVE THEM
toolkit.save_system("systems/lithiumhydride.npz", qubitop, ket)
