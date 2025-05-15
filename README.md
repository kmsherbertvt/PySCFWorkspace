# PySCFWorkspace

This repository provides the tools to generate molecular Hamiltonians suitable for quantum computing algorithms.

Similar modules exist scattered about other Virginia Tech code-bases, eg.:
- https://github.com/mayhallgroup/adapt-vqe/blob/master/src/pyscf_helper.py
- https://github.com/mayhallgroup/ctrlq/blob/master/mole/molecule.py

But I recommend this one, because:
- It is a standalone repository, designed to produce portable data files.
- I'm using openfermion's "Binary Code" interface for generating qubit mappings,
   which is a bit more flexible than the others, I think.
- Well...I comment my code. ^_^

## What can I do with this code?

The following features are available:
- Convenient constructions for typical molecular geometries encountered in quantum computing literature,
  including H₂, arbitrary-length H-chains, LiH, and others.
- Thorough tutorials demonstrating how to solve Hartree-Fock for these molecules in pyscf.
- Generate the `openfermion.FermionOperator` corresponding to a given molecule with a given active space.
- Implement Z₂ symmetry reductions (particle conservation and spin conservation) for any standard qubit mapping.
- Generate the `openfermion.QubitOperator` and Hartree-Fock reference state corresponding to a given molecule and qubit mapping.
- Save the qubit operator and reference state to a portable and compact `.npz` file, easily loaded in other workflows.

## Installation and Usage

This repository is not meant to be used as a package,
    but as an independent workspace where you generate the files you need in other projects.
As such, installation and usage are flexible.

The following commands will work, but may be overkill if you aren't using a Mac M1 processor.
```
> git clone https://github.com/kmsherbertvt/PySCFWorkspace.git
> cd PySCFWorkspace
> bash install_dependenices_M1.sh
```
Please see the file `install_dependencies_M1.sh` for more thorough installation instructions.

Recommended usage is through the `ipython` shell.
Please see the example scripts (starting with `__x__`) for more thorough instruction,
   but here is a minimalist example generating the portable .npz file for molecular hydrogen,
   with an experimental equilibrium geometry, the Jordan-Wigner mapping, .
```
> cd PySCFWorkspace
> conda activate pyscf
> ipython
:
: import pyscf, openfermion
: import geometries, toolkit
:
: molecule = pyscf.gto.M(atom=geometries.H2(), charge=0, spin=0, basis="sto-3g")
: scf = pyscf.scf.RHF(molecule)
: scf.run()
: fermiop = toolkit.fermiop_from_molecule(molecule, scf.mo_coeff)
:
: _, n, _, ηα, ηβ = toolkit.quantum_numbers(molecule)
: reference = toolkit.referencevector(n, ηα, ηβ)
: code = toolkit.taperedcode(n, ηα, ηβ)
:
: qubitop = toolkit.encode_operator(fermiop, code)
: ket = toolkit.encode_vector(reference, code)
: toolkit.save_system("systems/README.npz", qubitop, ket)
```
Please study the docstrings in `toolkit.py` and `geometries.py` for full documentation.
