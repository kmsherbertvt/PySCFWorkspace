# PySCFWorkspace

This repository provides the tools to generate molecular Hamiltonians suitable for quantum computing algorithms.

Similar modules exist scattered about other Virginia Tech code-bases, eg.:
- https://github.com/mayhallgroup/adapt-vqe/blob/master/src/pyscf_helper.py
- https://github.com/mayhallgroup/ctrlq/blob/master/mole/molecule.py

But I recommend this one, because:
- It is a standalone repository, designed to produce portable data files.
- I'm using openfermion's "Binary Code" interface for generating qubit mappings,
   which is a bit more flexible than the others, I think.
- I think it's a simpler yet more extensible interface, at least for my needs.
- Well...I comment my code. ^_^

## What can I do with this code?

So far, the following features are available:
- Design molecules with custom geometries with a given basis, charge, and spin.
- Convenient constructions for typical H₂ and arbitrary-length H-chains.
- Generate the `openfermion.InteractionOperator` and `openfermion.FermionOperator` corresponding to a given molecule.
- Implement qubit mappings for the "standard" Jordan-Wigner, Bravyi-Kitaev, and Parity mappings,
   *as well as* Z₂ symmetry reductions (particle conservation and spin conservation) for each mapping.
- Generate the `openfermion.QubitOperator` and dense matrix `numpy.ndarray` corresponding to a given molecule and qubit mapping.
- Save the dense matrix operator to a portable `.npy` file, easily loaded in other workflows.

The following features are currently in development:
- Standardized serialization of the `openfermion` data structures, so that they are as portable as matrices.
- Interface for freezing orbitals within a basis (eg. remove core orbitals from quantum computation).
  (This is essential, but I don't really know how to do it yet.)
- Convenient constructions for additional model molecules (LiH, H2O, etc.).
  (Should be trivial, once I understand orbital freezing.)

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
The following example generates a dense 16x16 (4 qubit) matrix for H2 with a 1.5 Ang bond separation,
   using the Jordan-Wigner mapping.
```
> cd PySCFWorkspace
> conda activate pyscf
> ipython
:
: import molecules
: d = 1.5 # Ang
: geometry = [('H', (0,0,0)), ('H', (0,0,d))]
: molecule = molecules.custom(geometry, label="1.5")
:
: import operators
: mapping = "JW"
: matrix = operators.matrixoperator(molecule, mapping)
```
Please study the docstrings in `mappings.py`, `molecules.py`, and `operators.py` for full documentation.

In particular, find the variable `CONSTRUCT_BINARY_CODE` in `mappings.py` to see available mappings,
   and do not hesitate to add your own.

