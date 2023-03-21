""" Construct portable matrix representations of a molecule's qubit operator.

This module provides the function `constructmatrix`,
    which takes a molecule (`openfermion.MolecularData`) and a mapping (as a string),
    to produce a matrix (`numpy.ndarray`) and save it as a file.

The "mapping" string must be a key in the `CONSTRUCT_BINARY_CODE` dict,
    defined in the `mappings` module.

"""

MATRIX_DIRECTORY = "./matrix"

import openfermion
import numpy

import molecules
import mappings

def matrixoperator(molecule, mapping, save=True, bigendian=False):
    """ Construct the matrix representation of a molecular Hamiltonian.

    Parameters
    ----------
    molecule (openfermion.MolecularData)
    mapping (str): a key of the `mappings.CONSTRUCT_BINARY_CODE` dict
    save (bool or str): whether to save the matrix to a .npy file

        If `save` is True, the .npy file is saved to the path in `MATRIX_DIRECTORY`,
            with the name given by `molecule.name`,
            which `openfermion` has already carefully selected.

        If `save` is a string, it is used directly as both path and name.

    bigendian (bool): determines qubit ordering

        The default qubit ordering (`bigendian=False`) associates qubit i
            with the 2ⁱ place of the binary expansion of each matrix index.
        This could get confusing, because we like to place qubit 0 on the left,
            but we generally write the 2⁰ place on the right of a binary expansion.
        Thus, the option is provided to reverse the ordering,
            which just amounts to pretending the binary expansion
            is written in bigendian convention.
        But, the default ordering is pretty standard by now
            so I recommend just getting used to it.

    Returns
    -------
    matrix (numpy.ndarray)

    """
    qubitop = qubitoperator(molecule, mapping)

    # CONSTRUCT MATRIX
    pauliop = openfermion.qubit_operator_to_pauli_sum(qubitop)
    qbits = reversed(pauliop.qubits) if bigendian else pauliop.qubits
    if bigendian: mapping = f"{mapping}_2=01"     # UPDATE LABEL FOR FILENAME
    matrix = pauliop.matrix(qbits)

    # SAVE MATRIX AS A .npy FILE
    if isinstance(save, str): numpy.save(save, matrix)
    elif save: numpy.save(f"{MATRIX_DIRECTORY}/{molecule.name}_{mapping}.npy", matrix)

    return matrix

def interactionoperator(molecule):
    """ Construct a molecular Hamiltonian as an interaction operator.

    This data structure makes explicit the tensors h0, h1, h2 in:

        H = h0 + ∑ h1[p,q] a'[p] a[q] + ∑ h2[p,q,r,s] a'[p] a'[q] a[r] a[s]

    Notice the physicist's convention in the two-body terms.

    Parameters
    ----------
    molecule (openfermion.MolecularData)

    Returns
    -------
    interop (openfermon.InteractionOperator)


    """
    if molecule.hf_energy is None:
        molecule = molecules.fill_electronicstructure(molecule)
    return molecule.get_molecular_hamiltonian()

def fermioperator(molecule):
    """ Construct a molecular Hamiltonian as a ferionic operator.

    Parameters
    ----------
    molecule (openfermion.MolecularData)

    Returns
    -------
    fermiop (openfermon.FermionOperator)

    """
    interop = interactionoperator(molecule)
    return openfermion.get_fermion_operator(interop)

def qubitoperator(molecule, mapping):
    """ Map a molecular Hamiltonian to a qubit operator.

    Parameters
    ----------
    molecule (openfermion.MolecularData)
    mapping (str): a key of the `CONSTRUCT_BINARY_CODE` dict

    Returns
    -------
    qubitop (openfermon.QubitOperator)

    """
    fermiop = fermioperator(molecule)
    code = mappings.CONSTRUCT_BINARY_CODE[mapping](molecule)
    return openfermion.binary_code_transform(fermiop, code)