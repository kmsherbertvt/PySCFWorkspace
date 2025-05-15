""" Construct portable matrix representations of a molecule's qubit operator.

This module provides the function `constructmatrix`,
    which takes a molecule (`openfermion.MolecularData`) and a mapping (as a string),
    to produce a matrix (`numpy.ndarray`) and save it as a file.

The "mapping" string must be a key in the `CONSTRUCT_BINARY_CODE` dict,
    defined in the `mappings` module.

"""

MATRIX_DIRECTORY = "./matrix"
SECTOR_DIRECTORY = "./sector"
PROBLEM_DIRECTORY = "./problem"

import numpy
import cirq
import openfermion
import json

import molecules
import mappings

def matrixoperator(qubitop, n=None, bigendian=False):
    """ Construct the matrix representation of a given qubit operator.

    Parameters
    ----------
    qubitop (openfermion.QubitOperator): the operator to transform
    n (int): total number of qubits

        This is inferred automatically from `qubitop` by default,
            but you may need to specify it explicitly if
                `qubitop` acts trivially on the highest-index qubits.
        Of course, in that case,
            you should ask yourself if you *really* need those qubits...
            (But there are good reasons, sometimes!)

    bigendian (bool):determines qubit ordering

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
    matrix (numpy.ndarray) the qubit operator, as a matrix

    """
    # INFER THE NUMBER OF QUBITS
    if n is None:
        n = 0
        for term in qubitop.terms:
            for (q,σ) in term:
                n = max(q,n)

    # CONSTRUCT THE QUBIT ARRAY
    qbits = cirq.LineQubit.range(n)
    if bigendian: qbits = reversed(qbits)

    # CONSTRUCT MATRIX
    pauliop = openfermion.qubit_operator_to_pauli_sum(qubitop)
    return pauliop.matrix(qbits)

def qubitoperator(fermiop, mapping, sector):
    """ Convert a fermi operator into a pauli represenation, under a given mapping.

    NOTE: For convenience only.
    The `code` from `CONSTRUCT_BINARY_CODE` is a useful object to have,
        so I wouldn't bother calling this in most workflows.

    Parameters
    ----------
    fermiop (openfermion.FermionOperator): the operator to convert
    mapping (string): a key of the `mappings.CONSTRUCT_BINARY_CODE` dict
    sector (mappings.Sector): argument for the `mappings.CONSTRUCT_BINARY_CODE` value

        ie. construct the code via `mappings.CONSTRUCT_BINARY_CODE[mapping](sector)`

    """
    code = mappings.CONSTRUCT_BINARY_CODE[mapping](sector)
    return openfermion.binary_code_transform(fermiop, code)






def make_molecule_matrix(
    molecule, mapping,
    bigendian=False, save=True, dir=MATRIX_DIRECTORY,
):
    if molecule.hf_energy is None:
        molecule = molecules.fill_electronicstructure(molecule)

    sector = mappings.Sector(
        n=molecule.n_qubits,
        Ne=molecule.n_electrons,
        Nα=molecule.get_n_alpha_electrons(),
    )

    code = mappings.CONSTRUCT_BINARY_CODE[mapping](sector)

    interop = molecule.get_molecular_hamiltonian()
    fermiop = openfermion.get_fermion_operator(interop)

    name = save if isinstance(save, str) else molecule.name if save else None
    return make_matrix(fermiop, mapping, sector, bigendian=bigendian, name=name, dir=dir)

def make_matrix(
    fermiop, mapping, sector,
    bigendian=False, name=None, dir=MATRIX_DIRECTORY,
):
    code = mappings.CONSTRUCT_BINARY_CODE[mapping](sector)
    qubitop = openfermion.binary_code_transform(fermiop, code)
    matrix = matrixoperator(qubitop, n=code.n_qubits, bigendian=bigendian)

    if name is not None:
        filename = f"{name}_{mapping}"
        if bigendian: filename = f"{filename}_2=01"
        numpy.save(f"{dir}/{filename}.npy", matrix)

    return matrix





def creator(z):
    """ The fermionic operator preparing a particular orbital configuration from vacuum.

    Parameters
    ----------
    z (str): a string of "0" and "1"

    Returns
    -------
    openfermion.FermionOperator: the operator A such that A[z] |⟩ = |z⟩

    """
    ops = []
    for q, bit in enumerate(z):
        if bit == "1":
            ops.append((q,1))
    ops = list(reversed(ops))
    return openfermion.FermionOperator(ops)

def selector(z):
    """ The fermionic operator selecting a particular orbital configuration.

    Parameters
    ----------
    z (str): a string of "0" and "1"

    Returns
    -------
    openfermion.FermionOperator: the operator A such that A[z] |i⟩ = δ[i,z] |i⟩

    """
    op = 1
    for q, bit in enumerate(z):
        n = openfermion.hamiltonians.number_operator(len(z), q)
        op *= (n if bit == "1" else (1-n))
    return openfermion.normal_ordered(op)

def reference_configuration(sector):
    """ The standard configuration which fills up orbitals from left to right.

    Known as the "Hartree-Fock" state when orbitals are selected to minimize mean-field.

    Parameters
    ----------
    sector (mappings.Sector): encapsulates number of orbitals and electrons

    Returns
    -------
    str: a string of "0" and "1"

    """
    z = ""
    for i in range(sector.n >> 1):              # ITERATE OVER SPATIAL ORBITALS
        z += "1" if i < sector.Nα else "0"
        z += "1" if i < sector.Nβ else "0"
    return z

def antiferro_configuration(sector):
    """ The configuration which alternated between spin-up and spin-down.

    This represents an solution to the Hubbard model in the infinite coupling limit.

    Parameters
    ----------
    sector (mappings.Sector): encapsulates number of orbitals

    Returns
    -------
    str: a string of "0" and "1"

    """
    assert sector.n & 1 == 0            # ONLY VALID FOR EVEN NUMBER OF SPATIAL ORBITALS
    assert sector.Ne == sector.n >> 1   #   HALF ARE FILLED
    assert sector.Nα == sector.Ne >> 1  #   AND α LOSES THE ODD ELECTRON

    z = ""
    for i in range(sector.n >> 1):
        z += "01" if i & 1 == 0 else "10"
    return z

def N__operator(sector):
    return openfermion.hamiltonians.number_operator(sector.n)

def S2_operator(sector):
    return openfermion.hamiltonians.s_squared_operator(sector.n >> 2)

def Sz_operator(sector):
    return openfermion.hamiltonians.sz_operator(sector.n >> 1)

def make_sector_matrices(
    sector, mapping, bigendian=False, save=True,
    ref=True, N=True, S2=True, Sz=True,
    dir=SECTOR_DIRECTORY,
):
    ret = []
    if ref:
        z = reference_configuration(sector)
        ret.append(make_reference(z, sector, mapping, bigendian=bigendian, name="REF"))
    if N:
        name = f"N__{sector}" if save else None
        ret.append(make_matrix(
            N__operator(sector), mapping, sector, bigendian=bigendian,
            name=name, dir=dir,
        ))
    if S2:
        name = f"S2_{sector}" if save else None
        ret.append(make_matrix(
            S2_operator(sector), mapping, sector, bigendian=bigendian,
            name=name, dir=dir,
        ))
    if Sz:
        name = f"Sz_{sector}" if save else None
        ret.append(make_matrix(
            Sz_operator(sector), mapping, sector, bigendian=bigendian,
            name=name, dir=dir,
        ))
    return tuple(ret)



def make_molecule_reference(
    molecule, mapping,
    bigendian=False, save=True, dir=SECTOR_DIRECTORY,
):
    sector = mappings.Sector(
        n=molecule.n_qubits,
        Ne=molecule.n_electrons,
        Nα=molecule.get_n_alpha_electrons(),
    )
    z = reference_configuration(sector)
    name = "REF" if save else None
    return make_reference(z, sector, mapping, bigendian=bigendian, name=name, dir=dir)

def make_reference(
    z, sector, mapping,
    bigendian=False, name=None, dir=SECTOR_DIRECTORY,
):
    fermiop = selector(z)
    code = mappings.CONSTRUCT_BINARY_CODE[mapping](sector)

    qubitop = openfermion.binary_code_transform(fermiop, code)
    matrix = matrixoperator(qubitop, n=code.n_qubits, bigendian=bigendian)
    state = numpy.diag(matrix)

    # SAVE STATE AS A .npy FILE
    if name is not None:
        filename = f"{name}_{sector}_{mapping}"
        if bigendian: filename = f"{filename}_2=01"
        numpy.save(f"{dir}/{filename}.npy", state)

    return state



def get_reference_energy(z, obs_op):
    ref_op = creator(z)
    totalop = openfermion.hermitian_conjugated(ref_op) * obs_op * ref_op
    return openfermion.normal_ordered(totalop).constant.real





def as_xz(actions, n):
    """ Convert an openfermion tuple-of-tuples of Pauli actions
        to integers in symplectic form. """
    x = 0
    z = 0

    for (q, σ) in actions:
        mask = 1<<(n-q-1)
        if   σ=="X":
            x ^= mask
        elif σ=="Y":
            x ^= mask
            z ^= mask
        elif σ=="Z":
            z ^= mask
        else:
            raise ValueError(f"Invalid action {σ}")
    return x, z

def minqubits(qubitop):
    """ Compute the minimum number of qubits a qubitop acts on. """
    n = 0
    for actions in qubitop.terms:
        for (q, pauli) in actions:
            n = max(1+q, n)
    return n

def xzclists(qubitop):
    """ Transform a qubitop into serializable lists of
        x and z integers and associated coeffecients.
    """
    xs = []
    zs = []
    cs = []
    n = minqubits(qubitop)
    for actions, c in qubitop.terms.items():
        x, z = as_xz(actions, n)
        xs.append(x)
        zs.append(z)
        cs.append(c)
    return xs, zs, cs

def write_qubitop(qubitop, filename):
    """ Save a qubitop to an npz file with lists of
        x and z integers and associated coeffecients.
    """
    x,z,c = xzclists(qubitop)
    numpy.savez(filename, x=x, z=z, c=c)
    return {"x": x, "z": z, "c": c}







def configurationindex(config, code, bigendian=False):
    fermiop = selector(config)
    qubitop = openfermion.binary_code_transform(fermiop, code)

    ket = numpy.zeros(code.n_qubits, dtype=int)
    for actions, c in qubitop.terms.items():
        if not len(actions) == 1: continue
        q, σ = actions[0]
        assert σ == "Z"
        ket[q] = 0 if c > 0 else 1
    return int("".join(str(b) for b in (reversed(ket) if bigendian else ket)), 2)

def make_molecule_problem(
    molecule, mapping,
    bigendian=False, save=True, dir=PROBLEM_DIRECTORY,
):
    """ Compactly bundle a molecular Hamiltonian in a given mapping,
        alongisde the number of qubits and the standard reference state.

        The data format is a dict with fields:
        - `X`, `Z`: vectors of ints identifying Pauli operators in symplectic notation
        - `c`: vector of floats giving coefficients for each Pauli
        - `n`: number of qubits
        - `z`: integer index of the standard reference state (e.g. |1..10..0⟩)
    """
    if molecule.hf_energy is None:
        molecule = molecules.fill_electronicstructure(molecule)

    sector = mappings.Sector(
        n=molecule.n_qubits,
        Ne=molecule.n_electrons,
        Nα=molecule.get_n_alpha_electrons(),
    )

    code = mappings.CONSTRUCT_BINARY_CODE[mapping](sector)
    n = code.n_qubits

    # BUILD OUT THE COMPACT HAMILTONIAN
    interop = molecule.get_molecular_hamiltonian()
    fermiop = openfermion.get_fermion_operator(interop)
    qubitop = openfermion.binary_code_transform(fermiop, code)
    X, Z, c = xzclists(qubitop)

    # IDENTIFY THE STANDARD REFERENCE STATE
    config = reference_configuration(sector)
    z = configurationindex(config, code, bigendian=bigendian)
    problem = {"X": X, "Z": Z, "c": c, "n": n, "z": z}

    # SAVE STATE AS A .npy FILE
    name = save if isinstance(save, str) else molecule.name if save else None
    if name is not None:
        filename = f"{name}_{mapping}"
        if bigendian: filename = f"{filename}_2=01"
        numpy.savez(f"{dir}/{filename}.npy", **problem)

    return problem
