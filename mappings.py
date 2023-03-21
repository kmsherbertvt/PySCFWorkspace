""" Collect various qubit mappings into one extensible interface.

This module provides the `CONSTRUCT_BINARY_CODE` dict,
    used by `operators.constructmatrix` to determine the correct qubit mapping.

If your mapping is already implemented,
    the only interaction you need to have with this module
    is to find which string key your mapping corresponds to.

If your mapping is not already implemented,
    you'll need to add your own to the `CONSTRUCT_BINARY_CODE` dict.

"""

import openfermion
from openfermion import jordan_wigner_code as JW
from openfermion import bravyi_kitaev_code as BK
from openfermion import parity_code as P

""" Map string labels to a qubit-mapping function.

Each value is a function accepting a single argument `openfermion.MolecularData`,
    and returning an `openfermion.BinaryData`.

Look at the code to see all the valid string labels.

Typical labeling scheme uses an acronym, eg. "JW" for Jordan-Wigner,
    and attaches a suffix like "-n" to indicate some modification.

Modifications:
    "-n": a one-qubit tapering valid for Hamiltonians preserving particle number
    "-m": a two-qubit tapering valid for Hamiltonians preserving spin and particle number

"""
CONSTRUCT_BINARY_CODE = {
    # JW: Jordan-Wigner
    "JW": lambda molecule: JW(molecule.n_qubits),
    "JW-n": lambda molecule: _taper_n(molecule) * JW(molecule.n_qubits-1),
    "JW-m": lambda molecule: _taper_m(molecule) * JW(molecule.n_qubits-2),

    # BK: Bravyi-Kitaev
    "BK": lambda molecule: BK(molecule.n_qubits),
    "BK-n": lambda molecule: _taper_n(molecule) * BK(molecule.n_qubits-1),
    "BK-m": lambda molecule: _taper_m(molecule) * BK(molecule.n_qubits-2),

    # P: Parity
    "P":  lambda molecule: P(molecule.n_qubits),
    "P-n": lambda molecule: _taper_n(molecule) * P(molecule.n_qubits-1),
    "P-m": lambda molecule: _taper_m(molecule) * P(molecule.n_qubits-2),
}

def _taper_n(molecule):
    """ Construct a binary code tapering one qubit by particle number conservation.

    Parameters
    ----------
    molecule (openfermion.MolecularData)

        The molecule is assumed to include basic derived data. In other words,
        call `molecules.fill_electronicstructure(molecule)` before this method.

    Returns
    -------
    code (openfermion.BinaryCode)

    """
    Ne = molecule.n_electrons
    return openfermion.checksum_code(molecule.n_qubits, Ne & 1)

def _taper_m(molecule):
    """ Construct a binary code tapering two qubits by conservation of each spin.

    Parameters
    ----------
    molecule (openfermion.MolecularData)

        The molecule is assumed to include basic derived data. In other words,
        call `molecules.fill_electronicstructure(molecule)` before this method.

    Returns
    -------
    code (openfermion.BinaryCode)

    """
    Nα = molecule.get_n_alpha_electrons()
    Nβ = molecule.get_n_beta_electrons()
    α_code = openfermion.checksum_code(molecule.n_qubits//2, Nα & 1)
    β_code = openfermion.checksum_code(molecule.n_qubits//2, Nβ & 1)
    stagger = openfermion.interleaved_code(molecule.n_qubits)
    return stagger * (α_code + β_code)
