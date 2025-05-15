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

import scipy

class Sector:
    def __init__(self, n=None, Ne=None, Nα=None):
        """ Encapsulate the parameters needed to know what to do for any mapping.

        NOTE: Not every parameter is needed for every mapping.
            You only need to specify those parameters relevant to your intended mapping.
            (Other values may be filled in for computational simplicity,
                but they can be ignored.)

        NOTE: Probably the code will change completely
            if we add more ambitious mappings like BK-superfast or first-quantization.
        But in principle the same interface can be used, if the constructor is modified.

        Parameters
        ----------
        n (int): the number of spin orbitals being represented in second-quantization
        Ne (int): the number of electrons in the sector
        Nα (int): the number of spin-α electrons in the sector

        """
        if n is None: raise ValueError("Please tell me something about the sector.")
        if Ne is None:  Ne = n >> 1
        if Nα is None:  Nα = Ne >> 1    # NOTE: By default, Nα ≤ Nβ.
                                            # This is opposite to pyscf convention.
        Nβ = Ne - Nα
        assert n >= Nα + Nβ             # THERE MUST BE ENOUGH ORBITALS FOR EACH ELECTRON

        self.n = n
        self.Ne = Ne
        self.Nα = Nα
        self.Nβ = Nβ
        self.m = abs(Nα-Nβ) + 1         # Multiplicity.

    def __str__(self):
        return f"{self.n}_{self.Ne}_{self.Nα}"

    def nstates(self):
        """ Calculate the total number of configurations in this sector. """
        L = self.n >> 1
        return int(scipy.special.comb(L,self.Nα) * scipy.special.comb(L,self.Nβ))


""" Map string labels to a qubit-mapping function.

Each value is a function accepting a `Sector` object,
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
    "JW": lambda sector: JW(sector.n),
    "JW-n": lambda sector: _taper_n(sector) * JW(sector.n-1),
    "JW-m": lambda sector: _taper_m(sector) * JW(sector.n-2),

    # BK: Bravyi-Kitaev
    "BK": lambda sector: BK(sector.n),
    "BK-n": lambda sector: _taper_n(sector) * BK(sector.n-1),
    "BK-m": lambda sector: _taper_m(sector) * BK(sector.n-2),

    # P: Parity
    "P":  lambda sector: P(sector.n),
    "P-n": lambda sector: _taper_n(sector) * P(sector.n-1),
    "P-m": lambda sector: _taper_m(sector) * P(sector.n-2),
}

def _taper_n(sector):
    """ Construct a binary code tapering one qubit by particle number conservation.

    Parameters
    ----------
    obj: has the following fields:
        L (int): number of spatial orbitals
        N (int): number of electrons

    Returns
    -------
    code (openfermion.BinaryCode)

    """
    return openfermion.checksum_code(sector.n, sector.Ne & 1)

def _taper_m(sector):
    """ Construct a binary code tapering two qubits by conservation of each spin.

    Parameters
    ----------
    obj: has the following fields:
        L (int): number of spatial orbitals
        M (int): number of α electrons
        N (int): number of electrons

    Returns
    -------
    code (openfermion.BinaryCode)

    """
    L = sector.n // 2       # NUMBER OF SPATIAL ORBITALS
    α_code = openfermion.checksum_code(L, sector.Nα & 1)
    β_code = openfermion.checksum_code(L, sector.Nβ & 1)
    stagger = openfermion.interleaved_code(2*L)
    return stagger * (α_code + β_code)