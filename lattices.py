""" Construct the fermionic operators for simple lattices, eg. Hubbard models.

The module provides a number of "lattice factory" methods,
    which produce an `openfermion.MolecularData` object according to some template.

Once you have your molecule,
    pass it along to one of the methods in the `operators` module.

This module also provides the `fill_electronicstructure` method
    to conveniently run FCI or other methods on your molecule when you need them,
    but the SCF calculations are run as needed in `matrices.constructmatrix`.

"""

import numpy
import pyscf
import openfermion

import mappings

class Hubbard:
    def __init__(self, Lx, Ly, Lz, t, U, μ, h, pbc, phs):
        """ I don't expect we'll ever care to do this,
                but it's worth recalling a generic Hubbard lattice has all these parameters.

            Lx, Ly, Lz: size of lattice for up to three dimensions
            t, U, μ, h: parameters in the actual Hamiltonian,
                including chemical potential μ and magnetic field h

            pbc: periodic boundary conditions
            phs: particle hole symmetry, ensures ground-state is a half-filling state

        """
        self.L = Lx * Ly * Lz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz            # NOTE: openfermion does NOT admit Lz...

        self.t = t
        self.U = U
        self.μ = μ
        self.h = h

        self.pbc = pbc
        self.phs = phs

        self.spinless = False   # NOTE: I have no idea what this is or what it's for.

        # Calculate core orbital basis (diagonalizing one-body terms)
        _, self.C = numpy.linalg.eigh(self.one_body_integrals())

        # Call `run_eigenspectrum` to fill these.
        self._fci = False
        self.Λ = None       # To be filled with openfermion eigenspectrum.

    def __str__(self):
        """ Uniquely identify this model. """
        raise NotImplementedError

    def one_body_integrals(self, C=None):
        """

        By default, use AO basis.
        If `C` is `True`, use CO basis.
        If `C` is an `ndarray`, use `C` as the basis.

        TODO: Generalize to higher dimensions.

        TODO: Ah, if h is non-zero, this whole framework stops making sense.
            You'd need different integrals for α vs β sector.
            No idea how that works...

        """
        if isinstance(C, numpy.ndarray):
            obi = self.one_body_integrals()
            return C.conjugate().T @ obi @ C

        if self.Ly > 1 or self.Lz > 1:
            raise NotImplementedError("obi for 2d. easy to do")
        if self.h != 0:
            raise NotImplementedError("obi for h? not easy")

        obi = numpy.zeros((self.L,self.L))
        for i in range(self.L-1):
            obi[i,i+1] = obi[i+1,i] = -self.t
        for i in range(self.L):
            obi[i,i] = -self.μ

        if self.pbc:
            obi[self.L-1,0] = obi[0,self.L-1] = -self.t
        if self.phs:
            for i in range(self.L):
                obi[i,i] += -self.U/2
        return obi

    def two_body_integrals(self, C=None):
        """

        By default, use AO basis.
        If `C` is `True`, use CO basis.
        If `C` is an `ndarray`, use `C` as the basis.

        """
        if isinstance(C, numpy.ndarray):
            eri = self.two_body_integrals()
            return pyscf.ao2mo.incore.full(eri, C)

        eri = numpy.zeros((self.L,self.L,self.L,self.L))
        for i in range(self.L):
            eri[i,i,i,i] = self.U
        return eri

    def get_fermionic_operator(self, C=None):
        """

        By default, use AO basis.
        If `C` is `True`, use CO basis.
        If `C` is an `ndarray`, use `C` as the basis.

        """
        if C is True:
            return self.get_fermionic_operator(C=self.C)

        if isinstance(C, numpy.ndarray):
            # CONSTRUCT THE ONE- AND TWO-BODY INTEGRAL TENSORS (MO basis)
            obi = self.one_body_integrals(C=C)
            tbi = self.two_body_integrals(C=C)

            # USE `openfermion` TO MAKE THE FERMION OPERATOR
            h0 = (self.U*self.L)/4 if self.phs else 0.0
            h1, h2 = openfermion.ops.representations.get_tensors_from_integrals(obi, tbi)
            interop = openfermion.InteractionOperator(h0, h1, h2)
            return openfermion.get_fermion_operator(interop)

        if self.Lz > 1: raise NotImplementedError
        return openfermion.hamiltonians.fermi_hubbard(
            self.Lx, self.Ly,   # NOTE: openfermion does not admit Lz
            self.t, self.U, self.μ, self.h,
            self.pbc,
            False,              # NOTE: openfermion admits `spinless` flag
            self.phs,
        )


    def run_scf(self, Ne=None, Nα=None, **kwargs):
        """ Perform pyscf calculations to fill in fields `C` and `REF`. """
        ##################################################################################
        # The following code is adapted from an example on the pyscf website:
        #   https://pyscf.org/quickstart.html#integrals-density-fitting
        # and from the published code example in Figure 4 of arxiv::1701.08223.

        if Ne is None: Ne = self.L  # DEFAULT TO HALF-FILLING
        if Nα is None: Nα = Ne >> 1 # DEFAULT TO SINGLET STATE, OR Nβ-Nα = 1 IF Ne IS ODD
                                    # (Note that this is how `half_filling()` is defined.)

        # INITIALIZE THE MOLECULE OBJECT
        n = self.L
        molecule = pyscf.gto.M()
        molecule.nelectron = Ne
        molecule.spin = (Nα << 1) - Ne  # Nα - Nβ, where Nβ = Ne - Nα
        molecule.incore_anyway = True           # NOTE: Guarantee SCF uses custom `_eri`.

        # CONSTRUCT THE ONE- AND TWO-BODY INTEGRAL TENSORS (AO basis)
        obi = self.one_body_integrals()
        eri = self.two_body_integrals()

        # CONSTRUCT AND RUN THE SCF OBJECT
        rhf = pyscf.scf.RHF(molecule)
        rhf.get_hcore = lambda *args: obi       # NOTE: This...feels like a "hack" to me.
        rhf.get_ovlp = lambda *args: numpy.eye(n)   #   It is published in a paper.
                                                    #   I find that delightful.
        rhf._eri = pyscf.ao2mo.restore(8, eri, n)   # NOTE: Store eri with smaller array.
        rhf.init_guess = "hcore"    # NOTE: Without any atoms, there is no other choice.
        rhf.run(**kwargs)

        # CORRECT FOR CONSTANT SHIFT
        if self.phs:    rhf.e_tot += (self.U * self.L) / 4
        return rhf

    def run_eigenspectrum(self):
        fermiop = self.get_fermionic_operator()
        self.Λ = openfermion.eigenspectrum(fermiop)
        self._fci = True

    def half_filling(self):
        """ Following Essler et al., default to Nα ≤ Nβ.
            This is opposite to pyscf convention.
        """
        L = self.L
        return mappings.System(L<<1, L, L>>1)

class SimpleHubbard(Hubbard):
    def __init__(self, L, u, pbc=False, phs=False):
        """ Strictly 1d, parameterized only by u≡U/4t. """
        super().__init__(L, 1, 1, 1.0, 4*u, 0.0, 0.0, pbc, phs)
        self.u = u

    def __str__(self):
        name = f"Hubbard_L{self.L}_u{self.u}"
        if self.pbc:    name += "_pbc"
        if self.phs:    name += "_phs"
        return name
