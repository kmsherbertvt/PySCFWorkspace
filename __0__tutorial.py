""" pyscf AND openfermion TUTORIAL

VQE begins with a Hamiltonian H and a reference state |Φ⟩.
Typically, we want H as a linear combination of Paulis,
    and |Φ⟩ as a computational basis state.
Making that happen involves a lot of chemistry and quantum information behind the scenes.
This script describes all that work, to the best of my ability.

The `toolkit` module helps do a lot of this work without having to understand it,
    but this tutorial does it all from "scratch" (i.e. using pyscf and openfermion),
    so that you CAN understand it. :)

"""

import numpy            # Always useful.
import pyscf            # Knows how to do chemistry.
import openfermion      # Knows how to put chemistry on a quantum computer.

##########################################################################################
# PySCF phase: build out the molecule and obtain the molecular-orbital integrals.

""" BUILD OUT A MOLECULE

Naturally, pyscf has a very sophisticated internal representation for molecules.

In our electronic structure calculations,
    we pretty much ALWAYS take the Born-Oppenheimer approximation,
    aka that nuclear coordinates are fixed for the energy scales of interest.
So, a molecule is defined principally by these nuclear coordinates,
    identifying the relative locations of each `atom`.

But the same nuclear configuration can exist in a multitude of electronic species.
The `charge` determines the number of electrons.
The `spin` determines the relative number of spin "up" and spin "down" electrons.

The pyscf representation of a molecule also bakes in
    some decisions on how to do the mathematics.
Most importantly, the `basis` defines the functional form
    of the "orbitals" which a single-electron wavefunction is decomposed into.
These are like the s,p,d,f,... orbitals of hydrogen,
    but contrived heuristically to enhance performance in some vague way.
There is a whole zoo of them, all specified by some standard string name.
The most common choice for quantum information is "sto-3g"
    (Slater Type Orbitals with 3 Gaussians...don't ask me what any of that means),
    which is considered a bare-minimum basis.

You can also choose whether or not to exploit point-group `symmetry` of the molecule.
I'm...really not sure why you *wouldn't* do this.
As far as I understand,
    it should only ever make the math faster and more easily interprable.
I seem to recall that pyscf has relatively limited support for point-groups,
    though, so beware that other alternatives to pyscf could be better sometimes.

"""
molecule = pyscf.gto.M(
    # SET THE PHYSICS
    atom = [                        # There are many ways to specify the geometry,
        ['Li',  (0, 0, 0)],         #   including reading from a file.
        ['H',   (0, 0, 1.5)],
    ],
    charge = 0,                     # charge = num protons - num electrons
    spin = 0,                       # spin = num α electrons - num β electrons
    # SET THE MATH
    basis = "sto-3g",               # Sets the number of orbitals.
    symmetry = True,                # Makes the math nicer.
)


""" FETCH THE ATOMIC INTEGRALS

By restricting our electronic wavefunctions to a finite `basis` of orbitals,
    we are able to represent our real-space Hamiltonian with a finite matrix.
Each of those matrix elements is defined by some integral over real-space
    of the orbital wavefunctions.
The main reason to use pyscf is that it knows how to compute those integrals.

The Hamiltonian has several components (kinetic, Coulomb attraction, Coulomb repulsion)
    which can be broken down into those involving one electron, or two.
Or zero - that is, there is a constant shift to the energy
    from the Coulomb repulsion of all the nuclei.
So that's a scalar floating about, which we'll want to keep track of (`nuc`).

With one electron, two different orbitals can be involved in the integral.
Thus, you can talk about an N⨯N matrix of integrals, the one body integrals.
There are two types of one-body integrals,
    an electron's Coulomb interaction with all the nuclei,
    and an electron's self-interaction (aka kinetic energy).
At the level of quantum computing, there is no reason to distinguish them,
    so we just add them together to make `obi`.

With two electrons, as many as four different orbitals can be involved,
    so you can talk about an N⨯N⨯N⨯N array of integrals.
The only type of two-body integral comes from the electronic repulsion interaction,
    so pyscf refers to this with the acronym `eri`.
(I might have called it the more symmetric `tbi` instead,
    but tutorials are meant to familiarize you with existing conventions,
    even if they're silly...)

"""
nuc = molecule.energy_nuc()             # Nuclear repulsion.
ao_kin = molecule.intor('int1e_kin')    # Electronic kinetic energy.
ao_nuc = molecule.intor('int1e_nuc')    # Nuclear-electronic attraction.
ao_obi = ao_kin + ao_nuc                # Single-electron hamiltonian.
ao_eri = molecule.intor('int2e')        # Electonic repulsion interaction.

""" RUN HARTREE-FOCK TO FIND MOLECULAR ORBITAL BASIS

We defined the so-called ATOMIC orbitals by setting the `basis` above.
The next step is to design the so-called MOLECULAR orbitals,
    which are linear combinations of atomic orbitals forming an eigenbasis
    for some chemically meaningful operator, called the Fock operator.
All we need is the matrix which describes those linear combinations,
    so we can use it as a basis rotation
    to transform the atomic integrals into molecular integrals.

The main reason to use the molecular orbital basis is that
    it produces a very reasonable and well-defined starting point for VQE.
(The natural choice of reference state in the molecular orbital basis
    is provably THE best one can do with a single computational basis state,
    when taking a "one qubit per orbital" perspective. I think. :/)
One COULD work in the atomic basis set if one really wanted to.
However, you would still need to ORTHOGONALIZE the atomic orbitals,
    so you would still need to do SOME sort of basis rotation.
And the process to get the basis rotation for molecular orbitals is so streamlined
    that it's usually easier to just get on that bandwagon.
Seriously - I've had projects where I wanted to use an atomic orbital basis before,
    but the only good way I found to orthogonalize them
    involved finding the molecular orbitals first.

The Fock operator is a version of the electronic Hamiltonian
    in which there is just a single quantum electron,
    interacting with the nuclei and a classical "mean-field" of all the other electrons.
It's a funny sort of operator, because (at least the way *I* think of it)
    the classical "mean-field" of all the other electrons is defined
    in terms of the eigenstates of the quantum electron.
It's a bit circular.
So you can't merely diagonalize a matrix and be done;
    you've got to run some kind of loop until your eigenstates have converged.
And there are nearly as many ways to do that as there are `basis` sets. o_o

The simplest (in some vague sense) is known as "restricted" Hartree-Fock (RHF).
Other systems may require something more or less sophisticated.
Select the method in pyscf by constructing a different type of object.

Even for a given method (i.e. RHF),
    there are many other settings to control the flow of optimization.
Below, I'm highlighting just one, `max_cycle`,
    which is the maximum number of iterations in the optimization loop.
If this number is reached and the molecular orbitals have not yet converged,
    pyscf will give up prematurely.
The default is 50, but I've seen certain systems
    (e.g. cyclic or hedral hydrogen clusters) need more.
Other systems may require something more or less sophisticated.

"""
scf = pyscf.scf.RHF(molecule)   # This initializes the calculation but does not run.
scf.max_cycle = 1000            # Lets optimization run longer if it needs to.
scf.run()                       # This line runs the self-consistent optimization.
U = scf.mo_coeff                # Basis transformation from atomic to molecular orbitals.

""" TRANSFORM ATOMIC INTEGRALS INTO MOLECULAR INTEGRALS

Our N⨯N and N⨯N⨯N⨯N array of integrals are currently with respect to atomic orbitals.
We just want to do a linear transformation to put them in terms of molecular orbitals.

This is really simple written out as a tensor contraction,
    but of course our computational chemistry software framework would not be complete
    without a very confusing interface.
See `toolkit.py` for a less confusing interface, but see below for, near as I can tell,
    the standard way in which pyscf is used.

"""
full_obi = U.T @ ao_obi @ U                     # Rotated single-body integrals.
full_eri = pyscf.ao2mo.incore.full(ao_eri, U)   # Rotated two-body integrals.

""" IDENTIFY THE RELEVANT ORBITALS

The choice of atomic orbital `basis` set is the primary way
    of determining the size of the system.
But maybe that still leaves you with too many orbitals to simulate.

The Hartree-Fock reference state is a single basis state in the molecular orbital basis;
    each spin orbital is definitely occupied (low energy) or unoccupied (high energy).
One can reasonably expect that orbitals with particularly low energy
    compared to others (aka "core" orbitals)
    will REMAIN approximately occupied in a more exact ansatz.
Similarly, molecular orbitals with particularly high energy
    may remain approximately unoccupied.
So maybe we can get away with leaving those orbitals alone.

The unoccupied orbitals you decide to leave out can simply be omitted
    from your one and two body integrals.
Core orbitals that you would like to freeze in their definitely occupied state
    need to be handled more carefully,
    because the electrons frozen in those orbitals still interact with active electrons -
    the integrals we retain will need to be modified accordingly.
This is a simple but tedious "downfolding" procedure, which openfermion will handle.
We just need to tell it which core orbital indices to `freeze`,
    and which orbital indices to treat as `active`.
The unoccupied orbitals you want to omit are simply
    all those orbitals not specified in `freeze` or `active`.

I do not know how to decide which orbitals to freeze or omit.
The example below is, as I understand, a common choice for LiH,
    but I do not where it falls in the spectrum between algorithm and heuristic.

"""
freeze = [0]            # Orbitals to be traced over, modifying numbers for the rest.
active = [1,2,5]        # Orbitals to keep. All others will simply be dropped.

##########################################################################################
# OpenFermion phase: select the active space and build out a fermionic operator.

""" TRANSFORM INTEGRALS ONTO ACTIVE SPACE

This first line is the downfolding onto the active space,
    as discussed in the previous cell.

The only other thing to point out here is that
    each of the four orbitals appears in the two body integral in different ways,
    so there is a difference between `h[p,q,r,s]` and `h[p,r,q,s]`.
Unfortunately, openfermion chooses to order its two body integrals differently than pyscf,
    so we have to permute (aka `transpose`) the indices.

"""
core, obi, eri = openfermion.ops.representations.get_active_space_integrals(
    full_obi, full_eri.transpose(0,2,3,1),
    # Transpose to account for tensor index conventions between pyscf and openfermion.
    freeze, active,
)

""" BUILD OUT THE SPIN-ORBITAL TENSORS

To this point, we have been discussing only SPATIAL orbitals.
But it is SPIN orbitals that we will be associating with qubits.
We now need to take our molecular integrals
    and construct the coefficients for each one- and two-body spin orbital interaction
    appearing in the second-quantized Hamiltonian.

This is another simple but tedious antisymmetrization procedure;
    openfermion will handle it.

"""
h0 = nuc + core         # Tracing out core orbitals shifts the constant term.
h1, h2 = openfermion.ops.representations.get_tensors_from_integrals(
    obi, eri,
)

""" CONSTRUCT THE FERMIONIC OPERATOR

The final step of this phase is to construct the second-quantized Hamiltonian.
We'll start with the chemistry-aware `InteractionOperator` representation,
    but our goal is openfermion's flagship representation, the `FermionOperator`.

"""
interop = openfermion.InteractionOperator(h0, h1, h2)
fermiop = openfermion.get_fermion_operator(interop)

##########################################################################################
# Qubit mapping phase (also OpenFermion): perform the mapping and the tapering.

""" IDENTIFY SOME NUMBERS

This next phase transforms a Hamiltonian defined with fermionic operators
    to one defined with Pauli operators.
The simplest way to do this is via the Jordan-Wigner transformation,
    but there are more sophisticated methods, which we will use in this tutorial.
They require some knowledge of the quantum numbers in the system.

You'll generally know which numbers you are working with
    and you could just input them by hand, but in this cell,
    I demonstrate how to extract them programatically from the objects we have on hand.

"""
ηα, ηβ = molecule.nelec     # Total α and β electron counts.
ηα -= len(freeze)           # Account for filled core orbitals.
ηβ -= len(freeze)           #       "                   "
η = ηα + ηβ                 # Active electron count.

nspatial = len(active)      # Active spatial orbital count.
nspin = 2 * nspatial        # Active spin orbital count.

""" DESIGN THE BINARY CODE FOR THE PARITY MAPPING

The Jordan-Wigner mapping is the default choice for moving from fermions to qubits,
    but it is not the only choice.
The parity mapping and the Bravyi-Kitaev mapping are also common alternatives.
The openfermion package has one-liners to perform all three of these;
    however, we will also be applying two-qubit tapering in this tutorial,
    so it is more convenient to use a slightly more elaborate framework
    which uses the language of binary codes.

Thus, below, we construct the binary code corresponding to the parity mapping,
    specifying a number of qubits two fewer than the number of spin orbitals
    (anticipating the two Z2 symmetry reductions constructed in the next cell).

"""
basecode = openfermion.parity_code(nspin - 2)

""" DESIGN THE BINARY CODE FOR 2-QUBIT TAPERING

Qubit tapering refers to exploiting symmetries in the Hamiltonian to remove qubits.

It is easiest (though not easy) to understand in terms of Z2 symmetries.
Think of it this way: for each independent Z2 symmetry in the Hamiltonian,
    half of the states in the full Hilbert space are redundant.
So, if we are clever enough, we should be able to do equivalent simulations
    on just one half of the Hilbert space,
    and so we should be able to make do with one less qubit.

Closed-system molecular Hamiltonians come ready-made with a few symmetries:
    they never change the number of electrons,
    nor do they change the number of electrons in the α or β registers.
Now, these symmetries aren't Z2 - they're actually quite a bit stronger than Z2.
But they do CONTAIN some Z2 symmetries:
    we can say it isn't the NUMBER of electrons,
    but the PARITY of the number of electrons which is conserved.
We have three such symmetries, but they're not independent:
    given the parity of the number of electrons in the α and β registers,
    the parity of the total number of electrons is fixed.
Thus, we have TWO Z2 symmetries, which allow us to taper off TWO qubits.

I can't really explain to you how exactly this thing works.
From the perspective of binary codes,
    it's a bit like the inverse of a parity check code classical error correcting code,
    hence the function name `checksum_code`.
There is also a good explanation in terms of stabilizers and Clifford transformations,
    but I'm not qualified to give it.
Look it up. :) The term is "qubit tapering".

"""
α_code = openfermion.checksum_code(nspatial, ηα & 1)
β_code = openfermion.checksum_code(nspatial, ηβ & 1)
stagger = openfermion.interleaved_code(nspin)
tapercode = stagger * (α_code + β_code)

""" CONCATENATE THE CODES AND APPLY IT TO THE FERMIONIC OPERATOR

The reason the binary code formalism is so nice is that
    you can string together multiple codes in sequence with code concatenation,
    which openfermion has elected to implement using the * operator.
So qubit tapering is utterly compatible with any other mapping technique.

"""
binarycode = tapercode * basecode
qubitop = openfermion.binary_code_transform(fermiop, binarycode)

""" ENCODE THE REFERENCE KET

The Hartree-Fock estimate of the multi-electron wavefunction
    for the ground state of a molecule with η electrons consists of
    a single configuration with the η lowest-energy spin orbitals being occupied.
In the Jordan-Wigner mapping (with openfermion ordering `αβαβ`),
    that yields the very intuitive reference state `|1..10..0⟩`.
But what does it look like when applying an arbitrary binary code?
It becomes very, very difficult to interpret bitstrings after qubit tapering.

Fortunately, the binary code is *defined* in terms of what it does to each bitstring.
The openfermion `BinaryCode` object has the `encoder` attribute,
    which is a linear transformation (represented with a sparse matrix)
    that acts on binary vectors.
So here, we prepare our `|1..10..0⟩` state as a binary vector,
    and then we simply do the matrix-vector multiplication (% 2 to keep it binary).

"""
reference = numpy.zeros(nspin, dtype=int)
for i in range(ηα): reference[2*i] = 1
for i in range(ηβ): reference[2*i+1] = 1
ket = (binarycode.encoder @ reference) % 2

##########################################################################################
# Serialization phase: write the qubit operator and reference ket compactly to a file.

""" SERIALIZE IN SYMPLECTIC FORM

At this point, we have a Pauli sum and a reference state in the target qubit mapping.
All that remains is to export them for use in VQE code, possibly not in Python.

We will use the `numpy` data format,
    specifically an `npz` file saving four arrays:
- `ket`: the binary vector representing the bitstring reference state
- `C`: the float vector giving coefficients for each Pauli term
- `X`, `Z`: parallel integer vectors identifying the Pauli word in each Pauli term

The integers in X and Z are the so-called symplectic notation for Paulis.
Writing the integers as length-n bitstrings, the 1's identify which qubits have an X or Z.
If a qubit has both an X and a Z, it is interpreted as having a Y.
So e.g. the Pauli word `XIYZ` would have bitstrings x `1010` and z `0011`,
    so the `X` and `Z` vectors would have integers x=10 and z=3.

"""
n = len(ket)
X = []; Z = []; C = []
for actions, c in qubitop.terms.items():
    x = 0; z = 0
    for (q, σ) in actions:
        mask = 1<<(n-q-1)                       # Selects a specific qubit.
        if σ == "X" or σ == "Y": x ^= mask      # XORs the specific qubit.
        if σ == "Z" or σ == "Y": z ^= mask      #   "               "
    X.append(x); Z.append(z); C.append(c)
numpy.savez("systems/tutorial.npz", X=X, Z=Z, C=C, ket=ket)

