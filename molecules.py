""" Construct the fermionic operators for simple molecules.

The module provides a number "molecule factory" methods,
    which produce an `openfermion.MolecularData` object according to some template.

Once you have your molecule,
    pass it along to one of the methods in the `operators` module.

This module also provides the `fill_electronicstructure` method
    to conveniently run FCI or other methods on your molecule when you need them,
    but the SCF calculations are run as needed in `matrices.constructmatrix`.

"""

MOLECULE_DIRECTORY = "./molecule"

import openfermion
import openfermionpyscf

def custom(geometry, basis='sto-3g', charge=0, multiplicity=1, label=""):
    """ Manually specify all inputs to `openfermion` `MolecularData` class.

    Parameters
    ----------
    geometry: specifies atoms and configuration

        `geometry` is (typically) a list of two-tuples:
            - the first element of each tuple is a chemical symbol
            - the second is a three-tuple specifying (x,y,z) coordinates in Å

        For example, H2 might be specified with:
            `geometry=[('H',(0,0,0)),('H',(0,0,0.75))]`

        `openfermion` accepts other formats, which also work here.

    basis (str): specifies atomic orbital basis set

        The list of valid basis sets can be found here:
            https://pyscf.org/_modules/pyscf/gto/basis.html
        A standardized repository is here: https://www.basissetexchange.org/

        `pyscf` strips special characters and lowers all capital letters.

    charge (int): specifies the total charge on the molecule

    multiplicity (int): specifies the multiplicity of the molecule

        Multiplicity `m` refers to the number of splittings in an IR spectrum
            (ie. "singlet" peaks have m=1, "triplet" peaks have m=3, etc.)
        These peaks relate to symmetry-breaking of spin states under a magnetic field.
        The more atomic quantity is "total spin" `J`.
        The two are related by the simple formula `J = 2m - 1`.

    label (str): unique identifier for the molecule

        Actually, `openfermion` creates an identifier based on the other inputs.
        Therefore, `label` needs only be unique for molecules
            with the same basis, charge, multiplicity, and *chemical formula*.
        Please note that "same chemical formula" may include
            not only different geometries of one molecule,
            but also isomers of that molecule, maybe.

    Returns
    -------
    molecule (openfermion.MolecularData)

    """
    return openfermion.MolecularData(
        geometry=geometry,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        description=label,
        data_directory=MOLECULE_DIRECTORY,
    )

def H2(d, basis='sto-3g', multiplicity=1):
    """ Diatomic molecular hydrogen.

    Parameters
    ----------
    d (float): distance between the two nuclei
    basis (str): see `custom` method for details
    multiplicity (int): see `custom` method for details

    Returns
    -------
    molecule (openfermion.MolecularData)

    """
    geometry = [
        ('H', (0, 0, 0)),
        ('H', (0, 0, d)),
    ]
    label = str(round(d,8))
    return custom(geometry, basis=basis, multiplicity=multiplicity, label=label)

def HChain(n, d, basis='sto-3g', multiplicity=None):
    """ A uniformly-spaced chain of hydrogen atoms.

    Parameters
    ----------
    n (int): number of hydrogen nuclei
    d (float): distance between each nucleus
    basis (str): specifies atomic orbital basis set

        See `custom` method for additional details.

    multiplicity (int): specifies the multiplicity of the molecule

        Defaults to 1 (singlet) when n is even, or 3 (triplet) when n is odd.

        See `custom` method for additional details.

    Returns
    -------
    molecule (openfermion.MolecularData)

    """
    if multiplicity is None:
        J = n & 1               # LOWEST SPIN STATE (0 if n is even, 1 if n is odd)
        m = (J+1) & 1           # CORRESPONDING MULTIPLICTY

    geometry = [('H', (0, 0, i*d)) for i in range(n)]
    label = str(round(d,8))
    return custom(geometry, basis=basis, multiplicity=m, label=label)

# TODO: LiH
# TODO: HeH+
# TODO: H2O
# TODO: BeH2

# TODO: Freeze orbitals. This is awkward because it should appropriately be some metadata attached to the `MolecularData` object, so we should handle it here. But, the actual freezing seems to happen on the `FermionOperator` object. :?




def fill_electronicstructure(molecule, load=True, SCF=True,
                       FCI=False, CCSD=False, MP2=False, CISD=False):
    """ Fill out a molecule with pyscf calculations, if they haven't already been done.

    Modifies the molecule object, and also returns it for convenience/aesthetic.

    Parameters
    ----------
    molecule (openfermion.MolecularData)

    load (bool): load existing data, if it exists

        The method works perfectly well if the data *doesn't* exist,
            so only set this to False if you don't trust previous calculations
            and you actively want to overwrite them.

        If the loaded data already contains calculations for any method below,
            those calculations are skipped, regardless of the following parameters.

    SCF (bool): run self-consistent Hartree-Fock calculations?

        You'll always need these,
            so I don't think there is any reason to set this False.

    FCI (bool): run full configuration interaction calculations?

        FCI is the "exact" solution for a given basis set,
            so it's always important to have for assessing accuracy.
        It can get intractable for large molecules,
            but probably not ones we're running. ;)

    CCSD (bool): run singles/doubles coupled-cluster theory calculations?

        CCSD is the so-called "gold standard" of classical methods,
            ie. not the best but usually good enough.
        It's probably good to include for reference in actual publications,
            but otherise isn't very important.

    MP2 (bool): run Moller-Plosset 2nd-order perturbation theory calculations?
    CISD (bool): run singles/doubles configuration interaction calculations?

        Near as I can tell,
            these and other methods aren't that useful for quantum computation,
            but `openfermionpyscf` provides the interface for these two.

    Returns
    -------
    molecule (openfermion.MolecularData): same object as input, now modified

    """
    # LOAD PREVIOUS CALCULATIONS
    if load:
        try:
            molecule.load()
        except OSError:
            pass    # NEED TO RUN CALCULATIONS ANYWAY

    # NO NEED TO RE-RUN CALCULATIONS WE'VE ALREADY DONE
    SCF = SCF and molecule.hf_energy is None
    FCI = FCI and molecule.fci_energy is None
    MP2 = MP2 and molecule.mp2_energy is None
    CISD = CISD and molecule.cisd_energy is None
    CCSD = CCSD and molecule.ccsd_energy is None

    # RUN NEW CALCULATIONS
    openfermionpyscf.run_pyscf(
        molecule,
        run_scf=SCF,
        run_mp2=MP2,
        run_cisd=CISD,
        run_ccsd=CCSD,
        run_fci=FCI,
        verbose=False,
    )
    molecule.save()

    return molecule