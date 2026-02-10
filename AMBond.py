#!/usr/bin/env python3.11
from ase.io import read
import spglib
import numpy as np
import ase
from ase.neighborlist import NeighborList
from ase import Atoms
from amcheck import is_altermagnet
import argparse
import sys
from diophantine import lllhermite

DEFAULT_TOLERANCE = 1e-3

class Bond:
    def __init__(self,start,end,offset = None, tol = 1e-4):
        if offset is None:
            start = np.round(start,round(-np.log10(tol)))
            end = np.round(end,round(-np.log10(tol)))
            end = end - np.floor(start)
            start = start - np.floor(start)
            offset = np.floor(end)
            end = end - offset

        self.start = start 
        self.end = end 
        self.offset = offset
        self.tol = tol

    def __eq__(self,other, tol = None):
        if tol is None:
            tol = self.tol

        if np.allclose(self.start, other.start, atol = tol) and np.allclose(self.end, other.end, atol = tol) and np.allclose(self.offset,other.offset):
            return True
        elif np.allclose(self.start, other.end, atol = tol) and np.allclose(self.end, other.start, atol = tol) and np.allclose(self.offset,-other.offset):
            return True
        return False

    def __str__(self):
        return f"Bond(start={self.start.tolist()}, end={self.end.tolist()}, offset={self.offset.tolist()})"

class BondOrbit:
    def __init__(self,bonds,cell, tol = 1e-4):
        self.bonds = bonds
        self.cell = cell
        self.tol = tol

    def append(self,bond):
        self.bonds.append(bond)

    def get_bond_distance(self):
        return np.linalg.norm((self.bonds[0].end + self.bonds[0].offset - self.bonds[0].start) @ self.cell)

    def __contains__(self,bond, tol = None):
        if tol  is None:
            tol = self.tol

        for b in self.bonds:
            if bond == b:
                return True

        return False

    def __str__(self):
        for bond in self.bonds:
            print(bond)
        return f"Bond distance: {self.get_bond_distance()}" 

    def __getitem__(self,index):
        return self.bonds[index]

def input_spins(num_atoms):
    """
    Read a list of spin designations for a given Wyckoff orbit from stdin.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in the given Wyckoff orbit.

    Returns
    -------
    spins : list of strings
        List of string objects denoting the spin designation of each atom.
        Possible values are: "u", "U" for spin-up, "d", "D" for spin-down and
        "n", "N" to mark a non-magnetic atom.
        "nn" or "NN" can be used to mark entire Wyckoff orbit as non-magnetic.

    Raises
    ------
    ValueError
        If the number of spins from input is not the same as num_atoms.

    ValueError
        If the number of up and down spins is not the same: for an altermagnet,
        spins should be compensated.
    """

    print("Type spin (u, U, d, D, n, N, nn or NN) for each of them (space separated):")
    spins = input().split()
    # "normalize" spin designations to lowercase for easier bookkeeping
    spins = [s.lower() for s in spins]

    # empty line or "nn" in input marks all atoms in the orbit as nonmagnetic
    if len(spins) < 1 or spins[0] == 'nn':
        return ["n"]*num_atoms

    if len(spins) != num_atoms:
        raise ValueError(
            "Wrong number of spins was given: got {} instead of {}!".format(len(spins), num_atoms))

    if not all(s in ["u", "d", "n"] for s in spins):
        raise ValueError("Use u, U, d, D, n or N for spin designation!")

    N_u = spins.count("u")
    N_d = spins.count("d")
    if N_u != N_d:
        raise ValueError("The number of up spins should be the same as the number of down spins: " +
                         "got {} up and {} down spins!".format(N_u, N_d))

    # all atoms in the orbit are nonmagnetic
    if N_u == 0:
        return ["n"]*num_atoms

    return spins

        
def get_symmetry_operations(atoms, symprec=1e-5, verbose=False, tol=1e-3, fullSearch = True):
    cell = atoms.get_cell(complete=True)[:]

    # get the space group number
    spglib_cell = (
        atoms.cell, atoms.get_scaled_positions(), atoms.numbers)
    sg = spglib.get_spacegroup(spglib_cell, symprec=symprec)
    sg_no = int(sg[sg.find('(') + 1:sg.find(')')])


    primitive = spglib.standardize_cell((atoms.cell, atoms.get_scaled_positions(),
                                            atoms.numbers),
                                        to_primitive=True, no_idealize=True,
                                        symprec=symprec)
    prim_cell, prim_pos, prim_num = primitive

    # The given unit cell might be non-primitive, and if this is the
    # case, we will ask the user: shall we keep using it, or shall we
    # use a primitive one instead?

    rotations = []
    translations = []
    if abs(np.linalg.det(atoms.cell) - np.linalg.det(prim_cell)) > tol and fullSearch:
        symmetry = spglib.get_symmetry(primitive, symprec=symprec)
        rotations = symmetry['rotations']
        translations = symmetry['translations']
        N_ops_prim = len(translations)


        symmetry_dataset = spglib.get_symmetry_dataset(
            spglib_cell, symprec=symprec)

        if verbose:
            print(
                "Atoms mapping from primitive to non-primitive unit cell:")
            for i in range(len(prim_num)):
                atoms_mapped = np.where(
                    symmetry_dataset['mapping_to_primitive'] == i)[0]
                print("{}->{}".format(i+1, atoms_mapped+1))

        equiv_atoms = symmetry_dataset['crystallographic_orbits']

        # S = T*P, where S is a unit cell of a supercell and P of
        # a primitive cell
        T = np.rint(np.dot(cell, np.linalg.inv(prim_cell)))

        det_T = np.linalg.det(T)
        det_ratios = np.linalg.det(cell)/np.linalg.det(prim_cell)
        assert np.isclose(det_T, det_ratios),\
        "Sanity check: the determinant of transformation is not equal to \
original cell/primitive cell ratio: got {} instead of {}!".format(det_T, det_ratios)

        if verbose:
            print(
                "Transformation from primitive to non-primitive cell, T:")
            print(T)

        # All possible supercells can be grouped by det(T) and
        # within each group, the amount of possible distinct
        # supercells is finite.
        # All of them can be enumerated using the Hermite Normal
        # Form, H.
        H, _, _ = lllhermite(T)
        H = np.array(H, dtype=int)
        if verbose:
            print("HNF of T:")
            print(H)

        # By knowing the HNF we can determine the direction and
        # multiplicity of fractional translations and transform
        # them into the basis of the original supercell
        tau = [np.mod([i, j, k] @ prim_cell @ np.linalg.inv(cell), 1)
                for i in range(H[0, 0]) for j in range(H[1, 1]) for k in range(H[2, 2])]

        # The final collection of symmetry operations is a copy of
        # the original operations augmented by the new translations:
        # (R,t) = (R0,t0)*{(E,0) + (E,t1) + ... + (E,tN)}
        # However, original fractional translations should also be
        # transformed to the basis of a new cell.
        N = int(np.rint(np.linalg.det(H)))
        rotations = np.tile(rotations, (N, 1, 1))
        translations = np.tile(translations, (N, 1))
        for (i, t) in enumerate(tau):
            for j in range(i*N_ops_prim, (i+1)*N_ops_prim):
                translations[j] = np.mod(translations[j] @ prim_cell @ np.linalg.inv(cell) + t, 1)
        rotations = list(rotations)
        translations = list(translations)

    # The original unit cell is primitive
    if True:
        symmetry = spglib.get_symmetry(
            spglib_cell, symprec=symprec)

        rotations += list(symmetry['rotations'])
        translations += list(symmetry['translations'])
        equiv_atoms = symmetry['equivalent_atoms']

    if verbose:
        print("Number of symmetry operations: ",
                len(rotations), len(translations))
    return rotations, translations, equiv_atoms

def apply_symmetry(pos, rotation, translation):
    return np.dot(pos, rotation.T) + translation 

def apply_symmetry_bond(bond, rotation, translation):
    start = apply_symmetry(bond.start,rotation,translation)
    end = apply_symmetry(bond.end+bond.offset,rotation,translation)
    return Bond(start,end) 

def generate_bond_orbit(bond,rotations,translations):
    bond_orbit_bonds = []
    for rotation,translation in zip(rotations,translations):
        b = apply_symmetry_bond(bond, rotation, translation) 
        if b not in bond_orbit_bonds:
            bond_orbit_bonds.append(b)
    return bond_orbit_bonds

        
def reduce_symmetry_to_bond(rotations,translations,bondOrbit):
    frot = []
    ftrans = []
    for rotation,translation in zip(rotations,translations):
        goodOperation = True
        for bond in bondOrbit:
            if apply_symmetry_bond(bond,rotation,translation) not in bondOrbit:
                goodOperation = False
                continue
        if goodOperation:
            frot.append(rotation)
            ftrans.append(translation)
    return frot,ftrans

def generate_bond_orbits_to_cutoff(atoms, rotations,translations,cutoff = 5.0):
    cutoffs = [cutoff/2] * len(atoms)  # example: 3 Å for all atoms
    nl = NeighborList(cutoffs, self_interaction=False, bothways=False)
    nl.update(atoms)
    
    ref_index = 0
    ref_atom = atoms[ref_index]
    indices, offsets = nl.get_neighbors(ref_index)

    # Collect all initial bonds
    bonds = []
    ref_atom_pos =  atoms.get_scaled_positions()[ref_index]
    for j, offset in zip(indices, offsets):
        end_atom_pos = atoms.get_scaled_positions()[j] + offset
        bonds.append(Bond(ref_atom_pos,end_atom_pos))



    bondOrbits = []
    while len(bonds)>0:
        bond = bonds[-1]
        inBondOrbits = False
        for bondOrbit in bondOrbits:
            if bond in bondOrbit:
                inBondOrbits = True
                continue

        if not inBondOrbits:
            orbit_bonds = generate_bond_orbit(bond,rotations,translations) 
            bondOrbits.append(BondOrbit(orbit_bonds,atoms.cell))

        bonds.pop()
                
    bondOrbits.sort(key=lambda bond: bond.get_bond_distance())
    return bondOrbits

def reduce_atoms_to_subset(atoms, equiv_atoms, idx):
    mag_atoms = atoms.copy() 
    ligands_i = np.where(equiv_atoms != idx)[0]

    for i in ligands_i[::-1]:
        mag_atoms.pop(i=i)
    return mag_atoms

def main(args):
    """ Run altermagnet/antiferromagnet structure analysis interactively. """
    if args.verbose:
        print('spglib version:', spglib.__version__)

    for filename in args.file:
        print("="*80)
        print("Processing:", filename)
        print("-"*80)
        atoms = ase.io.read(filename)

        rotations, translations, equiv_atoms = get_symmetry_operations(atoms, symprec=args.symprec, verbose=args.verbose)

        symops = [(r,t) for (r, t) in zip(rotations, translations)]

        if args.verbose:
            print("Symmetry operations:")
            for (i, (r, t)) in enumerate(symops):
                sym_type = ""
                if (abs(np.trace(r)+3) < args.tol):
                    sym_type = "inversion"
                if (abs(np.trace(r)-3) < args.tol and np.linalg.norm(t) > args.tol):
                    sym_type = "translation"

                print("{}: {}".format(i+1, sym_type))
                print(r)
                print(t)

        # for convenience, we will create an auxiliary file that the user can
        # use to assign spins while examining the file in some visualizer
        aux_filename = filename+"_amcheck.vasp"
        print()
        print("Writing the used structure to auxiliary file: check {}.".format(
            aux_filename))
        ase.io.vasp.write_vasp(aux_filename, atoms, direct=True)

        # get spins from user's input
        chemical_symbols = atoms.get_chemical_symbols()
        spins = ['n' for i in range(len(chemical_symbols))]
        for u in np.unique(equiv_atoms):
            atom_ids = np.where(equiv_atoms == u)[0]
            positions = atoms.get_scaled_positions()[atom_ids]
            print()
            print("Orbit of {} atoms at positions:".format(
                chemical_symbols[atom_ids[0]]))
            for (i, j, p) in zip(atom_ids, range(1, len(atom_ids)+1), positions):
                print(i+1, "({})".format(j), p)

            if len(positions) == 1:
                print("Only one atom in the orbit: skipping.")
                continue

            orbit_spins = input_spins(len(positions))
            for i in range(len(orbit_spins)):
                spins[atom_ids[i]] = orbit_spins[i]
        
        is_am = is_altermagnet(symops, atoms.get_scaled_positions(),
                                   equiv_atoms, chemical_symbols, spins,
                                   args.tol, args.verbose, False)

        if is_am:
            print("The structure is an altermagnet.")
        else:
            print("The structure is not an altermagnet. Stopping.")
            break

        for u in np.unique(equiv_atoms):
            mag_atoms = reduce_atoms_to_subset(atoms, equiv_atoms, u)
            atom_ids = np.where(equiv_atoms == u)[0]
            if len(mag_atoms) <= 1:
                continue
            if spins[atom_ids[0]] == 'n':
                continue
            
            bondOrbits = generate_bond_orbits_to_cutoff(mag_atoms, rotations, translations, cutoff=args.cutoff)
            mag_rotations,mag_translations, _ = get_symmetry_operations(mag_atoms, symprec=args.symprec, verbose=False, fullSearch=True)
            fmag_rotations,fmag_translations, _ = get_symmetry_operations(mag_atoms, symprec=args.symprec, verbose=False, fullSearch=False)
            #print(np.round(mag_rotations - fmag_rotations, 2))
            for i in range(len(mag_translations)):
                mag_translations[i] = np.mod(np.round(mag_translations[i],4),1)
            


            equiv_mag_atoms = np.zeros(len(mag_atoms),dtype=int)
            chemical_symbols = mag_atoms.get_chemical_symbols()
            mag_spins = [spins[i] for i in np.where(equiv_atoms == u)[0]]

            for i, bondOrbit in enumerate(bondOrbits):
                f_rotations,f_translations = reduce_symmetry_to_bond(mag_rotations,mag_translations,bondOrbit)
                
                sym_ops = [(r,t) for r,t in zip(f_rotations,f_translations)]
                if is_altermagnet(sym_ops, mag_atoms.get_scaled_positions(), equiv_mag_atoms, chemical_symbols, mag_spins, silent=True, verbose=False):
                    print(f"Bond {i+1} {bondOrbit[0]} at distance {round(bondOrbit.get_bond_distance(), 4)} A causes altermagnetic splitting")



def cli(args=None):
    if not args:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog='amcheck',
        description='A tool to check if a given material is an altermagnet.')
    #parser.add_argument('--version', action='version',
    #                    version='%(prog)s {version}'.format(version=amcheck.__version__))
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="verbosely list the information during the execution")
    parser.add_argument('file', nargs='+',
                        help="name of the structure file to analyze")

    parser.add_argument('-s', '--symprec', default=DEFAULT_TOLERANCE, type=float,
                        help="tolerance spglib uses during the symmetry analysis")
    parser.add_argument('-ms', '--mag_symprec', default=-1.0, type=float,
                        help="tolerance for magnetic moments spglib uses during the magnetic symmetry analysis")
    parser.add_argument('-t', '--tol', '--tolerance', default=DEFAULT_TOLERANCE, type=float,
                        help="tolerance for internal numerical checks")

    parser.add_argument('-c', '--cutoff', default=10.0, type=float,
                        help="cutoff distance in Angstrom to consider bonds between magnetic atoms")

    args = parser.parse_args()

    main(args) 


if __name__ == "__main__":
    cli()
