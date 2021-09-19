from rdkit import Chem
import pandas as pd
import numpy as np
import torch
import csv
import os
import pickle
import re
import pdb
from tqdm import tqdm, trange
from concurrent.futures import ProcessPoolExecutor
from rdkit import RDLogger
'''
aroma: [B, L]
e: [B, L]
b: [B, L, 4]
c: [B, L]
m: [B, L]
'''
MAX_BONDS = 6
MAX_DIFF = 4
prefix = "data"

def molecule(mols, src_len, reactant_mask = None, ranges = None):
    features = {}
    element = np.zeros(src_len, dtype='int32')
    aroma = np.zeros(src_len, dtype='int32')
    bonds = np.zeros((src_len, MAX_BONDS), dtype='int32') #! Note... 2D array
    charge = np.zeros(src_len, dtype='int32')

    reactant = np.zeros(src_len, dtype='int32') # 1 for reactant
    mask = np.ones(src_len, dtype='int32') # 1 for masked
    segment = np.zeros(src_len, dtype='int32')

    for molid, mol in enumerate(mols):
        for atom in mol.GetAtoms(): #! Note.. each atom has a different idx
            idx = atom.GetAtomMapNum()-1 #! atom idxes are 1-indexed in USPTO-MIT

            #! VERY IMPORTANT -
            #! the order of atoms (important for transformer depends on the mapping in the dataset.
            #! So it is upto the dataset annotators to map closeby atoms with closeby atom indices.

            segment[idx] = molid
            element[idx] = atom.GetAtomicNum()
            charge[idx] = atom.GetFormalCharge()
            mask[idx] = 0
            if reactant_mask:
                reactant[idx] = reactant_mask[molid]
            #! getting features of atoms in mol

            cnt = 0 #! tracks the TOTAL number of bonds from this atom
            for j, b in enumerate(atom.GetBonds()): # mark existence of bond first

                other = b.GetBeginAtomIdx() + b.GetEndAtomIdx() - atom.GetIdx() #! get other atom Idx
                other = mol.GetAtoms()[other].GetAtomMapNum() - 1 #! atom idx of other atom
                num_map = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'AROMATIC': 1}
                num = num_map[str(b.GetBondType())] #! int number of bonds (or 1 for aromatic)
                for k in range(num):
                    if cnt == MAX_BONDS: #! if TOTAL no. of bonds > MAX_BONDS, ignore this reaction (return None)
                        return None
                    bonds[idx][cnt] = other #! IMPORTANT = for idx atom, bond number 'cnt' is with this atom idx...
                    cnt += 1
                if str(b.GetBondType()) == 'AROMATIC':
                    aroma[idx] = 1
            tmp = bonds[idx][0:cnt]
            tmp.sort()
            bonds[idx][0:cnt] = tmp #! sort the atom idxes
            while cnt < MAX_BONDS:
                bonds[idx][cnt] = idx #! IMPORTANT = for others cnt values till MAX_BONDS, put lone pairs.
                cnt += 1

    #! testing if any atom (IN TGT) has zero bonds (all zeros, not even all bonds with itself)...
    # for row in bonds:
    #     if sum(row) == 0:
    #         print("\nATOM WITH NO BOND FOUND....")
    #         if reactant_mask is None:
    #             exit()

    features = {'element':element, 'bond':bonds, 'charge':charge, 'aroma':aroma, 'mask':mask, 'segment':segment, 'reactant': reactant}
    return features


def reaction(args):
    """ processes a reaction, returns dict of arrays"""
    src, tgt = args

    # #! Note...
    # print("\n\nSource is....")
    # print(src)
    # print("\n\nTarget is....")
    # print(tgt)

    pattern = re.compile(":(\d+)\]") # atom map numbers
    src_len = Chem.MolFromSmiles(src).GetNumAtoms()
    tgt_len = Chem.MolFromSmiles(tgt).GetNumAtoms() #! I added this... (tgt_len != src_len)

    # #! Note...
    # print("\nSource length = ", src_len)
    # print("Target length = ", tgt_len)

    # reactant mask
    src_mols = src.split('.')

    # #! Note...
    # print("\n#src_mols = ", len(src_mols))

    src_atoms = pattern.findall(src) #! I added this.. [this will be from 1 to some max number]
    tgt_atoms = pattern.findall(tgt) #! RHS side SHOULD BE MAPPED..

    # #! Note...
    # print("#src_atoms = ", src_atoms)
    # print("#tgt_atoms = ", tgt_atoms) #! len(tgt_atoms) = tgt_len
    assert(len(tgt_atoms) <= len(src_atoms))

    reactant_mask = [False for i in src_mols]
    for j, item in enumerate(src_mols):
        atoms = pattern.findall(item)
        for atom in atoms:
            if atom in tgt_atoms:
                reactant_mask[j] = True
                break

    #! Checking if there are repeated molecules in src or tgt
    # print()
    # if len(set(src_mols)) < len(src_mols):
    #     print("REPEATED MOLS FOUND IN SRC")
    #     exit()
    # if len(set(tgt.split("."))) < len(tgt.split(".")):
    #     print("REPEATED MOLS FOUND IN TGT")
    #     exit()

    # #! Note...
    # print("\nreactant_mask = ", reactant_mask) #! reactants that take part in rxn

    # the atom map num ranges of each molecule for segment mask
    src_mols = [Chem.MolFromSmiles(item) for item in src_mols]
    tgt_mols = [Chem.MolFromSmiles(item) for item in tgt.split(".")]

    # #! Note...
    # print("\n#src_mols = ", len(src_mols))
    # print("#tgt_mols = ", len(tgt_mols))

    ranges = []
    for mol in src_mols:
        lower = 999
        upper = 0
        for atom in mol.GetAtoms():
            lower = min(lower, atom.GetAtomMapNum()-1)
            upper = max(upper, atom.GetAtomMapNum())
        ranges.append((lower, upper)) #! For each src molecule, get (min_mapno-1, max_mapno)
        #! However, ranges is not used anywhere.... [even in molecule()]

    src_features = molecule(src_mols, src_len, reactant_mask, ranges)
    tgt_features = molecule(tgt_mols, src_len) #! length that is passed is src_len

    # #! Note...
    # print("\n#src_features = ", len(src_features))
    # print("#tgt_features = ", len(tgt_features)) #! both are of same length

    # #! Note...
    # print("\n#src_features.element = ", src_features["element"])
    # print("#tgt_features.element = ", tgt_features["element"])

    # #! Note...
    # print("\n#src_features.bond = ", src_features["bond"])
    # print("#tgt_features.bond = ", tgt_features["bond"])
    #! Bond = matrix src_len X MAX_BONDS, showing atom IDX of j-th bond of i-th atom.
    #! The atom IDXes of covalent bond neighbour atoms are sorted and the rest of the
    #! column elements (after all bonds are done) are lone pairs and self IDX is filled

    if not (src_features and tgt_features):
        return None

    #! This section is just to ignore any rxn with diff > MAX_DIFF bond changes (verify wording)
    src_bond = src_features['bond']
    tgt_bond = tgt_features['bond']
    bond_inc = np.zeros((src_len, MAX_DIFF), dtype='int32')
    bond_dec = np.zeros((src_len, MAX_DIFF), dtype='int32')
    for i in range(src_len):
        if tgt_features['mask'][i]: # 1 for masked
            continue
        inc_cnt = 0
        dec_cnt = 0
        diff = [0 for _ in range(src_len)]
        for j in range(MAX_BONDS):
            diff[tgt_bond[i][j]] += 1 #! for i-th atom, diff of j-th bond in RHS atom increased
            diff[src_bond[i][j]] -= 1 #! for i-th atom, diff of j-th bond in LHS atom decreased
        for j in range(src_len):
            if diff[j] > 0:
                if inc_cnt + diff[j] >MAX_DIFF:
                    return None
                bond_inc[i][inc_cnt:inc_cnt+diff[j]] = j
                inc_cnt += diff[j]
            if diff[j] < 0:
                bond_dec[i][dec_cnt:dec_cnt-diff[j]] = j
                dec_cnt -= diff[j]
        assert inc_cnt == dec_cnt

    item = {}
    for key in src_features:
        if key in ["element", "reactant"]:
            item[key] = src_features[key]
        else:
            item['src_' + key] = src_features[key]
            item['tgt_' + key] = tgt_features[key]

    return item


def process(name):
    tgt = []
    src = []
    with open(name + ".txt") as file:
        for i, line in enumerate(file):

            # if i == 0:
            #     continue #! Skip first

            rxn = line.split()[0].split('>>')
            src.append(rxn[0])
            tgt.append(rxn[1])

            # break # only one reaction #! Note..

    # pool = ProcessPoolExecutor(10) #! Skip this
    dataset = []
    batch_size = 2048
    for i in trange(len(src)//batch_size+1):
        upper = min((i+1)*batch_size, len(src))
        arg_list = [(src[idx], tgt[idx]) for idx in range(i*batch_size, upper)]
        # result = pool.map(reaction, arg_list, chunksize= 64) #! Skip this
        # result = list(result)

        # for item in result:
        #     if not item is None:
        #         #! each 'item' is a dictionary
        #         dataset += [item]

        for args in arg_list:
            item = reaction(args)
            if not item is None:
                #! each 'item' is a dictionary
                dataset += [item]

    print(len(dataset))
    # pool.shutdown() #! Skip this
    # return

    #! dataset = list of single dicts corresponding to reactions
    with open(name +"_"+prefix+ '.pickle', 'wb') as file:
        pickle.dump(dataset, file)
    print("total %d, legal %d"%(len(src), len(dataset)))
    print(name, 'file saved.')

if __name__ =='__main__':
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    RDLogger.DisableLog('rdApp.info')
    process("data/valid")
    process("data/test")
    process("data/train")
