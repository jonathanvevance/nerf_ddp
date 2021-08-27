import torch

MAX_BONDS = 6

#! LHS = N202
#! N-0, N-1, O-2, O-3
src_bonds = [
    [1, 2, 2, 0, 0, 0],
    [0, 3, 3, 1, 1, 1],
    [0, 0, 2, 2, 2, 2],
    [1, 1, 3, 3, 3, 3],
]
N = len(src_bonds) # no of atoms
src_mask = torch.Tensor([0, 0, 0, 0]).view(1, N) # 1 iff masked
pad_mask = 1 - src_mask.float()
pad_mask = torch.einsum("bl,bk->blk", pad_mask, pad_mask)
src_bonds = torch.Tensor(src_bonds).int().view(1, len(src_bonds), len(src_bonds[0]))

#! RHS = N2 + O2
#! N-0, N-1, O-2, O-3
# tgt_bonds = [
#     [1, 1, 1, 0, 0, 0],
#     [0, 0, 0, 1, 1, 1],
#     [3, 3, 2, 2, 2, 2],
#     [2, 2, 3, 3, 3, 3],
# ]
# N = len(tgt_bonds) # no of atoms
# tgt_mask = torch.Tensor([0, 0, 0, 0]).view(1, N) # 1 iff masked

#! RHS = N2 (O2 missing)
#! N-0, N-1, O-2, O-3
tgt_bonds = [
    # [1, 1, 1, 0, 0, 0],
    # [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [3, 3, 2, 2, 2, 2],
    [2, 2, 3, 3, 3, 3],
]
N = len(tgt_bonds) # no of atoms
# tgt_mask = torch.Tensor([0, 0, 1, 1]).view(1, N) # 1 iff masked
tgt_mask = torch.Tensor([1, 1, 0, 0]).view(1, N) # 1 iff masked


tgt_bonds = torch.Tensor(tgt_bonds).int().view(1, len(tgt_bonds), len(tgt_bonds[0]))
or_mask = 1 - torch.einsum("bl,bk->blk", tgt_mask, tgt_mask)
and_mask = torch.einsum("bl,bk->blk", 1-tgt_mask, 1-tgt_mask)

eye = torch.eye(N)
Ep = torch.index_select(eye, dim=0, index=tgt_bonds.reshape(-1)).view(1, N, MAX_BONDS, N).sum(dim=2)*and_mask

print(Ep)
print()

predEp = torch.Tensor([[
    [0., 0., 0., 0.],
    [0., 0., 0., 0.],
    [0., 0., 0., 0.],
    [0., 0., 0., 0.]
]])

error = Ep - predEp
error = error*error*pad_mask*or_mask
print(error)
