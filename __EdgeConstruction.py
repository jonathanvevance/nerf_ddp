import torch

MAX_BONDS = 6

bonds = [
    [1, 2, 2, 3, 0, 0],
    [0, 1, 1, 1, 1, 1],
    [0, 0, 2, 2, 2, 2],
    [0, 3, 3, 3, 3, 3],
]
# CH3 C 0 0H
#  1  0 2 3


N = len(bonds) # no of atoms

bonds = torch.Tensor(bonds).int().view(1, len(bonds), len(bonds[0]))

eye = torch.eye(N)
msk = torch.einsum("bl,bk->blk", torch.ones((1, N)), torch.ones((1,N))) # torch.ones = mask of active atoms

Ep = torch.index_select(eye, dim=0, index=bonds.reshape(-1)).view(1, N, MAX_BONDS, N).sum(dim=2) * msk

print(Ep)
# tensor([[[2., 1., 2., 1.],
#          [1., 5., 0., 0.],
#          [2., 0., 4., 0.],
#          [1., 0., 0., 5.]]])

