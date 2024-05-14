import typing as t
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import numpy.typing as npt
from pyvista import PolyData
from collections import namedtuple
import einops


class LBOEmbedding:
    def __init__(
        self,
        verts: Tensor,
        faces: Tensor,
        k_evecs: int,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        super(LBOEmbedding, self).__init__()

        # Eliminate unused vertices and offset faces
        self.verts, self.faces = self.remove_detached_verts(verts, faces)

        # Compute symmetric normalized graph Laplacian
        self.A: Tensor = self.adjacency_matrix(self.verts, self.faces[:, 1:]).to(device)
        self.D: Tensor = self.degree_matrix(self.A).to(device)
        self.L: Tensor = self.sym_norm_graph_laplacian(self.D, self.A)

        # Compute the first k eigenpairs in ascending order
        self.evals, self.evecs = torch.lobpcg(self.L, k=k_evecs, largest=False, method='ortho')
        self.evecs = torch.nn.functional.normalize(self.evecs, p=2, dim=0)

        # Move data back to CPU to avoid OOM
        self.A = self.A.cpu()
        self.D = self.D.cpu()
        self.L = self.L.cpu()
        self.evals: Tensor = self.evals.cpu()
        self.evecs: Tensor = self.evecs.cpu()

    def __call__(self, verts: Tensor, bary: Tensor | None = None) -> Tensor:
        bary = torch.full((verts.size(0), 3), 1/3) if bary is None else bary
        assert verts.size(0) == bary.size(0), 'size mismatch between verts and barycentric values'
        assert torch.all(einops.reduce(bary, 'B C -> B', 'sum') - 1 < 1e-6), 'barycentric values do not sum to 1'

        # Select values from each eigenvector for each given vertex (and the neighbors)
        index: Tensor = einops.repeat(verts, 'B C -> B C K', K=self.evecs.size(1))
        evecs: Tensor = einops.repeat(self.evecs, 'V K -> C V K', C=verts.size(1))
        print(evecs.shape, index.shape)
        eigen: Tensor = torch.gather(input=evecs, dim=-2, index=index)

        # Compute baricentric interpolation using eigenvector values
        eigen = torch.einsum('VCK,VC->VK', eigen, bary)

        # Use sqrt(N) as factor
        coef: Tensor = torch.sqrt(torch.tensor(self.verts.size(0)))
        eigen = coef * eigen
        return eigen

    @staticmethod
    def sym_norm_graph_laplacian(D: Tensor, A: Tensor) -> Tensor:
        D_inv_half: Tensor = torch.diag(D.diag() ** (-1/2)).to_sparse()
        return D_inv_half @ (D - A).to_sparse() @ D_inv_half

    @staticmethod
    def adjacency_matrix(verts: Tensor, faces: Tensor) -> Tensor:
        matrix: Tensor = torch.zeros(size=(verts.size(0), verts.size(0)))
        matrix[faces[:, 0], faces[:, 1]] = 1
        matrix[faces[:, 1], faces[:, 0]] = 1
        matrix[faces[:, 0], faces[:, 2]] = 1
        matrix[faces[:, 2], faces[:, 0]] = 1
        matrix[faces[:, 1], faces[:, 2]] = 1
        matrix[faces[:, 2], faces[:, 1]] = 1
        return matrix

    @staticmethod
    def degree_matrix(adjacency_matrix: Tensor) -> Tensor:
        matrix: Tensor = torch.sum(adjacency_matrix, dim=1)
        matrix = torch.diag(matrix, diagonal=0)
        return matrix

    @staticmethod
    def from_poly(poly: PolyData, *args, **kwargs) -> 'LBOEmbedding':
        verts: Tensor = torch.tensor(poly.points)
        faces: Tensor = torch.tensor(poly.faces.reshape((-1, 4)))
        return LBOEmbedding(verts, faces, *args, **kwargs)

    @staticmethod
    def remove_detached_verts(verts: Tensor, faces: Tensor) -> t.Tuple[Tensor, Tensor]:
        # Eliminate vertices that don't appear in faces
        verts_idx: Tensor = torch.arange(verts.size(0))
        verts_idx_inuse: Tensor = torch.unique(faces.flatten())
        verts_idx_inuse: Tensor = torch.isin(verts_idx, verts_idx_inuse)
        verts_inuse: Tensor = verts[verts_idx_inuse, :]

        # Offset vertex indices in each face to account for removal
        offsets: Tensor = (~verts_idx_inuse).cumsum(0)
        offset = np.vectorize(lambda x: x - offsets[x])
        faces_inuse = faces.numpy()
        faces_inuse[:, 1:] = np.apply_along_axis(offset, axis=0, arr=faces_inuse[:, 1:])
        faces_inuse = torch.from_numpy(faces_inuse)

        # Give back data in the same order
        return verts_inuse, faces_inuse
