import typing as t
from typing import overload
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import numpy.typing as npt
from pyvista import PolyData
from collections import namedtuple
from torch import multinomial
from torch.distributions import Dirichlet
import torchvision.transforms.v2 as T
import einops
import pyvista as pv


class TextureMapper:
    def __call__(self, verts: Tensor, image: Tensor) -> Tensor:
        # Create point cloud and compute uv coordinates using plane mapping
        point_cloud: PolyData = PolyData(verts.numpy())
        point_cloud.texture_map_to_plane(inplace=True)

        # Compute UV coordinates from [-1, -1] to [1, 1] corners
        uv: Tensor = 2 * torch.tensor(point_cloud.active_texture_coordinates) - 1

        # Normalize image values in [0., 1.]
        norm_image: Tensor = einops.rearrange(image, 'H W C -> 1 C H W')
        norm_image = T.functional.to_dtype(norm_image, scale=True)

        # Extract uv interpolated values from norm_image
        index: Tensor = einops.rearrange(uv, 'N C -> N 1 1 1 C')
        rgb: Tensor = torch.cat([torch.nn.functional.grid_sample(norm_image, coords, align_corners=True) for coords in list(index)], dim=0)
        rgb = einops.rearrange(rgb, 'N C 1 1 -> N C')
        return rgb


class MeshSampler:
    """Sample points on a 3d mesh uniformly."""

    def __init__(
        self,
        verts: Tensor,
        faces: Tensor,
        device:torch.device = torch.device('cpu'),
    ) -> None:
        super(MeshSampler, self).__init__()
        self.verts: Tensor = verts
        self.faces: Tensor = faces
        self.device: torch.device = device
        self.dirichlet = Dirichlet(concentration=torch.ones(3))

    def sample(self, n: int, indices: bool=True) -> t.Tuple[Tensor, Tensor, Tensor]:
        # Sample n triangle mesh faces without replacement
        sample_verts_idx: Tensor = multinomial(torch.ones(self.faces.size(0)), num_samples=n, replacement=False)
        sample_faces: Tensor = self.faces[sample_verts_idx]

        # Sample barycentric coordinates for each triangle mesh
        coefs: Tensor = self.dirichlet.sample(torch.Size([n])).type(torch.float32)

        # Compute R^3 coordinates based on barycentric interpolation
        pos: Tensor = self.verts.gather(dim=0, index=einops.repeat(sample_faces[:, 1:], 'N T -> (N T) C', C=self.verts.size(1)))
        pos = einops.rearrange(pos, '(N T) C -> N T C', N=sample_faces.size(0)).type(torch.float32)
        pos = torch.einsum('NT,NTC->NC', coefs, pos)

        sample_faces = sample_faces if indices is False else self.faces[sample_verts_idx][:, 1:]
        return pos, sample_faces, coefs

    @staticmethod
    def from_poly(poly: PolyData, *args, **kwargs) -> 'MeshSampler':
        verts: Tensor = torch.tensor(poly.points)
        faces: Tensor = torch.tensor(poly.faces.reshape((-1, 4)))[:, 1:]
        return MeshSampler(verts, faces, *args, **kwargs)


class LBOEmbedding:
    """
    Compute the first k eigenvectors accordin to the smallest eigenvalues for a 3d mesh using
    eigen-decomposition of LBO (symmetric normalized graph Laplacian).
    """

    def __init__(
        self,
        verts: Tensor,
        faces: Tensor,
        k_evecs: int=10,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        super(LBOEmbedding, self).__init__()

        # Eliminate unused vertices and offset faces
        self.verts, self.faces = remove_detached_verts(verts, faces)

        # Compute symmetric normalized graph Laplacian
        A: Tensor = self.adjacency_matrix(self.verts, self.faces[:, 1:])
        D: Tensor = self.degree_matrix(A)
        L: Tensor = self.sym_norm_graph_laplacian(D.to(device), A.to(device))

        # Compute the first k eigenpairs in ascending order
        _, self.evecs = torch.lobpcg(L, k=k_evecs, largest=False, method='ortho')
        self.evecs = torch.nn.functional.normalize(self.evecs, p=2, dim=0).cpu()

    def __call__(self, verts: Tensor, bary: Tensor | None = None) -> Tensor:
        bary = torch.full((verts.size(0), 3), 1/3) if bary is None else bary
        verts = einops.repeat(verts, 'V -> V C', C=3) if verts.ndim == 1 else verts
        assert verts.size(0) == bary.size(0), 'size mismatch between verts and barycentric values'
        assert torch.all(einops.reduce(bary, 'B C -> B', 'sum') - 1 < 1e-6), 'barycentric values do not sum to 1'

        # Select values from each eigenvector for each given vertex (and the neighbors)
        index: Tensor = einops.repeat(verts, 'B C -> C B K', K=self.evecs.size(1))
        evecs: Tensor = einops.repeat(self.evecs, 'V K -> C V K', C=verts.size(1))
        eigen: Tensor = torch.gather(input=evecs, dim=1, index=index)
        eigen = einops.rearrange(eigen, 'C V K -> V C K')

        # Compute baricentric interpolation using eigenvector values
        eigen = torch.einsum('VCK,VC->VK', eigen, bary)

        # Use sqrt(N) as factor
        coefs: Tensor = torch.sqrt(torch.tensor(self.verts.size(0)))
        eigen = coefs * eigen
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
