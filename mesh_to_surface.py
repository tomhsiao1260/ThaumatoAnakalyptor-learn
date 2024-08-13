### Julian Schilliger - ThaumatoAnakalyptor - 2024

import numpy as np
import argparse
import nrrd
import tifffile
import open3d as o3d
import torch
from torch.utils.data import Dataset

class PPMAndTextureModel():
  def __init__(self, max_side_triangle: int = 10):
    print("instantiating model")
    self.max_side_triangle = max_side_triangle
    self.new_order = [2,1,0]

  def ppm(self, pts, tri):
    # pts T x W*H x 2
    # tri_pts T x 3 x 2
    # triangles 3
    v0 = tri[:, 2, :].unsqueeze(1) - tri[:, 0, :].unsqueeze(1)
    v1 = tri[:, 1, :].unsqueeze(1) - tri[:, 0, :].unsqueeze(1)
    v2 = pts - tri[:, 0, :].unsqueeze(1)

    dot00 = v0.pow(2).sum(dim=2)
    dot01 = (v0 * v1).sum(dim=2)
    dot11 = v1.pow(2).sum(dim=2)
    dot02 = (v2 * v0).sum(dim=2)
    dot12 = (v2 * v1).sum(dim=2)

    invDenom = 1 / (dot00 * dot11 - dot01.pow(2))
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    is_inside = (u >= 0) & (v >= 0) & ((u + v) <= 1 )

    w = 1 - u - v

    bary_coords = torch.stack([u, v, w], dim=2)
    bary_coords = normalize(bary_coords, p=1, dim=2)

    return bary_coords, is_inside

  def create_grid_points_tensor(self, starting_points, w, h):
    device = starting_points.device
    n = starting_points.shape[0]
        
    # Generate all combinations of offsets in the grid
    dx = torch.arange(w, device=device)  # Shape (w,)
    dy = torch.arange(h, device=device)  # Shape (h,)
        
    # Create a meshgrid from dx and dy
    mesh_dx, mesh_dy = torch.meshgrid(dx, dy, indexing='xy')  # Shapes (h, w)
        
    # Stack and reshape to create the complete offset grid
    offset_grid = torch.stack((mesh_dx, mesh_dy), dim=2).view(-1, 2)  # Shape (w*h, 2)
        
    # Expand starting points for broadcasting
    starting_points_expanded = starting_points.view(n, 1, 2)  # Shape (n, 1, 2)
        
    # Add starting points to offset grid (broadcasting in action)
    grid_points = starting_points_expanded + offset_grid  # Shape (n, w*h, 2)

    return grid_points

  def forward(self, x):
    # grid_cell: B x W x W x W, vertices: T x 3 x 3, normals: T x 3 x 3, uv_coords_triangles: T x 3 x 2
    grid_cells, vertices, normals, uv_coords_triangles = x

    # Step 1: Compute AABBs for each triangle (only starting points of AABB rectangles)
    min_uv, _ = torch.min(uv_coords_triangles, dim=1)
    # Floor and ceil the UV coordinates
    min_uv = torch.floor(min_uv)

    nr_triangles = vertices.shape[0]
    max_triangles_per_loop = 5000
    values_list = []
    grid_points_list = []
    for i in range(0, 1):
    # for i in range(0, nr_triangles, max_triangles_per_loop):
      min_uv_ = min_uv[i:i+max_triangles_per_loop]
      vertices_ = vertices[i:i+max_triangles_per_loop]
      normals_ = normals[i:i+max_triangles_per_loop]
      uv_coords_triangles_ = uv_coords_triangles[i:i+max_triangles_per_loop]

      # Step 2: Generate Meshgrids for All Triangles
      # create grid points tensor: T x W*H x 2
      grid_points = self.create_grid_points_tensor(min_uv_, self.max_side_triangle, self.max_side_triangle)
      del min_uv_

      # Step 3: Compute Barycentric Coordinates for all Triangles grid_points
      # baryicentric_coords: T x W*H x 3, is_inside: T x W*H
      baryicentric_coords, is_inside = self.ppm(grid_points, uv_coords_triangles_)
      grid_points = grid_points[is_inside] # S x 2

      # adjust to new_order
      vertices_ = vertices_[:, self.new_order, :]
      normals_ = normals_[:, self.new_order, :]

      # vertices: T x 3 x 3, normals: T x 3 x 3, baryicentric_coords: T x W*H x 3
      coords = torch.einsum('ijk,isj->isk', vertices_, baryicentric_coords).squeeze()
      norms = torch.einsum('ijk,isj->isk', normals_, baryicentric_coords).squeeze()
      # Handle case where T == 1 by ensuring dimensions are not squeezed away
      if coords.dim() == 2:
        coords = coords.unsqueeze(0)
      if norms.dim() == 2:
        norms = norms.unsqueeze(0)
      norms = normalize(norms,dim=2)
      del vertices_, normals_, uv_coords_triangles_

      # broadcast grid_coords to T x W*H x 3 -> S x 3
      grid_coords_ = grid_coords_.unsqueeze(-2).expand(-1, baryicentric_coords.shape[1], -1)
      del baryicentric_coords, grid_coords_

      # coords: S x 3, norms: S x 3
      coords = coords[is_inside]
      norms = norms[is_inside]

      # Poper axis order
      coords = coords[:, self.new_order]
      norms = norms[:, self.new_order]

class MeshDataset(Dataset):
  def __init__(self, path, scroll):
    self.path = path
    self.scroll = scroll
    self.grid_size = 768

    self.load_mesh(path)

  def load_mesh(self, path):
    """Load the mesh from the given path and extract the vertices, normals, triangles, and UV coordinates."""
    mesh = o3d.io.read_triangle_mesh(path)
    self.mesh = mesh

    y_size, x_size = 1000, 1000

    self.vertices = np.asarray(self.mesh.vertices)
    self.normals = np.asarray(self.mesh.vertex_normals)
    self.triangles = np.asarray(self.mesh.triangles)
    uv = np.asarray(self.mesh.triangle_uvs).reshape(-1, 3, 2)
    # scale numpy UV coordinates to the image size
    self.uv = uv * np.array([y_size, x_size])
    self.image_size = (y_size, x_size)

    # vertices of triangles
    self.triangles_vertices = self.vertices[self.triangles]
    self.triangles_normals = self.normals[self.triangles]

  def load_grid(self, path):
    with tifffile.TiffFile(path) as tif:
      grid_cell = tif.asarray()
    return grid_cell

  def __getitem__(self, idx):
    vertices = self.triangles_vertices
    normals = self.triangles_normals
    uv = self.uv

    # load grid cell from disk
    grid_cell = self.load_grid(self.scroll)
    grid_cell = grid_cell.astype(np.float32)

    # Convert NumPy arrays to PyTorch tensors
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
    normals_tensor = torch.tensor(normals, dtype=torch.float32)
    uv_tensor = torch.tensor(uv, dtype=torch.float32)
    grid_cell_tensor = torch.tensor(grid_cell, dtype=torch.float32)

    return grid_cell_tensor, vertices_tensor, normals_tensor, uv_tensor

def ppm_and_texture(obj_path, scroll):
  scroll_format = "grid cells"

  # Initialize the dataset and dataloader
  dataset = MeshDataset(obj_path, scroll)
  model = PPMAndTextureModel()
  model.forward(dataset[0])

if __name__ == '__main__':
  obj = '../ink-explorer/cubes/03513_01900_03398/03513_01900_03398_20230702185753.obj'
  scroll = '../ink-explorer/cubes/03513_01900_03398/03513_01900_03398_volume.tiff'

  ppm_and_texture(obj, scroll=scroll)

