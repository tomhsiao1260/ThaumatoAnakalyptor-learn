### Julian Schilliger - ThaumatoAnakalyptor - 2024

import numpy as np
import argparse
import nrrd
import tifffile
import open3d as o3d
import torch
from torch.utils.data import Dataset

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

    self.vertices = np.asarray(self.mesh.vertices)
    self.normals = np.asarray(self.mesh.vertex_normals)
    self.triangles = np.asarray(self.mesh.triangles)
    self.uv = np.asarray(self.mesh.triangle_uvs)

    # vertices of triangles
    self.triangles_vertices = self.vertices[self.triangles]
    self.triangles_normals = self.normals[self.triangles]

  def load_grid(self, path):
    with tifffile.TiffFile(path) as tif:
      return grid_cell = tif.asarray()

  def __getitem__(self, idx):
      vertices = self.triangles_vertices
      normals = self.triangles_normals

      # load grid cell from disk
      grid_cell = self.load_grid(self.scroll)
      grid_cell = grid_cell.astype(np.float32)

      # Convert NumPy arrays to PyTorch tensors
      vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
      normals_tensor = torch.tensor(normals, dtype=torch.float32)
      grid_cell_tensor = torch.tensor(grid_cell, dtype=torch.float32)

      return grid_cell_tensor, vertices_tensor, normals_tensor

def ppm_and_texture(obj_path, scroll):
  scroll_format = "grid cells"

  # Initialize the dataset and dataloader
  dataset = MeshDataset(obj_path, scroll)
  dataset[0]

if __name__ == '__main__':
  obj = '../ink-explorer/cubes/03513_01900_03398/03513_01900_03398_20230702185753.obj'
  scroll = '../ink-explorer/cubes/03513_01900_03398/03513_01900_03398_volume.tiff'

  ppm_and_texture(obj, scroll=scroll)

