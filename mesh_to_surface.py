### Julian Schilliger - ThaumatoAnakalyptor - 2024

import numpy as np
import argparse
import nrrd
import tifffile
import open3d as o3d

class MeshDataset():
  def __init__(self, path, scroll):
    self.path = path
    self.load_mesh(path)

  def load_mesh(self, path):
    """Load the mesh from the given path and extract the vertices, normals, triangles, and UV coordinates."""
    mesh = o3d.io.read_triangle_mesh(path)
    self.mesh = mesh
    print(f"Loaded mesh from {path}", end="\n")

    self.vertices = np.asarray(self.mesh.vertices)
    self.normals = np.asarray(self.mesh.vertex_normals)
    self.triangles = np.asarray(self.mesh.triangles)
    self.uv = np.asarray(self.mesh.triangle_uvs)

def ppm_and_texture(obj_path, scroll):
  scroll_format = "grid cells"

  # Initialize the dataset and dataloader
  dataset = MeshDataset(obj_path, scroll)

if __name__ == '__main__':
  obj = '../ink-explorer/cubes/03513_01900_03398/03513_01900_03398_20230702185753.obj'
  scroll = '../ink-explorer/cubes/03513_01900_03398/03513_01900_03398_volume.tif'

  ppm_and_texture(obj, scroll=scroll)

