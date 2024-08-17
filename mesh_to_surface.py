### Julian Schilliger - ThaumatoAnakalyptor - 2024

import os
import tempfile
import numpy as np
import nrrd
import tifffile
import open3d as o3d
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import BasePredictionWriter

class MyPredictionWriter(BasePredictionWriter):
  def __init__(self, save_path, image_size, r):
    super().__init__(write_interval="batch")  # or "epoch" for end of an epoch

  def write_on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, prediction, batch_indices, batch, batch_idx: int, dataloader_idx: int):
    # print(prediction)
    print("End predict")

class MeshDataset(Dataset):
  def __init__(self, path, scroll, r=32):
    self.path = path
    self.scroll = scroll
    self.grid_size = 500
    self.r = r+1

    self.load_mesh(path)
    self.grids_to_process = [(3513, 1900, 3398)]

    working_path = os.path.dirname(path)
    write_path = os.path.join(working_path, "layers")
    self.writer = MyPredictionWriter(write_path, self.image_size, r)

  def get_writer(self):
    return self.writer

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

  def __len__(self):
    return len(self.grids_to_process)

  def __getitem__(self, idx):
    grid_index = self.grids_to_process[idx]
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
    grid_coord = torch.tensor(np.array(grid_index), dtype=torch.int32)

    return grid_coord, grid_cell_tensor, vertices_tensor, normals_tensor, uv_tensor

class PPMAndTextureModel(pl.LightningModule):
  def __init__(self, r: int = 32, max_side_triangle: int = 10):
    print("instantiating model")
    self.r = r
    self.max_side_triangle = max_side_triangle
    self.new_order = [2,1,0] # [2,1,0], [2,0,1], [0,2,1], [0,1,2], [1,2,0], [1,0,2]
    super().__init__()

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
    # origin
    # grid_coords: B x 3, grid_cell: B x W x W x W, vertices: B x T x 3 x 3, normals: B x T x 3 x 3, uv_coords_triangles: B x T x 3 x 2

    # grid_coords: T x 3, grid_cell: B x W x W x W, vertices: T x 3 x 3, normals: T x 3 x 3, uv_coords_triangles: T x 3 x 2
    grid_coords, grid_cells, vertices, normals, uv_coords_triangles = x

    # Step 1: Compute AABBs for each triangle (only starting points of AABB rectangles)
    min_uv, _ = torch.min(uv_coords_triangles, dim=1)
    # Floor and ceil the UV coordinates
    min_uv = torch.floor(min_uv)

    nr_triangles = vertices.shape[0]
    max_triangles_per_loop = 5000
    values_list = []
    grid_points_list = []
    for i in range(0, nr_triangles, max_triangles_per_loop):
      min_uv_ = min_uv[i:i+max_triangles_per_loop]
      grid_coords_ = grid_coords[i:i+max_triangles_per_loop]
      vertices_ = vertices[i:i+max_triangles_per_loop]
      normals_ = normals[i:i+max_triangles_per_loop]
      uv_coords_triangles_ = uv_coords_triangles[i:i+max_triangles_per_loop]

      # Step 2: Generate Meshgrids for All Triangles
      # create grid points tensor: T x W*H x 2
      grid_points = self.create_grid_points_tensor(min_uv_, self.max_side_triangle, self.max_side_triangle)
      del min_uv_

    return x

# Custom collation function
def custom_collate_fn(batch):
  try:
    # Initialize containers for the aggregated items
    grid_cells = []
    vertices = []
    normals = []
    uv_coords_triangles = []
    grid_coords = []
    
    # Loop through each batch and aggregate its items
    for i, items in enumerate(batch):
      if items is None:
        continue
      grid_coord, grid_cell, vertice, normal, uv_coords_triangle = items
      if grid_cell is None:
        continue
      if len(grid_cell) == 0:
        continue
      if grid_cell.size()[0] == 0:
        continue
      grid_cells.append(grid_cell)
      vertices.append(vertice)
      normals.append(normal)
      uv_coords_triangles.append(uv_coords_triangle)
      grid_coord = grid_coord.unsqueeze(0).expand(vertice.shape[0], -1)
      grid_coords.extend(grid_coord)
            
    if len(grid_cells) == 0:
      return None, None, None, None, None, None
            
    # Turn the lists into tensors
    grid_cells = torch.stack(grid_cells, dim=0)
    vertices = torch.concat(vertices, dim=0)
    normals = torch.concat(normals, dim=0)
    uv_coords_triangles = torch.concat(uv_coords_triangles, dim=0)
    grid_coords = torch.stack(grid_coords, dim=0)
    
    # Return a single batch containing all aggregated items
    return grid_coords, grid_cells, vertices, normals, uv_coords_triangles
  except:
    return None, None, None, None, None, None

def ppm_and_texture(obj_path, scroll):
  scroll_format = "grid cells"

  # Initialize the dataset and dataloader
  dataset = MeshDataset(obj_path, scroll)
  dataloader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn, shuffle=False, num_workers=0)
  # dataloader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn, shuffle=False, num_workers=1, prefetch_factor=3)
  model = PPMAndTextureModel()

  writer = dataset.get_writer()
  trainer = pl.Trainer(callbacks=[writer], strategy="ddp", logger=False)
  # trainer = pl.Trainer(callbacks=[writer], accelerator='gpu', devices=int(gpus), strategy="ddp")

  trainer.predict(model, dataloaders=dataloader, return_predictions=False)

if __name__ == '__main__':
  obj = '../ink-explorer/cubes/03513_01900_03398/03513_01900_03398_20230702185753.obj'
  scroll = '../ink-explorer/cubes/03513_01900_03398/03513_01900_03398_volume.tiff'

  ppm_and_texture(obj, scroll=scroll)
