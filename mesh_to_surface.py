### Julian Schilliger - ThaumatoAnakalyptor - 2024

import os
import tempfile
import numpy as np
from tqdm import tqdm
import nrrd
import tifffile
import open3d as o3d
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytorch_lightning as pl
from torch.nn.functional import normalize
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import BasePredictionWriter
from rendering_utils.interpolate_image_3d import extract_from_image_4d
from multiprocessing import cpu_count, shared_memory

class MyPredictionWriter(BasePredictionWriter):
  def __init__(self, save_path, image_size, r):
    super().__init__(write_interval="batch")  # or "epoch" for end of an epoch
    self.save_path = save_path
    self.num_workers = cpu_count()
    self.surface_volume_np = None
    self.r = r
    self.image_size = image_size

  def write_on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, prediction, batch_indices, batch, batch_idx: int, dataloader_idx: int):
    self.surface_volume_np, self.shm = self.create_shared_array((2*self.r+1, self.image_size[0], self.image_size[1]), np.uint16, name="surface_volume")
    # Gather the shared memory name
    torch.distributed.barrier()

    self.process_and_write_data(prediction)

    print("End predict")

  def create_shared_array(self, shape, dtype, name="shared_array"):
    array_size = np.prod(shape) * np.dtype(dtype).itemsize
    try:
      # Create a shared array
      shm = shared_memory.SharedMemory(create=True, size=array_size, name=name)
    except FileExistsError:
      print(f"Shared memory with name {name} already exists.")
      # Clean up the shared memory if it already exists
      shm = shared_memory.SharedMemory(create=False, size=array_size, name=name)

    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    arr.fill(0)  # Initialize the array with zeros
    return arr, shm

  def process_and_write_data(self, prediction):
    try:
      # print("Writing to Numpy")
      if prediction is None:
        return
      if len(prediction) == 0:
        return

      values, indexes_3d = prediction
            
      if indexes_3d.shape[0] == 0:
        return

      # save into surface_volume_np
      self.surface_volume_np[indexes_3d[:, 0], indexes_3d[:, 1], indexes_3d[:, 2]] = values
    except Exception as e:
      print(e)

  def write_tif(self):
    def save_tif(i, filename):
      image = self.surface_volume_np[i]
      image = image.T
      image = image[::-1, :]
      tifffile.imsave(filename, image)

    os.makedirs(self.save_path, exist_ok=True)

    with ThreadPoolExecutor(self.num_workers) as executor:
      futures = []
      for i in range(self.surface_volume_np.shape[0]):
        i_str = str(i).zfill(len(str(self.surface_volume_np.shape[0])))
        filename = os.path.join(self.save_path, f"{i_str}.tif")
        futures.append(executor.submit(save_tif, i, filename))

      # Wait for all futures to complete if needed
      for future in tqdm(as_completed(futures), desc="Writing TIF"):
        future.result()

    # Create Composite max image from all tifs
    composite_image = np.zeros((self.surface_volume_np.shape[1], self.surface_volume_np.shape[2]), dtype=np.float32)
    for i in range(self.surface_volume_np.shape[0]):
      composite_image = np.maximum(composite_image, self.surface_volume_np[i])

    composite_image = composite_image.astype(np.uint16)
    composite_image = composite_image.T
    composite_image = composite_image[::-1, :]
    tifffile.imsave(os.path.join(os.path.dirname(self.save_path), "composite.tif"), composite_image)

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
    # origin
    # grid_coords: B x 3, grid_cell: B x W x W x W, vertices: B x T x 3 x 3, normals: B x T x 3 x 3, uv_coords_triangles: B x T x 3 x 2

    # grid_coords: T x 3, grid_cell: B x W x W x W, vertices: T x 3 x 3, normals: T x 3 x 3, uv_coords_triangles: T x 3 x 2, grid_index: T
    grid_coords, grid_cells, vertices, normals, uv_coords_triangles, grid_index = x

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
      grid_index_ = grid_index[i:i+max_triangles_per_loop]

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

      # broadcast grid index to T x W*H -> S
      grid_index_ = grid_index_.unsqueeze(-1).expand(-1, baryicentric_coords.shape[1])
      grid_index_ = grid_index_[is_inside]
      # broadcast grid_coords to T x W*H x 3 -> S x 3
      grid_coords_ = grid_coords_.unsqueeze(-2).expand(-1, baryicentric_coords.shape[1], -1)
      coords = coords - grid_coords_ # Reorient coordinate system origin to 0 for extraction on grid_cells
      del baryicentric_coords, grid_coords_

      # coords: S x 3, norms: S x 3
      coords = coords[is_inside]
      norms = norms[is_inside]

      # Poper axis order
      coords = coords[:, self.new_order]
      norms = norms[:, self.new_order]

      # Step 4: Compute the 3D coordinates for every r slice
      r_arange = torch.arange(-self.r, self.r+1, device=coords.device).reshape(1, -1, 1)

      # coords: S x 2*r+1 x 3, grid_index: S x 2*r+1 x 1
      coords = coords.unsqueeze(-2).expand(-1, 2*self.r+1, -1) + r_arange * norms.unsqueeze(-2).expand(-1, 2*self.r+1, -1)
      grid_index_ = grid_index_.unsqueeze(-1).unsqueeze(-1).expand(-1, 2*self.r+1, -1)

      # Expand and add 3rd dimension to grid points
      r_arange = r_arange.expand(grid_points.shape[0], -1, -1) + self.r # [0 to 2*r]
      grid_points = grid_points.unsqueeze(-2).expand(-1, 2*self.r+1, -1)
      grid_points = torch.cat((grid_points, r_arange), dim=-1)
      del r_arange, is_inside

      # Step 5: Filter out the points that are outside the grid_cells
      mask_coords = (coords[:, :, 0] >= 0) & (coords[:, :, 0] < grid_cells.shape[1]) & (coords[:, :, 1] >= 0) & (coords[:, :, 1] < grid_cells.shape[2]) & (coords[:, :, 2] >= 0) & (coords[:, :, 2] < grid_cells.shape[3])

      # coords: S' x 3, norms: S' x 3
      coords = coords[mask_coords]
      grid_points = grid_points[mask_coords] # S' x 2
      grid_index_ = grid_index_[mask_coords] # S'

      # Step 5: Extract the values from the grid cells
      # grid_cells: T x W x H x D, coords: S' x 3, grid_index: S' x 1
      values = extract_from_image_4d(grid_cells, grid_index_, coords)
      del coords, mask_coords, grid_index_

      # Step 6: Return the 3D Surface Volume coordinates and the values
      values = values.reshape(-1)
      grid_points = grid_points.reshape(-1, 3) # grid_points: S' x 3
      
      # Empty the cache to free up memory
      torch.cuda.empty_cache()

      # reorder grid_points
      grid_points = grid_points[:, [2, 0, 1]]

      values_list.append(values)
      grid_points_list.append(grid_points)

    del grid_cells, grid_index, min_uv, vertices, normals, uv_coords_triangles
    # Empty the cache to free up memory
    torch.cuda.empty_cache()

    if len(values_list) == 0:
        return None, None
        
    values = torch.cat(values_list, dim=0)
    grid_points = torch.cat(grid_points_list, dim=0)

    values = values.cpu().numpy().astype(np.uint16)
    grid_points = grid_points.cpu().numpy().astype(np.int32)

    # Return the 3D Surface Volume coordinates and the values
    return values, grid_points

# Custom collation function
def custom_collate_fn(batch):
  try:
    # Initialize containers for the aggregated items
    grid_cells = []
    vertices = []
    normals = []
    uv_coords_triangles = []
    grid_index = []
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
      grid_index.extend([i]*vertice.shape[0])
      grid_coord = grid_coord.unsqueeze(0).expand(vertice.shape[0], -1)
      grid_coords.extend(grid_coord)
            
    if len(grid_cells) == 0:
      return None, None, None, None, None, None
            
    # Turn the lists into tensors
    grid_cells = torch.stack(grid_cells, dim=0)
    vertices = torch.concat(vertices, dim=0)
    normals = torch.concat(normals, dim=0)
    uv_coords_triangles = torch.concat(uv_coords_triangles, dim=0)
    grid_index = torch.tensor(grid_index, dtype=torch.int32)
    grid_coords = torch.stack(grid_coords, dim=0)
    
    # Return a single batch containing all aggregated items
    return grid_coords, grid_cells, vertices, normals, uv_coords_triangles, grid_index
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

  writer.write_tif()

if __name__ == '__main__':
  obj = '../ink-explorer/cubes/03513_01900_03398/03513_01900_03398_20230702185753.obj'
  scroll = '../ink-explorer/cubes/03513_01900_03398/03513_01900_03398_volume.tiff'

  ppm_and_texture(obj, scroll=scroll)
