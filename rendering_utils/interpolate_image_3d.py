# Adjusted from https://github.com/teamtomo/libtilt/blob/b09dd9b245a3ca48354161cb6126d192a6af3e78/src/libtilt/interpolation/interpolate_image_3d.py, https://github.com/teamtomo/libtilt/blob/b09dd9b245a3ca48354161cb6126d192a6af3e78/src/libtilt/coordinate_utils.py#L7

import einops
import torch
import torch.nn.functional as F

from typing import Sequence

def array_to_grid_sample(
  array_coordinates: torch.Tensor, array_shape: Sequence[int]
) -> torch.Tensor:
  """Generate grids for `torch.nn.functional.grid_sample` from array coordinates.

  These coordinates should be used with `align_corners=True` in
  `torch.nn.functional.grid_sample`.


  Parameters
  ----------
  array_coordinates: torch.Tensor
    `(..., d)` array of d-dimensional coordinates.
    Coordinates are in the range `[0, N-1]` for the `N` elements in each dimension.
  array_shape: Sequence[int]
      shape of the array being sampled at `array_coordinates`.
  """
  dtype, device = array_coordinates.dtype, array_coordinates.device
  array_shape = torch.as_tensor(array_shape, dtype=dtype, device=device)
  grid_sample_coordinates = (array_coordinates / (0.5 * array_shape - 0.5)) - 1
  grid_sample_coordinates = torch.flip(grid_sample_coordinates, dims=(-1,))
  return grid_sample_coordinates

def extract_from_image_4d(
  image: torch.Tensor,
  image_index: torch.Tensor,
  coordinates: torch.Tensor
) -> torch.Tensor:
  """Sample a volume with linear interpolation.

  Parameters
  ----------
  image: torch.Tensor
    `(n, d, h, w)` volume.
  image_index: torch.Tensor
    `(..., )` array of indices of the image to sample from.
  coordinates: torch.Tensor
    `(..., zyx)` array of coordinates at which `image` should be sampled.
    Coordinates should be ordered zyx, aligned with image dimensions `(d, h, w)`.
    Coordinates should be array coordinates, spanning `[0, N-1]` for a
    dimension of length N.
  Returns
  -------
  samples: torch.Tensor
    `(..., )` array of complex valued samples from `image`.
  """
  device = image.device

  # pack coordinates into shape (b, 4)
  coordinates, ps = einops.pack([coordinates], pattern='* zyx')
  n_samples = coordinates.shape[0]
  # pack image_index into shape (b, 1)
  image_index, _ = einops.pack([image_index], pattern='*')
  image_index = image_index.long()  # Convert to long dtype

  # sample dft at coordinates
  image = einops.repeat(image, 'n d h w -> b n d h w', b=n_samples)  # b n d h w

  coordinates = einops.rearrange(coordinates, 'b zyx -> b 1 1 1 zyx')  # b d h w zyx
  samples = F.grid_sample(
    input=image,
    grid=array_to_grid_sample(coordinates, array_shape=image.shape[-3:]),
    mode='bilinear',  # this is trilinear when input is volumetric
    padding_mode='border',  # this increases sampling fidelity at edges
    align_corners=True,
  )
  samples = einops.rearrange(samples, 'b complex 1 1 1 -> b complex')
  # extract the image_index
  samples = samples[torch.arange(n_samples), image_index]

  # zero out samples from outside of volume
  coordinates = einops.rearrange(coordinates, 'b 1 1 1 zyx -> b zyx')
  volume_shape = torch.as_tensor(image.shape[-3:], device=device)
  inside = torch.logical_and(coordinates >= 0, coordinates <= volume_shape)
  inside = torch.all(inside, dim=-1)  # (b, d, h, w)
  samples[~inside] *= 0

  # pack data back up and return
  [samples] = einops.unpack(samples, pattern='*', packed_shapes=ps)
  return samples  # (...)
