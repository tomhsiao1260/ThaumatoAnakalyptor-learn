import torch
import torch.nn as nn

import tifffile
import numpy as np

# Function to create a 3D Uniform kernel
def get_uniform_kernel(size=3, channels=1):
    # Create a 3D kernel filled with ones and normalize it
    kernel = torch.ones((size, size, size))
    kernel = kernel / torch.sum(kernel)
    return kernel

# Function to create a 3D convolution layer with a Uniform kernel
def uniform_blur3d(channels=1, size=3, device=None):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernel = get_uniform_kernel(size, channels)
    # Repeat the kernel for all input channels
    kernel = kernel.repeat(channels, 1, 1, 1, 1)
    # Create a convolution layer
    blur_layer = nn.Conv3d(in_channels=channels, out_channels=channels, 
                           kernel_size=size, groups=channels, bias=False, padding=size//2)
    # Set the kernel weights
    blur_layer.weight.data = nn.Parameter(kernel)
    # Make the layer non-trainable
    blur_layer.weight.requires_grad = False
    blur_layer.to(device)
    return blur_layer

# Function to detect surface points in a 3D volume
def surface_detection(volume, global_reference_vector, blur_size=3, sobel_chunks=4, sobel_overlap=3, window_size=20, stride=20, threshold_der=0.1, threshold_der2=0.001, convert_to_numpy=True):
    # device
    device = volume.device
    # using half percision to save memory
    volume = volume
    # Blur the volume
    blur = uniform_blur3d(channels=1, size=3, device=device)
    blurred_volume = blur(volume.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

    return (blurred_volume, global_reference_vector)

if __name__ == '__main__':
    path = '../2dtifs_8um_grids/cell_yxz_006_008_004.tif'

    volume = tifffile.imread(path)
    volume = volume[:300, :300, :300]
    volume = np.uint8(volume//256)
    volume = torch.from_numpy(volume).float()  # Convert to float32 tensor

    tensor_tuple = surface_detection(volume, None)
    points_r_tensor, normals_r_tensor = tensor_tuple

    volume = points_r_tensor
    volume = volume.numpy()
    volume = np.uint8(volume)

    tifffile.imwrite('output.tif', volume)
