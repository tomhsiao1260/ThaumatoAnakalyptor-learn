### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import torch
import torch.nn as nn
import torch.nn.functional as F

import tifffile
import numpy as np

## sobel_filter_3d from https://github.com/lukeboi/scroll-viewer/blob/dev/server/app.py
### adjusted for my use case and improve efficiency
def sobel_filter_3d(input, chunks=4, overlap=3, device=None):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = input.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16)

    # Define 3x3x3 kernels for Sobel operator in 3D
    sobel_x = torch.tensor([
        [[[ 1, 0, -1], [ 2, 0, -2], [ 1, 0, -1]],
         [[ 2, 0, -2], [ 4, 0, -4], [ 2, 0, -2]],
         [[ 1, 0, -1], [ 2, 0, -2], [ 1, 0, -1]]],
    ], dtype=torch.float16).to(device)

    sobel_y = sobel_x.transpose(2, 3)
    sobel_z = sobel_x.transpose(1, 3)

    # Add an extra dimension for the input channels
    sobel_x = sobel_x[None, ...]
    sobel_y = sobel_y[None, ...]
    sobel_z = sobel_z[None, ...]

    assert len(input.shape) == 5, "Expected 5D input (batch_size, channels, depth, height, width)"

    depth = input.shape[2]
    chunk_size = depth // chunks
    chunk_overlap = overlap // 2

    # Initialize tensors for results and vectors if needed
    vectors = torch.zeros(list(input.shape) + [3], device=device, dtype=torch.float16)

    for i in range(chunks):
        # Determine the start and end index of the chunk
        start = max(0, i * chunk_size - chunk_overlap)
        end = min(depth, (i + 1) * chunk_size + chunk_overlap)

        if i == chunks - 1:  # Adjust the end index for the last chunk
            end = depth

        chunk = input[:, :, start:end, :, :]

        # Move chunk to GPU
        chunk = chunk.to(device, non_blocking=True)  # Use non_blocking transfers

        G_x = nn.functional.conv3d(chunk, sobel_x, padding=1)
        G_y = nn.functional.conv3d(chunk, sobel_y, padding=1)
        G_z = nn.functional.conv3d(chunk, sobel_z, padding=1)

        # Overlap removal can be optimized
        actual_start = 0 if i == 0 else chunk_overlap
        actual_end = -chunk_overlap if i != chunks - 1 else None
        # Stack gradients in-place if needed
        vectors[:, :, start + actual_start:end + (actual_end if actual_end is not None else 0), :, :] = torch.stack((G_x, G_y, G_z), dim=5)[:, :, actual_start:actual_end, :, :]

        # Free memory of intermediate variables
        del G_x, G_y, G_z, chunk
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vectors.squeeze(0).squeeze(0).to(device)

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

# Function to normalize vectors to unit length
def normalize(vectors):
    return vectors / vectors.norm(dim=-1, keepdim=True)

# Function to calculate angular distance between vectors
def angular_distance(v1, v2):
    v1 = v1.unsqueeze(0) if v1.dim() == 1 else v1
    v2 = v2.unsqueeze(0) if v2.dim() == 1 else v2
    return torch.acos(torch.clamp((v1 * v2).sum(-1), -1.0, 1.0))

# Mean indiscriminative Loss function for a batch of candidate vectors and a batch of input vectors
def loss(candidates, vectors):
    vector_normed = normalize(vectors)
    pos_distance = angular_distance(candidates[:, None, :], vector_normed[None, :, :])
    neg_distance = angular_distance(-candidates[:, None, :], vector_normed[None, :, :])
    losses = torch.min(torch.stack((pos_distance, neg_distance), dim=-1), dim=-1)[0]
    
    # calculate the norm of the vectors and use it to scale the losses
    vector_norms = torch.norm(vectors, dim=-1)
    scaled_losses = losses * vector_norms
    
    return scaled_losses

# Function to find the vector that is the propper mean vector for the input vectors vs when -v = v
def find_mean_indiscriminative_vector(vectors, n, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Normalize the input vectors
    vectors = normalize(vectors)
    
    # Generate n random unit vectors
    random_vectors = torch.randn(n, 3, device=device)
    random_vectors = normalize(random_vectors)

    # Compute the total loss for each candidate
    total_loss = loss(random_vectors, vectors).sum(dim=-1)

    # Find the best candidate
    best_vector = random_vectors[torch.argmin(total_loss)]

    return best_vector

# Function that  projects vector a onto vector b
def vector_projection(a, b):
    return (a * b).sum(-1, keepdim=True) * b / b.norm(dim=-1, keepdim=True)**2

# Function that adjusts the norm of vector a to the norm of vector b based on their direction
def adjusted_norm(a, b):
    # Calculate the projection
    projection = vector_projection(a, b)
    
    # Compute the dot product of the projected vector and the original vector b
    dot_product = (projection * b).sum(-1, keepdim=True)
    
    # Compute the norm of the projection
    projection_norm = projection.norm(dim=-1)
    
    # Adjust the norm based on the sign of the dot product
    adjusted_norm = torch.sign(dot_product.squeeze()) * projection_norm
    
    return adjusted_norm

def scale_to_0_1(tensor):
    # Compute the 99th percentile
    #quantile_val = torch.quantile(tensor, 0.95)
    
    # Clip the tensor values at the 99th percentile
    clipped_tensor = torch.clamp(tensor, min=-1000, max=1000)
    
    # Scale the tensor to the range [0,1]
    tensor_min = torch.min(clipped_tensor)
    tensor_max = torch.max(clipped_tensor)
    tensor_scale = torch.max(torch.abs(tensor_min), torch.abs(tensor_max))
    scaled_tensor = clipped_tensor / tensor_scale
    return scaled_tensor

# Function that convolutes a 3D Volume of vectors to find their mean indiscriminative vector
def vector_convolution(input_tensor, window_size=20, stride=20, device=None):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    # get the size of your 4D input tensor
    input_size = input_tensor.size()

    # initialize an output tensor
    output_tensor = torch.zeros((input_size[0] - window_size + 1) // stride, 
                                (input_size[1] - window_size + 1) // stride, 
                                (input_size[2] - window_size + 1) // stride, 
                                3, device=device)

    # slide the window across the 3D volume
    for i in range(0, input_size[0] - window_size + 1, stride):
        for j in range(0, input_size[1] - window_size + 1, stride):
            for k in range(0, input_size[2] - window_size + 1, stride):
                # extract the vectors within the window
                window_vectors = input_tensor[i:i+window_size, j:j+window_size, k:k+window_size]
                window_vectors = window_vectors.reshape(-1, 3)  # flatten the 3D window into a 2D tensor
                
                # calculate the closest vector
                best_vector = find_mean_indiscriminative_vector(window_vectors, 100, device)  # adjust the second parameter as needed
                # check if the indices are within the output_tensor's dimension
                if i//stride < output_tensor.shape[0] and j//stride < output_tensor.shape[1] and k//stride < output_tensor.shape[2]:
                    # store the result in the output tensor
                    output_tensor[i//stride, j//stride, k//stride] = best_vector

    return output_tensor

# Function that interpolates the output tensor to the original size of the input tensor
def interpolate_to_original(input_tensor, output_tensor):
    # Adjust the shape of the output tensor to match the input tensor
    # by applying 3D interpolation. We're assuming that the last dimension
    # of both tensors is the channel dimension (which should not be interpolated over).
    output_tensor = output_tensor.permute(3, 0, 1, 2)
    input_tensor = input_tensor.permute(3, 0, 1, 2)

    # Use 3D interpolation to resize output_tensor to match the shape of input_tensor.
    interpolated_tensor = F.interpolate(output_tensor.unsqueeze(0), size=input_tensor.shape[1:], mode='trilinear', align_corners=False)

    # Return the tensor to its original shape.
    interpolated_tensor = interpolated_tensor.squeeze(0).permute(1, 2, 3, 0)

    return interpolated_tensor

# Function that adjusts the vectors in the input tensor to point in the same general direction as the global reference vector.
def adjust_vectors_to_global_direction(input_tensor, global_reference_vector):
    # Compute dot product of each vector in the input tensor with the global reference vector.
    # The resulting tensor will have the same shape as the input tensor, but with the last dimension squeezed out.
    dot_products = (input_tensor * global_reference_vector).sum(dim=-1, keepdim=True)
    # Create a mask of the same shape as the dot products tensor, 
    # with True wherever the dot product is negative.
    mask = dot_products < 0
    
    # Expand the mask to have the same shape as the input tensor
    mask = mask.expand(input_tensor.shape)

    # Negate all vectors in the input tensor where the mask is True.
    adjusted_tensor = input_tensor.clone()
    adjusted_tensor[mask] = -input_tensor[mask]

    return adjusted_tensor

# Function to detect surface points in a 3D volume
def surface_detection(volume, global_reference_vector, blur_size=3, sobel_chunks=4, sobel_overlap=3, window_size=20, stride=20, threshold_der=0.1, threshold_der2=0.001, convert_to_numpy=True):
    # device
    device = volume.device

    sobel_vectors = torch.load('../output/sobel.pt')
    sobel_vectors_subsampled = torch.load('../output/sobel_sampled.pt')
    sobel_vectors = torch.load('../output/sobel.pt')
    vector_conv = torch.load('../output/vector_conv.pt')
    adjusted_vectors_interp = torch.load('../output/adjusted_vectors_interp.pt')
    first_derivative = torch.load('../output/first_derivative.pt')

    # # using half percision to save memory
    # volume = volume
    # # Blur the volume
    # blur = uniform_blur3d(channels=1, size=blur_size, device=device)
    # blurred_volume = blur(volume.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    # torch.save(blurred_volume, '../output/blur.pt')

    # # Apply Sobel filter to the blurred volume
    # sobel_vectors = sobel_filter_3d(volume, chunks=sobel_chunks, overlap=sobel_overlap, device=device)
    # torch.save(sobel_vectors, '../output/sobel.pt')

    # # Subsample the sobel_vectors
    # sobel_stride = 10
    # sobel_vectors_subsampled = sobel_vectors[::sobel_stride, ::sobel_stride, ::sobel_stride, :]
    # torch.save(sobel_vectors_subsampled, '../output/sobel_sampled.pt')

    # # Apply vector convolution to the Sobel vectors
    # vector_conv = vector_convolution(sobel_vectors_subsampled, window_size=window_size, stride=stride, device=device)
    # torch.save(vector_conv, '../output/vector_conv.pt')

    # # Adjust vectors to the global direction
    # adjusted_vectors = adjust_vectors_to_global_direction(vector_conv, global_reference_vector)
    # torch.save(adjusted_vectors, '../output/adjusted_vectors.pt')

    # # Interpolate the adjusted vectors to the original size
    # adjusted_vectors_interp = interpolate_to_original(sobel_vectors, adjusted_vectors)
    # torch.save(adjusted_vectors_interp, '../output/adjusted_vectors_interp.pt')

    # # Project the Sobel result onto the adjusted vectors and calculate the norm
    # first_derivative = adjusted_norm(sobel_vectors, adjusted_vectors_interp)
    # fshape = first_derivative.shape
    
    # first_derivative = scale_to_0_1(first_derivative)
    # torch.save(first_derivative, '../output/first_derivative.pt')

    # Apply Sobel filter to the first derivative, project it onto the adjusted vectors, and calculate the norm
    sobel_vectors_derivative = sobel_filter_3d(first_derivative, chunks=sobel_chunks, overlap=sobel_overlap, device=device)
    second_derivative = adjusted_norm(sobel_vectors_derivative, adjusted_vectors_interp)
    second_derivative = scale_to_0_1(second_derivative)
    torch.save(second_derivative, '../output/second_derivative.pt')

    return (volume, global_reference_vector)

def torch_to_tif(tensor, path):
    volume = tensor.numpy()
    volume = np.abs(volume)
    volume = volume.astype(np.uint8)
    tifffile.imwrite(path, volume)

if __name__ == '__main__':
    path = '../2dtifs_8um_grids/cell_yxz_006_008_004.tif'

    volume = tifffile.imread(path)
    volume = volume[:300, :300, :300]
    volume = np.uint8(volume//256)
    volume = torch.from_numpy(volume).float()  # Convert to float32 tensor

    normal = np.array([0, 1, 0]) # z, y, x
    tensor_tuple = surface_detection(volume, normal, blur_size=11, window_size=9, stride=1, threshold_der=0.075, threshold_der2=0.002, convert_to_numpy=False)
    points_r_tensor, normals_r_tensor = tensor_tuple

    # tensor = torch.load('../output/blur.pt')
    # torch_to_tif(tensor, '../output/blur.tif')

    # tensor = torch.load('../output/sobel.pt')
    # torch_to_tif(tensor, '../output/sobel.tif')

    # tensor = torch.load('../output/sobel_sampled.pt')
    # torch_to_tif(tensor, '../output/sobel_sampled.tif')

    # tensor = torch.load('../output/vector_conv.pt') * 255
    # torch_to_tif(tensor, '../output/vector_conv.tif')

    # tensor = torch.load('../output/adjusted_vectors.pt') * 255
    # torch_to_tif(tensor, '../output/adjusted_vectors.tif')

    # tensor = torch.load('../output/adjusted_vectors_interp.pt') * 255
    # torch_to_tif(tensor, '../output/adjusted_vectors_interp.tif')

    # tensor = torch.load('../output/first_derivative.pt') * 255
    # torch_to_tif(tensor, '../output/first_derivative.tif')

    tensor = torch.load('../output/second_derivative.pt') * 255
    torch_to_tif(tensor, '../output/second_derivative.tif')
    
