import torch
import tifffile
import numpy as np

# Function to detect surface points in a 3D volume
def surface_detection(volume, global_reference_vector, blur_size=3, sobel_chunks=4, sobel_overlap=3, window_size=20, stride=20, threshold_der=0.1, threshold_der2=0.001, convert_to_numpy=True):
    return (volume, global_reference_vector)

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
    
    print(volume.shape)
