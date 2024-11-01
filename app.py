from pipeline_api import Node
from mesh_to_surface import ppm_and_texture

def handler(inputs):
    z, y, x = 10624, 2304, 2432
    label, grid_size = 1, 768

    obj = f'/Users/yao/Desktop/cubes/{z:05}_{y:05}_{x:05}/label_{label:02}/{z:05}_{y:05}_{x:05}_{label:02}.obj'
    scroll = f'/Users/yao/Desktop/cubes/{z:05}_{y:05}_{x:05}/{z:05}_{y:05}_{x:05}_volume.tif'
    mask = f'/Users/yao/Desktop/cubes/{z:05}_{y:05}_{x:05}/label_{label:02}/{z:05}_{y:05}_{x:05}_mask.png'

    ppm_and_texture(obj, scroll, mask, (z, y, x), grid_size)

    return {**inputs, "data": {"counter": 3}}

if __name__ == "__main__":
    Node(handler)
