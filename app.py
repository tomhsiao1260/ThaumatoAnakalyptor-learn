from flask import Flask, jsonify
import requests
from parse_args import parse_args
from mesh_to_surface import ppm_and_texture

app = Flask(__name__)

id_value, from_value = parse_args()

@app.route("/")
def process():

    inputs = requests.get("http://127.0.0.1:" + from_value).json()

    handler()

    outputs = {"data": {"counter": 10}}

    return jsonify(outputs)

def handler():
    label, grid_size = 1, 768
    z, y, x = 3513, 1900, 3400

    obj = f'/Users/yao/Desktop/cubes/{z:05}_{y:05}_{x:05}/{z:05}_{y:05}_{x:05}_{label:02}.obj'
    scroll = f'/Users/yao/Desktop/cubes/{z:05}_{y:05}_{x:05}/{z:05}_{y:05}_{x:05}_volume.tif'
    ppm_and_texture(obj, scroll, (z, y, x), grid_size)

if __name__ == "__main__":
    app.run(port=id_value)
