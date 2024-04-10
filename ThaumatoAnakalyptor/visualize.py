import os
import cv2
import tifffile
import numpy as np

def drawImage(name, repeats=1):
    # input & output path
    path_input = os.path.join('../output', name)
    path_output = os.path.join('../output/video', name.replace('.tif', '.mp4'))
    if not os.path.exists('../output/video'): os.makedirs('../output/video')

    # load tiff data
    data = tifffile.imread(path_input)[1:-1]
    data = np.repeat(data, repeats=repeats, axis=0)
    data = np.repeat(data, repeats=repeats, axis=1)
    data = np.repeat(data, repeats=repeats, axis=2)
    d, h, w = data.shape[:3]

    # create video writer
    time = 15
    fps = d / time
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(path_output, fourcc, fps, (w, h))

    # save video for each frame
    for layer in range(d):
        image = data[layer, :, :]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imshow('Volume', image)
        cv2.waitKey(int(1000/fps))
        cv2.destroyAllWindows()
        writer.write(image)

    writer.release()

if __name__ == '__main__':
    drawImage('origin.tif')
    drawImage('blur.tif')
    drawImage('sobel.tif')
    drawImage('sobel_sampled.tif', repeats=10)
    drawImage('adjusted_vectors_interp.tif')
    drawImage('first_derivative.tif')
    drawImage('second_derivative.tif')
    drawImage('recto_verso.tif')
