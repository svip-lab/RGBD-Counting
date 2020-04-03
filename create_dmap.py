import numpy as np
import math


def GaussianKernel(shape=(3, 3), sigma=0.5):
    """
    2D gaussian kernel which is equal to MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    radius_x, radius_y = [(radius-1.)/2. for radius in shape]
    y_range, x_range = np.ogrid[-radius_y:radius_y+1, -radius_x:radius_x+1]
    h = np.exp(- (x_range*x_range + y_range*y_range) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumofh = h.sum()
    if sumofh != 0:
        h /= sumofh
    return h


def create_dmap(img, gtLocation, depth, beta=0.25, downscale=8.0):
    width, height = img.size
    raw_width, raw_height = width, height
    width = math.floor(width / downscale)
    height = math.floor(height / downscale)
    raw_loc = gtLocation
    gtLocation = gtLocation / downscale
    gaussRange = 25
    # kernel = GaussianKernel(shape=(25, 25), sigma=3)
    pad = int((gaussRange - 1) / 2)
    densityMap = np.zeros((int(height + gaussRange - 1), int(width + gaussRange - 1)))
    for gtidx in range(gtLocation.shape[0]):
        if 0 <= gtLocation[gtidx, 0] < width and 0 <= gtLocation[gtidx, 1] < height:
            xloc = int(math.floor(gtLocation[gtidx, 0]) + pad)
            yloc = int(math.floor(gtLocation[gtidx, 1]) + pad)
            x_down = max(int(raw_loc[gtidx, 0] - 4), 0)
            x_up = min(int(raw_loc[gtidx, 0] + 5), raw_width)
            y_down = max(int(raw_loc[gtidx, 1]) - 4, 0)
            y_up = min(int(raw_loc[gtidx, 1] + 5), raw_height)
            depth_mean = np.sum(depth[y_down:y_up, x_down:x_up]) / (x_up - x_down) / (y_up - y_down)
            kernel = GaussianKernel((25, 25), sigma=beta * 5 / depth_mean)
            densityMap[yloc - pad:yloc + pad + 1, xloc - pad:xloc + pad + 1] += kernel
    densityMap = densityMap[pad:pad + height, pad:pad + width]
    return densityMap