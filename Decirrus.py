import numpy as np
from libtiff import TIFF

def cirrus_cal_gamma(cirrus):
    return  -0.14 * np.log(cirrus)


if __name__ == '__main__':

    data = TIFF.open(r"E:\cloudy_RGB.tif",mode='r').read_image()
    cirrus = TIFF.open(r"E:\cirrus.tif",mode='r').read_image()

    #拟合的cirrus与gamma的相关关系
    realgamma = np.zeros_like(cirrus)
    realgamma = np.where(cirrus > 0, cirrus_cal_gamma(cirrus), realgamma)

    B = data[2] - (np.power(1.375 / 0.4626,realgamma)*cirrus);
    G = data[1] - (np.power(1.375 / 0.5613,realgamma)*cirrus);
    R = data[0] - (np.power(1.375 / 0.6546,realgamma)*cirrus);


    B[B < 0] = 0
    G[G < 0] = 0
    R[R < 0] = 0
    rgb = np.stack([R, G, B], axis=0)

    outtiff = TIFF.open(r'E:\Decirrus.tif',mode='w')
    outtiff.write_image(rgb,compression=None, write_rgb=True)
