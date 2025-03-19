import numpy as np
from libtiff import TIFF

def cirrus_cal_gamma(cirrus):
    return  -0.14 * np.log(cirrus)


if __name__ == '__main__':

    data = TIFF.open(r"E:\cloudy.tif",mode='r').read_image()
    cirrus = TIFF.open(r"E:\cirrus.tif",mode='r').read_image()

    #拟合的cirrus与gamma的相关关系
    realgamma = np.zeros_like(cirrus)
    realgamma = np.where(cirrus > 0, cirrus_cal_gamma(cirrus), realgamma)
    
    CB = data[2] - (np.power(1.375 / 0.4500,realgamma)*cirrus);
    B = data[2] - (np.power(1.375 / 0.4626,realgamma)*cirrus);
    G = data[1] - (np.power(1.375 / 0.5613,realgamma)*cirrus);
    R = data[0] - (np.power(1.375 / 0.6546,realgamma)*cirrus);
    NIR = data[0] - (np.power(1.375 / 0.8650,realgamma)*cirrus);

    CB[CB < 0] = 0
    B[B < 0] = 0
    G[G < 0] = 0
    R[R < 0] = 0
    NIR[NIR < 0] = 0
    
    result = np.stack([CB, B, G, R, NIR], axis=0)

    outtiff = TIFF.open(r'E:\Decirrus.tif',mode='w')
    outtiff.write_image(result,compression=None, write_rgb=True)
