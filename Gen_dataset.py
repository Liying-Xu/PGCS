import os
import re
import time
import glob
import argparse
import numpy as np
import pandas as pd
from osgeo import gdal
from multiprocessing.dummy import Pool as ThreadPool  # 多线程
import xml.etree.ElementTree as ET
import torch
import random
import dnnlib # DNN权重初始化、网络定义、优化器
import legacy # 兼容旧版StyleGAN的权重代码
import click
from typing import List, Optional, Tuple, Union
from libtiff import TIFF
import cv2
import math
import linecache
import matplotlib.pyplot as plt

def parse_args():
    """
    参数设置参数
    """
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    parser.add_argument('--InDataPath', type=str,
                        default=r"E:\F1_LC08_L1TP_123039_20211110_20211117_02_T1",
                        help='程序输入：待处理影像数据文件夹|路径')
    parser.add_argument('--InROIPath', type=str,
                        default=r'E:\roi.xml',
                        help='程序输入：晴空区的ROI区域|路径')
    parser.add_argument('--OutDataset', type=str,
                        default=r"E:\Dataset",
                        help='程序输出：配对数据集构建存储路径')
    parser.add_argument('--CropSize', type=int,
                        default= 512,
                        help='影像裁剪尺寸')
    parser.add_argument('--Stride', type=int,
                        default= 128,
                        help='影像裁切步长')
    parser.add_argument('--Generator', type=str,
                        default= r"E:\network.pkl",
                        help='随机云影像生成器')
    parser.add_argument('--FreeThick', type=float,
                        default= (0.2,0.5),
                        help='数据集厚薄比例,根据需要调整')
    args = parser.parse_args()
    return args

#----------------------------------------------------------------------------
def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

def roi_geo2pixel(roi_path, dataset):
    roi = ET.parse(roi_path).getroot()  # 读取ROI文件
    coordinates = roi.find('.//Coordinates').text.strip().split()  # 获取感兴趣区坐标
    coordinates = [float(coord) for coord in coordinates]
    # ROI四个角点的地理坐标信息（顺序：左上、右上、右下、左下）
    ulx = coordinates[0]
    uly = coordinates[1]
    urx = coordinates[2]
    ury = coordinates[3]
    lrx = coordinates[4]
    lry = coordinates[5]
    llx = coordinates[6]
    lly = coordinates[7]
    # 计算ROI的像素坐标
    geotransform = dataset.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(geotransform)
    ulx_pixel, uly_pixel = gdal.ApplyGeoTransform(inv_gt, ulx, uly)
    urx_pixel, ury_pixel = gdal.ApplyGeoTransform(inv_gt, urx, ury)
    lrx_pixel, lry_pixel = gdal.ApplyGeoTransform(inv_gt, lrx, lry)
    llx_pixel, lly_pixel = gdal.ApplyGeoTransform(inv_gt, llx, lly)
    # 计算ROI的范围
    min_x = int(min(ulx_pixel, urx_pixel, lrx_pixel, llx_pixel))
    max_x = int(max(ulx_pixel, urx_pixel, lrx_pixel, llx_pixel))
    min_y = int(min(uly_pixel, ury_pixel, lry_pixel, lly_pixel))
    max_y = int(max(uly_pixel, ury_pixel, lry_pixel, lly_pixel))
    width = max_x - min_x
    height = max_y - min_y

    return min_x, min_y, width, height

def multiReadandCorrect(datapath,roipath):
    """
    批量读取Landsat原始影像数据，并做辐射定标为TOA反射率数据
    :param path: Landsat原始影像路径
    :return:
    """
    metadata = glob.glob(os.path.join(datapath, "*MTL*.txt"))[0]
    files = glob.glob(os.path.join(datapath, "*B[1-9].TIF"))

    f = open(metadata, 'r')
    byt = f.readlines()
    f.close()
    metadf = pd.DataFrame(byt)
    metadf[['label', 'value']] = metadf[0].str.split('=', n=1, expand=True)
    SUN_ELEVATION = metadf[metadf['label'].str.contains('SUN_ELEVATION')]['value'].reset_index(drop=True)
    SE = float(SUN_ELEVATION[0]) * np.pi / 180.0
    res = []

    def multiTOA(i):
        """
        Landsat8 大气表观反射率
        :param i: 波段号 减 1
        """
        if i < 5 :
            dataset: gdal.Dataset = gdal.Open(files[i], gdal.GA_ReadOnly)
            if roipath is None:
                data = dataset.ReadAsArray(1636, 1636, 4608, 4608)#估计的大概坐标，应用于裁剪中心区域的正方形4608=9*512
            else:
                min_x, min_y, width, height = roi_geo2pixel(roipath, dataset)
                data = dataset.ReadAsArray(min_x, min_y, width, height)

            REFLECTANCE_ADD_BAND_label = 'REFLECTANCE_ADD_BAND_' + str(i + 1)
            REFLECTANCE_ADD_BAND_val = metadf[metadf['label'].str.contains(REFLECTANCE_ADD_BAND_label)][
                'value'].reset_index(drop=True)
            REFLECTANCE_ADD_BAND_i = float(REFLECTANCE_ADD_BAND_val[0])

            REFLECTANCE_MULT_BAND_label = 'REFLECTANCE_MULT_BAND_' + str(i + 1)
            REFLECTANCE_MULT_BAND_val = metadf[metadf['label'].str.contains(REFLECTANCE_MULT_BAND_label)][
                'value'].reset_index(drop=True)
            REFLECTANCE_MULT_BAND_i = float(REFLECTANCE_MULT_BAND_val[0])

            # 大气表观反射率
            TOA_ = REFLECTANCE_MULT_BAND_i * data + REFLECTANCE_ADD_BAND_i
            TOA_[TOA_ == REFLECTANCE_ADD_BAND_i] = 0
            TOA = TOA_ / np.sin(SE)
            TOA[ TOA < 0] = 0
            TOA[ TOA > 1] = 1
            res.append([i, TOA, dataset])
            del dataset

    # 多线程大气表观发射率计算
    pool = ThreadPool()
    pool.imap(multiTOA, range(len(files)))
    pool.close()
    pool.join()
    return res

def crop_image(Oridata, crop_size, stride):
    cropped_images = []
    channel, height, width = Oridata.shape

    for i in range(0, height - crop_size + 1, stride):
        for j in range(0, width - crop_size + 1, stride):
            data = Oridata[:, i:i + crop_size, j:j + crop_size]
            cropped_images.append(data)

    return cropped_images

def plot_histogram(data, color, title, ax, maxv):
    # 统计数据点
    hist, bins = np.histogram(data.flatten(), bins=np.arange(0, maxv+0.001, 0.001))
    # 计算中心取值
    center_values = (bins[:-1] + bins[1:]) / 2
    # 绘制曲线
    ax.plot(center_values, hist, color=color, label=title)

def gen_cirrus(gen,num,p):
    cirrus_list = []
    pFree = p[0]
    pThick = p[1]/(1 - p[0])
    translate = 0,0
    rotate = 0
    device = torch.device('cuda')
    # 生成器预设
    class_idx = None
    with dnnlib.util.open_url(gen) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
        # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')
    # 逐一生成for i in range(num):

    while len(cirrus_list) < num:
        print(len(cirrus_list))
        p1 = random.random()
        if p1 < pFree:
            shape = (512, 512)
            img = np.zeros(shape)
            cirrus_list.append(img)
        else:
            seed = int(random.uniform(1, 10000))  # 完全随机的随机数

            z = torch.from_numpy( np.random.RandomState(seed).randn(1, G.z_dim)).to(device) #这里的随机数，确保当输入的seed相同时，生成的云也相同
            # Construct an inverse rotation/translation matrix and pass to the generator.  The
            # generator expects this matrix as an inverse to avoid potentially failing numerical
            # operations in the network.
            if hasattr(G.synthesis, 'input'):
                m = make_transform(translate, rotate)
                m = np.linalg.inv(m)
                G.synthesis.input.transform.copy_(torch.from_numpy(m))

            img = G(z, label, truncation_psi=1, noise_mode='random')#1,5,512,512

            img = (((img+1)*0.05).clamp(0, 1).cpu()).squeeze()#5,512,512

            # 随机选择旋转角度或翻转方式
            angle = np.random.choice([0, 90, 180, 270])
            flip = np.random.choice(['vertical', 'horizontal'])

            # 执行旋转或翻转操作
            if angle != 0:
                img = np.rot90(img, k=angle // 90)
            if flip == 'vertical':
                img = np.flipud(img)
            elif flip == 'horizontal':
                img = np.fliplr(img)

            # 这里可以设置一个比重系数
            p2 = random.random()  # 生成0到1之间的随机数,作为厚薄数量的比重系数

            if p2 < pThick: #40%的概率（数据集中50%的比例）作线性变化
                # 变厚缩放因子
                scale_factor = np.random.uniform(1, 3)
                # 对其他数值进行等倍数变化
                img = scale_factor * img
                cirrus_list.append(img)
            else:
                # 变薄缩放因子
                scale_factor = np.random.uniform(0.9, 1)

                # 对其他数值进行等倍数变化
                img = scale_factor * img
                cirrus_list.append(img)

            # 厚度控制模块 （根据需要调整）
            minv = np.quantile(img, 0.02, method='nearest')
            img = img - minv
            img[img < 0] = 0
            maxv = np.quantile(img, 0.98, method='nearest')

            if maxv > 0.02 and maxv <0.1 :
                cirrus_list.append(img)

    return cirrus_list

def channel_misalignmen(img):
    # 确定图像尺寸
    height, width = img.shape

    # 定义填充、偏移宽度，偏移量应该小于填充量
    padding_width = 2
    offset_width = 2
    # Landsat-8/9 2
    # Sentinel-2 5
    # Gaofen-2 0
    
    # 执行填充
    padded_img = np.pad(img, padding_width, mode='reflect')

    # 生成随机位置偏移
    offset_x = np.random.randint(-offset_width, offset_width+1)  # 在 0 到 10 之间随机偏移
    offset_y = np.random.randint(-offset_width, offset_width+1)

    # 计算中心区域的索引范围
    start_y = padding_width + offset_y
    end_y = start_y + height
    start_x = padding_width + offset_x
    end_x = start_x + width

    # 提取中心区域
    center_img = padded_img[start_y:end_y, start_x:end_x]

    return center_img

def gen_cloud(num, gen, p):
    list_cloud = []

    # 云形态预设
    cirrus = gen_cirrus(gen, num, p)

    # 云生成
    for i in range(num):
        # 检查 cirrus 中是否有零值或负值
        valid_indices = np.where(cirrus[i] > 0)

        # 计算 Cloud
        CloudCB = np.zeros_like(cirrus[i])
        CloudB = np.zeros_like(cirrus[i])
        CloudG = np.zeros_like(cirrus[i])
        CloudR = np.zeros_like(cirrus[i])
        CloudNIR = np.zeros_like(cirrus[i])

        # 仅对有效索引进行计算
        CloudCB[valid_indices] = pow(1.375 / 0.4500, (-0.14 * np.log(cirrus[i][valid_indices]))) * cirrus[i][valid_indices]#landsat8 0.45 /s2 0.443
        CloudB[valid_indices] = pow(1.375 / 0.4626, (-0.14 * np.log(cirrus[i][valid_indices]))) * cirrus[i][valid_indices]#landsat8 0.4626 /s2 0.49 /GF2 0.4855
        CloudG[valid_indices] = pow(1.375 / 0.5613, (-0.14 * np.log(cirrus[i][valid_indices]))) * cirrus[i][valid_indices]#landsat8 0.5613 /s2 0.56 /GF2 0.5575
        CloudR[valid_indices] = pow(1.375 / 0.6546, (-0.14 * np.log(cirrus[i][valid_indices]))) * cirrus[i][valid_indices]#landsat8 0.6546 /s2 0.665 /GF2 0.6635
        CloudNIR[valid_indices] = pow(1.375 / 0.8650, (-0.14 * np.log(cirrus[i][valid_indices]))) * cirrus[i][valid_indices]#landsat8 0.8650 /s2 0.842 /GF2 0.833

        CloudCB = channel_misalignmen(CloudCB)
        CloudB = channel_misalignmen(CloudB)
        CloudG = channel_misalignmen(CloudG)
        CloudR = channel_misalignmen(CloudR)
        CloudNIR = channel_misalignmen(CloudNIR)

        cloud = np.stack([CloudCB, CloudB, CloudG, CloudR, CloudNIR],axis=0)#cloud = np.stack([CloudR, CloudG, CloudB], axis=0) 
        list_cloud.append(cloud)

    return list_cloud


def write_image(img_list,tifpath):
    num = len(img_list)
    for i in range(num):
        img = img_list[i]
        out_tiff = TIFF.open(f'{tifpath}/img{i}.tif', mode='w')
        out_tiff.write_image(img, compression=None, write_rgb=True)




if __name__ == '__main__':
    args = parse_args()
    ##### 设置输入影像文件夹路径
    InDataPath = args.InDataPath
    InROIPath = args.InROIPath
    ##### 建立输出路径：Cloudy、Cloudy-free、Cloud
    OutDataset = args.OutDataset
    OutCloudy = os.path.join(OutDataset,'Cloudy')
    OutCloudyFree = os.path.join(OutDataset,'CloudyFree')
    OutCloud = os.path.join(OutDataset, 'Cloud')
    if not os.path.exists(OutCloudy):
        os.makedirs(OutCloudy)
    if not os.path.exists(OutCloudyFree):
        os.makedirs(OutCloudyFree)
    if not os.path.exists(OutCloud):
        os.makedirs(OutCloud)
    ##### 读取Band1-5并辐射定标晴空影像数据为TOA反射率
    t1 = time.perf_counter()
    data = multiReadandCorrect(InDataPath,InROIPath)
    data = sorted(data)
    # list格式转为numpy
    OriCB = data[0][1].astype(np.float32)
    OriB = data[1][1].astype(np.float32)
    OriG = data[2][1].astype(np.float32)
    OriR = data[3][1].astype(np.float32)
    OriNIR = data[4][1].astype(np.float32)
    # 源数据：BGR，改成RGB。注意与云保持统一！！！
    Oridata = np.stack((OriCB, OriB, OriG, OriR, OriNIR), axis=0)#Oridata = np.stack((OriR, OriG, OriB), axis=0)
    t2 = time.perf_counter()
    print("读取数据与辐射定标时间：%f 秒" % (t2 - t1))
    ##### 对晴空影像进行裁剪，并保存到Cloudy-free
    crop_size = args.CropSize
    stride = args.Stride
    CloudyFree_list = crop_image(Oridata, crop_size, stride)
    Num_Dataset = len(CloudyFree_list)# 获取Cloudy-free数据集长度
    ##### 模拟云生成
    Gen = args.Generator
    P = args.FreeThick
    Cloud_list = gen_cloud(Num_Dataset, Gen, P)
    ##### 叠加地表与云，获得Cloudy
    Cloudy_list = []
    for index, item in enumerate(CloudyFree_list):
        Cloudy_list.append(item + Cloud_list[index])
    ##### 将Cloudy、Cloudy-free按对应文件名保存并写出
    write_image(Cloudy_list, OutCloudy)
    write_image(CloudyFree_list, OutCloudyFree)
    write_image(Cloud_list, OutCloud)

