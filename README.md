# PGCS
L. Xu, H. Li*, H. Shen, M. Lei and T. Jiang, "PGCS: Physical Law embedded Generative Cloud Synthesis in Remote Sensing Images," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2025.3553239. 

# Code
Decirrus.py:This script is designed for the rapid removal of cirrus clouds from Landsat 8/9 imagery, corresponding to the PGCS_M.The required input data is top-of-atmosphere reflectance data.

Gen_dataset.py:This script can be used to generate an extensive "cloudy & cloud-free" paired dataset. It also supports radiometric calibration of DN (Digital Number) data and the cropping of ROI (Regions of Interest).

# Sample
The figures below show the effect of PGCS in the synthesis of cloud  and cloudy images. You can download the sample data from BaiduNetdisk. Link: https://pan.baidu.com/s/1ZpuB9qxVhHSOIxdMGM4Lhw?pwd=pgcs key：pgcs

|![image](https://github.com/user-attachments/assets/c2bf4141-c0a4-441b-ac2c-edaa7a24a327)|![image](https://github.com/user-attachments/assets/e25000b3-619d-4a20-b182-ba0a7837385f)|![image](https://github.com/user-attachments/assets/e0428cb9-fd6e-4640-a1df-6078f04dd3ca)|
|-----------------------------|-----------------------------|-----------------------------|

# Acknowledgments
Our code is inspired by [StyleGAN](https://github.com/NVlabs/stylegan3)
