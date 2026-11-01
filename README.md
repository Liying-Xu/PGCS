# PGCS
L. Xu, H. Li*, H. Shen, M. Lei and T. Jiang, "PGCS: Physical Law embedded Generative Cloud Synthesis in Remote Sensing Images," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2025.3553239. 

# Project and software

- [Project homepage](https://liying-xu.github.io/PGCS/): method, synthesis results, cloud correction results, and software interface.
- Download PGCS for Windows: [Part 1](https://github.com/Liying-Xu/PGCS/releases/download/software-v1.0.0/PGCS-Software-1-of-2.zip) · [Part 2](https://github.com/Liying-Xu/PGCS/releases/download/software-v1.0.0/PGCS-Software-2-of-2.zip) · [SHA-256 checksums](https://github.com/Liying-Xu/PGCS/releases/download/software-v1.0.0/SHA256SUMS.txt). Download **both** ZIP parts, extract them into the same folder, then run `CloudSynUI.exe`. The software requires a compatible NVIDIA GPU/CUDA environment. See the [release notes](https://github.com/Liying-Xu/PGCS/releases/tag/software-v1.0.0) for details.
<img width="982" height="562" alt="image" src="https://github.com/user-attachments/assets/2f5310a4-b157-40bf-bf72-9ce3e8b056ea" />


# Code
Decirrus.py:This script is designed for the rapid removal of cirrus clouds from Landsat 8/9 imagery, corresponding to the PGCS_M.The required input data is top-of-atmosphere reflectance data.

Gen_dataset.py:This script can be used to generate an extensive "cloudy & cloud-free" paired dataset. It also supports radiometric calibration of DN (Digital Number) data and the cropping of ROI (Regions of Interest).

# Sample
The figures below show the effect of PGCS in the synthesis of cloud  and cloudy images. You can download the sample data from BaiduNetdisk. Link: https://pan.baidu.com/s/1ZpuB9qxVhHSOIxdMGM4Lhw?pwd=pgcs key：pgcs

|![image](https://github.com/user-attachments/assets/c2bf4141-c0a4-441b-ac2c-edaa7a24a327)|![image](https://github.com/user-attachments/assets/e25000b3-619d-4a20-b182-ba0a7837385f)|![image](https://github.com/user-attachments/assets/e0428cb9-fd6e-4640-a1df-6078f04dd3ca)|
|-----------------------------|-----------------------------|-----------------------------|

# Acknowledgments
Our code is inspired by [StyleGAN](https://github.com/NVlabs/stylegan3)
