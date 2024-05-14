import pydicom
import skimage.transform as transform
import copy
import os
import numpy as np
import cv2
from skimage import feature, exposure
from PIL import Image
import matplotlib.pyplot as plt

'''
------------------------------------------------------------------------------------------------------------------------
导入PET-CT图像，读取图像数据后均衡化处理
------------------------------------------------------------------------------------------------------------------------
'''
def over_lap(x1,y1,x2,y2):
    if max(x1,y1) <= min(x2,y2) or min(x1,y1) >= max(x2,y2):
        return 0
    else:
        delta = max(max(x1,y1) - min(x1,y1), max(x2,y2) - min(x2,y2))
        op = min(abs(y2 - x1) + abs(y1 - x2), abs(y2 - y1) + abs(x2 - x1))
        if (delta - op)/delta < 0.2:
            return 0
        else:
            return 1
def dcm2png(file):
    #读取原文件并对原来文件的部分内容进行展示
    print('FIle:', file)
    ds = pydicom.dcmread(file, force=True)
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ori_img = np.array(ds.pixel_array)

    sharp = ori_img.shape
    _h = sharp[0]
    _w = sharp[1]
    if len(sharp) == 3:
        ori_img = ori_img[:, :, 0]
    img = transform.resize(ori_img, (256, 256))
    img1024 = transform.resize(ori_img, (_h, _w))

    start = img.min()
    end = img.max()
    print("img shape:", img.shape)

    vals = img.flatten()
    counts, bins = np.histogram(vals, bins=256) # 一个参数是数据集，第二个参数是划分数据的bins个数或边缘值
    temp_counts = copy.copy(counts) # 第一个频数的列表，第二个元素是数据的范围。
    print(len(counts), len(bins))

    hist_accum = np.zeros(256)
    temp_sum = 0
    for i in range(2, len(counts) - 2): # 计算累积直方图
        temp_sum = temp_sum + counts[i]
        hist_accum[i] = temp_sum / (65536.0 - counts[0] - counts[1] - counts[-2] - counts[-1])
        # print(i, ':', hist_accum[i])

    # 找到直方图中分布在0.3~0.7之间的像素值范围
    try:
        count_start = np.where(hist_accum > 0.3)[0][0]
        count_end = np.where(hist_accum > 0.7)[0][0]
    except Exception as e:
        print("get histgram start and end error:", e)
    # 将temp_counts中分布在count_start和count_end之外的像素值频数设为0
    temp_counts[0:count_start] = 0
    temp_counts[count_end:256] = 0

    max_v = temp_counts.max()    # 获取temp_counts中的最大值
    loc = np.where(counts == max_v) # 获取max_v在counts中的位置

    hist_sum_left = sum(counts[0: loc[0][0] + 1])
    hist_sum_right = 65536 - hist_sum_left
    # 根据左右两侧像素值的频数比例，确定start和end的值
    for inx in range(loc[0][0], 1, -1):
        left_ratio = hist_sum_left / 65536
        if counts[inx] + counts[inx - 1] < 50 and left_ratio < 0.05:
            start = (bins[inx] + bins[inx + 1]) / 2.0
            break
        hist_sum_left = hist_sum_left - counts[inx]

    for inx in range(loc[0][0] + 1, 255):
        right_ratio = hist_sum_right / 65536
        if counts[inx] + counts[inx + 1] < 50 and right_ratio < 0.05:
            end = (bins[inx] + bins[inx + 1]) / 2.0
            break
        hist_sum_right = hist_sum_right - counts[inx]

    try:
        print('WC type:', type(ds.WindowCenter).__name__)
        if type(ds.WindowCenter).__name__ == 'DSfloat':
            wc = int(ds.WindowCenter)
            wl = int(ds.WindowWidth)
        elif type(ds.WindowCenter).__name__ == 'MultiValue':
            wc = int(ds.WindowCenter[0])
            wl = int(ds.WindowWidth[0])
        low = int(wc - wl / 2)
        up = int(wc + wl / 2)
        print("low,up:", low, up)

        # 如果直方图分布范围太小，则使用窗位和窗宽参数代替start和end的值
        if count_end - count_start < 4:
            start = low
            end = up
        # 如果start和end与窗位和窗宽参数有重叠，则使用窗位和窗宽参数代替start和end的值
        elif over_lap(low, up, start, end):
            start = low
            end = up
    except Exception as e:
        print(e, " get window error")


    # 将img1024中小于start的像素值设为start，大于end的像素值设为end，并将像素值归一化到[0,255]范围内
    img1024[img1024 < start] = start
    img1024[img1024 > end] = end
    img1024 = np.array((img1024 - start) * 255.0 / (end - start))
    if hasattr(ds, 'PhotometricInterpretation'):
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            img1024 = 255 - img1024


    img1024 = img1024.astype(np.uint8)

    #显示原图像直方图
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(ori_img.flatten(), 128)
    plt.title('ordinal')
    plt.xlim([0, 2300])
    plt.ylim([0, 30000])
    #显示均衡化之后图像直方图
    plt.subplot(1, 2, 2)
    plt.hist(img1024.flatten(), 128)
    plt.title('junhenghua')
    plt.xlim([0, 254])
    plt.ylim([0, 30000])
    plt.show()


    cv2.imwrite(r'junhen.png', img1024)
    cv2.imshow('junhen', img1024)



dcm2png('1-01.dcm')


'''
------------------------------------------------------------------------------------------------------------------------
对得到的均衡化后的图像进行hog特征提取
------------------------------------------------------------------------------------------------------------------------
'''
#导入图片
image = cv2.imread('junhen.png')

#feature.hog函数对传入的图像完成了计算图像分布直方图、block归一化、计算hog特征向量的处理
fd, hog_image = feature.hog(image, orientations=256, pixels_per_cell=(9,9),
                    cells_per_block=(2, 2), visualize=True,channel_axis=2)

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))


#提取特征向量之后的照片
cv2.imshow('hog', hog_image_rescaled)
cv2.waitKey(0)==ord('q')