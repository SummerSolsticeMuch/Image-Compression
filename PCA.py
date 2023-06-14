from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog


def loadImage(path, mode):
    if mode == 'grey':
        img = Image.open(path).convert('L')
    elif mode == 'color':
        img = Image.open(path).convert('RGB')
    # 转为矩阵
    data = np.asarray(img)
    return data


def pca_single(data, k):
    # 求图片每一行的均值
    mean = np.array([np.mean(data[:, index]) for index in range(data.shape[1])])
    
    # 去中心化
    normal_data = data - mean
    
    # 得到协方差矩阵
    matrix = np.dot(np.transpose(normal_data), normal_data)
    
    # eig_val存储特征值，eig_vec存储对应的特征向量
    eig_val, eig_vec = np.linalg.eig(matrix)
    
    # 对矩阵操作，按从小到大的顺序对应获得此数的次序（从0开始）
    eig_index = np.argsort(eig_val)

    # 取前k个大特征值的下标
    eig_vec_index = eig_index[:-(k+1):-1]

    # 取前k个大特征值的特征向量
    feature = eig_vec[:, eig_vec_index]

    # 将特征值与对应特征向量矩阵乘得到最后的pca降维图
    new_data = np.dot(normal_data, feature)
    
    # 将降维后的数据映射回原空间
    return np.dot(new_data, np.transpose(feature)) + mean

def pca(arr, k, mode):
    global size
    if mode == 'grey':
        rec_data = pca_single(arr, k)
        # 得到降维后的图片
        newImage = Image.fromarray(np.uint8(rec_data))

    elif mode == 'color':
        # 以列表存储原图R,G,B值
        rgb = [arr[:,:,i] for i in range(3)]
        # 降维后的R,G,B值
        rec_data = []
        
        for data in rgb:    # 分别处理R,G,B
            rec_data.append(pca_single(data, k))
        # 得到降维后的图片
        new = np.array(rec_data).transpose((1, 2, 0))
        newImage = Image.fromarray(np.uint8(new)) 
    
    else:
        raise KeyboardInterrupt

    fpath = 'C:/Users/xzd66001/Desktop/my/Learning/机器学习及其安全应用/PCA/' + mode + '_k=' + str(k) + '.jpg'
    newImage.convert('RGB').save(fpath)
    # 计算压缩率
    rate = round(os.path.getsize(fpath) / size, 5)
    os.remove(fpath)
    fpath = 'C:/Users/xzd66001/Desktop/my/Learning/机器学习及其安全应用/PCA/' + mode + '_k=' + str(k) + '  ' + str(rate) + '.jpg'
    newImage.convert('RGB').save(fpath)
    
    

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    k = int(input('0.灰度图像处理\t1.彩色图像处理\n请选择处理模式:'))
    if k == 0:
        mode = 'grey'
    elif k == 1:
        mode = 'color'
    else:
        raise KeyboardInterrupt
    # 获取文件夹路径
    print('请选择图片')
    f_path = filedialog.askopenfilename()
    size = os.path.getsize(f_path)
    img = loadImage(f_path, mode)
    
    for i in tqdm(range(1, img.shape[1]+1)):
        pca(img, i, mode)

    print('处理完成,请查看PCA文件夹')
