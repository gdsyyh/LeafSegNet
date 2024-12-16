
import os
import time
import random
import numpy as np
import cv2
from tqdm import tqdm
import torch
from sklearn.utils import shuffle

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Load the Kvasir-SEG dataset """
def load_data(path):
    def load_names(path, file_path):
        f = open(file_path, "r")
        data = f.read().split("\n")[:-1]
        #加载数据需要根据后缀进行替换
        #####################################################################
        images = [os.path.join(path,"images", name) + ".png" for name in data]
        masks = [os.path.join(path,"masks", name) + ".png" for name in data]
        #####################################################################
        return images, masks

    train_names_path = f"{path}/train.txt" ######要求这里的txt文件要比原先多一行回车
    valid_names_path = f"{path}/test.txt"######要求这里的txt文件要比原先多一行回车

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    return (train_x, train_y), (valid_x, valid_y)

""" Shuffle the dataset. """
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")

def rle_encode(x):
    '''
    x: numpy array of shape (height, width), 1 - ori, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - ori, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

""" Initial ori build using Otsu thresholding. """
def init_mask(images, size):
    def otsu_mask(image, size):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        th = th.astype(np.int32)
        th = th/255.0
        th = th > 0.5
        th = th.astype(np.int32)
        return img, th

    mask = []
    for image in tqdm(images, total=len(images)):
        name = image.split("/")[-1]
        i, m = otsu_mask(image, size)
        cv2.imwrite(f"ori/{name}", np.concatenate([i, m*255], axis=1))
        mask.append(m)
    return mask

def init_edge(images, size):
    def edge_mask(image, size):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # 水平边缘
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # 垂直边缘
        # 计算梯度幅值
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        # 将梯度幅值缩放到0-255范围
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return gradient_magnitude
    mask = []
    for image in tqdm(images, total=len(images)):
        name = image.split("/")[-1]
        m = edge_mask(image, size)
        # cv2.imwrite(f"ori/{name}", np.concatenate([i, m*255], axis=1))
        mask.append(m)
    return mask

def init_gray(images, size):
    def gray_mask(image, size):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        return img
    mask = []
    for image in tqdm(images, total=len(images)):
        m = gray_mask(image, size)
        mask.append(m)
    return mask
