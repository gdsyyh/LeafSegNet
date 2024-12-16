#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd


# In[2]:


from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device);


# In[ ]:


def checkcolour(masks, hsv):
    colours = np.zeros((0,3))

    for i in range(len(masks)):
        color = hsv[masks[i]['segmentation']].mean(axis=(0))
        colours = np.append(colours,color[None,:], axis=0)
        
    idx_green = (colours[:,0]<75) & (colours[:,0]>35) & (colours[:,1]>35)
    if idx_green.sum()==0:
        # grow lights on adjust
        idx_green = (colours[:,0]<100) & (colours[:,0]>35) & (colours[:,1]>35)
    
    return(idx_green)


# In[ ]:


def checkfullplant(masks):
    mask_all = np.zeros(masks[0]['segmentation'].shape[:2])

    for mask in masks:
        mask_all +=mask['segmentation']*1
        
    iou_withall = []
    for mask in masks:
        iou_withall.append(iou(mask['segmentation'], mask_all>0))
        
    idx_notall = np.array(iou_withall)<0.9
    return idx_notall


# In[ ]:


def getbiggestcontour(contours):
    nopoints = [len(cnt) for cnt in contours]
    return(np.argmax(nopoints))

def checkshape(masks):
    cratio = []

    for i in range(len(masks)):
        test_mask = masks[i]['segmentation']
        
        if not test_mask.max():
            cratio.append(0)
        else:

            contours,hierarchy = cv2.findContours((test_mask*255).astype('uint8'), 1, 2)

            # multiple objects possibly detected. Find contour with most points on it and just use that as object
            cnt = contours[getbiggestcontour(contours)]
            M = cv2.moments(cnt)

            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)

            (x,y),radius = cv2.minEnclosingCircle(cnt)

            carea = np.pi*radius**2

            cratio.append(area/carea)
    idx_shape = np.array(cratio)>0.1
    return(idx_shape)


# In[ ]:


def iou(gtmask, test_mask):
    intersection = np.logical_and(gtmask, test_mask)
    union = np.logical_or(gtmask, test_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return (iou_score)


# In[ ]:


def issubset(mask1, mask2):
    # is mask2 subpart of mask1
    intersection = np.logical_and(mask1, mask2)
    return(np.sum(intersection)/mask2.sum()>0.9)

def istoobig(masks):
    idx_toobig = []
    
    mask_all = np.zeros(masks[0]['segmentation'].shape[:2])

    for mask in masks:
        mask_all +=mask['segmentation']*1 

    for idx in range(len(masks)):
        if idx in idx_toobig:
            continue
        for idx2 in range(len(masks)):
            if idx==idx2:
                continue
            if idx2 in idx_toobig:
                continue
            if issubset(masks[idx2]['segmentation'], masks[idx]['segmentation']):
                # check if actually got both big and small copy delete if do
                if mask_all[masks[idx2]['segmentation']].mean() > 1.5:
                
                    idx_toobig.append(idx2)
    
    idx_toobig.sort(reverse=True)        
    return(idx_toobig)

def remove_toobig(masks, idx_toobig):
    masks_ntb = masks.copy()

    idx_del = []
    for idxbig in idx_toobig[1:]:
        maskbig = masks_ntb[idxbig]['segmentation'].copy()
        submasks = np.zeros(maskbig.shape)

        for idx in range(len(masks_ntb)):
            if idx==idxbig:
                continue
            if issubset(masks_ntb[idxbig]['segmentation'], masks_ntb[idx]['segmentation']):
                submasks +=masks_ntb[idx]['segmentation']

        if np.logical_and(maskbig, submasks>0).sum()/maskbig.sum()>0.9:
            # can safely remove maskbig
            idx_del.append(idxbig)
            del(masks_ntb[idxbig])
            
    return(masks_ntb)


# In[ ]:


def render_mask(masks):
    if len(masks) == 0:
        return None
    res = np.zeros([masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 3])
    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    for mask in sorted_masks:
        m = mask["segmentation"]
        res[:, :, 0][m] = np.random.randint(0, 255)
        res[:, :, 1][m] = np.random.randint(0, 255)
        res[:, :, 2][m] = np.random.randint(0, 255)
    res = res.astype(np.uint8)
    return res


# In[ ]:


def render_mask_black(masks):
    if len(masks) == 0:
        return None
    res = np.zeros([masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 3])
    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    for mask in sorted_masks:
        m = mask["segmentation"]
        res[:, :, 0][m] = 255
        res[:, :, 1][m] = 255
        res[:, :, 2][m] = 255
    res = res.astype(np.uint8)
    return res


# In[5]:


current_folder = '03202310/3/'
orgin_folder = 'autodl-tmp/所有图像/' + current_folder
image_folder = 'download_image/' + current_folder
mask_folder = 'download_mask/' + current_folder


# In[6]:


def get_output_mask(mask, filename):
    color_mask = render_mask_black(mask).astype(np.uint8)
    DEST_image = mask_folder + "complete/" + filename + ".png" 
    if color_mask is not None:
        print(DEST_image)
        cv2.imwrite(DEST_image, color_mask)


# In[7]:


# imnames = [x for x in os.listdir(folder) if '.jpg' in x]  # get list of image files change .JPG is using files of different type
# print(imnames)


# In[8]:


import GPUtil

# 获取所有可用GPU的信息
gpus = GPUtil.getGPUs()

for gpu in gpus:
    print(f"GPU: {gpu.name}")
    print(f"GPU占用率: {gpu.load * 100}%")
    print(f"显存使用率: {gpu.memoryUtil * 100}%")
    print(f"显存总量: {gpu.memoryTotal} MB")
    print(f"显存已用: {gpu.memoryUsed} MB")
    print(f"显存空闲: {gpu.memoryFree} MB")
    print(f"温度: {gpu.temperature} °C")
    print("-" * 40)


# In[13]:


import time
start_time = time.time()
imnames = [x for x in os.listdir(orgin_folder) if '.jpg' in x]  # get list of image files change .JPG is using files of different type
print(imnames)

for imname in imnames:
    print(imname)
    image = cv2.imread(orgin_folder + imname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,None,fx=0.5,fy=0.5)   # downsize image to fit on gpu easier may not be needed
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
           
    # use crop_n_layer=1 to improve results on smallest leaves 
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=200,  
    )

    # get masks
    masks = mask_generator.generate(image)
    
    # remove things that aren't green enough to be leaves
    idx_green = checkcolour(masks,hsv)

    masks_g = []
    for idx, use in enumerate(idx_green):
        if use:
            masks_g.append(masks[idx])

    if len(masks_g) > 2:

        # check to see if full plant detected and remove
        idx_notall = checkfullplant(masks_g)

        masks_na = []

        for idx, use in enumerate(idx_notall):
            if use:
                masks_na.append(masks_g[idx])

    else:
        masks_na = masks_g

    idx_shape = checkshape(masks_na)

    masks_s = []
    for idx, use in enumerate(idx_shape):
        if use:
            masks_s.append(masks_na[idx])

    idx_toobig = istoobig(masks_s)
    masks_ntb = remove_toobig(masks_s, idx_toobig)
    
    # save results at each step as npz file 
    np.savez(mask_folder + imname.replace('.jpg','leafonly_allmasks.npz'),
              masks, masks_g, masks_na, masks_s, masks_ntb)

    # outputs = [x for x in os.listdir(folder_out) if '.npz' in x]
    # print(outputs)
    # output_image = np.load(folder_out + outputs[0], allow_pickle=True)
    get_output_mask(masks_ntb, imname[:-4])
    break
end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time} 秒")


# In[40]:


import cv2
import os

imnames = [x for x in os.listdir(orgin_folder) if '.jpg' in x]  # get list of image files change .JPG is using files of different type
print(imnames)
image_count = 0
for imname in imnames:
    # 加载原图像和mask
    print(imname)
    image = cv2.imread(orgin_folder + imname)
    image = cv2.resize(image,None,fx=0.5,fy=0.5)
    mask = cv2.imread(mask_folder + "complete/" + imname[:-4] + ".png")
    
    if mask is None :
        print("dont have mask")
        continue

    # 确保image和mask的大小相同
    assert image.shape == mask.shape, "Image and mask should have the same dimensions."

    # 定义截取小图片的大小
    crop_size = 256

    imagetime = "02202310_7_"
    filename = "image_" + imagetime + str(image_count)
    cv2.imwrite(os.path.join(image_folder + "complete/",  filename + ".png"), image)
    image_count = image_count + 1
    # 遍历原图像和mask，截取小图片
    count = 0
    for y in range(0, image.shape[0], crop_size):
        for x in range(0, image.shape[1], crop_size):
            # 获取截取区域的坐标
            x_end = min(x + crop_size, image.shape[1])
            y_end = min(y + crop_size, image.shape[0])

            # 截取原图像和mask中对应位置的区域
            cropped_image = image[y:y_end, x:x_end]
            cropped_mask = mask[y:y_end, x:x_end]

            # 保存截取的小图片
            output_folder = "cropped_images/image_" + str(y) + "_" + str(x) + "/";
            os.makedirs(output_folder, exist_ok=True)
            cv2.imwrite(os.path.join(image_folder, filename + "_" + str(count) + ".png"), cropped_image)
            cv2.imwrite(os.path.join(mask_folder, filename + "_" + str(count) + ".png"), cropped_mask)
            count = count + 1


# In[14]:


outputs = [x for x in os.listdir(mask_folder) if '.npz' in x]
print(outputs)


# In[45]:


output_image = np.load(mask_folder + outputs[0], allow_pickle=True)

# get_output_image(0, output_image)
# get_output_image(1, output_image)
# get_output_image(2, output_image)
# get_output_image(3, output_image)
get_output_image(4, output_image)

# thres = 0.75
# image = cv2.imread(folder + imnames[0])
# image = cv2.resize(image,None,fx=0.5,fy=0.5)
# render_img = (image * thres + color_mask * (1 - thres)).astype(np.uint8)


# In[24]:


output_image = np.load(mask_folder + outputs[0], allow_pickle=True)
print(output_image.files)
out1 = output_image['arr_4']
get_output_mask(out1, "mask4")
# color_mask = render_mask(out1)
# thres = 0.75
# image = cv2.imread(orgin_folder + "1697077006_1697076663_4.jpg")
# image = cv2.resize(image,None,fx=0.5,fy=0.5)
# render_img = (image * thres + color_mask * (1 - thres)).astype(np.uint8)
# DEST_image = mask_folder + "test.png"  
# if render_img is not None:
#     cv2.imwrite(DEST_image, render_img)


# In[4]:


import os
import cv2

def delete_images_with_wrong_size(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 拼接文件路径
        file_path = os.path.join(folder_path, filename)
        
        # 判断是否为文件
        if os.path.isfile(file_path):
            # 尝试读取图片
            image = cv2.imread(file_path)
            if image is not None:
                # 获取图片尺寸
                height, width, _ = image.shape
                # 如果尺寸不是256x256像素，则删除文件
                if height != 256 or width != 256:
                    os.remove(file_path)
                    print(f"删除文件：{file_path}")
            else:
                print(f"无法读取图像文件：{file_path}")

# 指定文件夹路径
folder_path = image_folder

# 调用函数删除不符合尺寸要求的图片
delete_images_with_wrong_size(folder_path)


# In[ ]:




