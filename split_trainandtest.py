import os
import random
import shutil

# 设置图片文件夹路径和保存文件名
image_folder = 'dataset/images'  # 替换为实际的图片文件夹路径
train_file = 'dataset/train.txt'
test_file = 'dataset/test.txt'

# 设置训练集和测试集的比例
train_ratio = 0.9  # 80%作为训练集，20%作为测试集

# 获取所有图片的文件名列表
image_names = os.listdir(image_folder)
random.shuffle(image_names)

# 计算划分位置
split_index = int(len(image_names) * train_ratio)

# 划分训练集和测试集
train_names = image_names[:split_index]
test_names = image_names[split_index:]

# 写入训练集文件
with open(train_file, 'w') as train_txt:
    for name in train_names:
        print("name:",name.split(".")[0])
        train_txt.write(name.split(".")[0] + '\n')

# 写入测试集文件
with open(test_file, 'w') as test_txt:
    for name in test_names:

        test_txt.write(name.split(".")[0] + '\n')

# 移动图片到相应的文件夹
train_folder = os.path.join(image_folder, 'train')
test_folder = os.path.join(image_folder, 'test')
# os.makedirs(train_folder, exist_ok=True)
# os.makedirs(test_folder, exist_ok=True)

# for name in train_names:
#     src_path = os.path.join(image_folder, name)
#     dest_path = os.path.join(train_folder, name)
#     shutil.move(src_path, dest_path)
#
# for name in test_names:
#     src_path = os.path.join(image_folder, name)
#     dest_path = os.path.join(test_folder, name)
#     shutil.move(src_path, dest_path)

print("训练集和测试集划分完成，并已保存到相应的文件和文件夹中。")
