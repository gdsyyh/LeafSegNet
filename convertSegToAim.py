import cv2
import os

# 指定文件夹路径
folder_path = 'dataset/ori'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否是 PNG 图像
    if filename.endswith('.png'):
        # 构建文件路径
        image_path = os.path.join(folder_path, filename)

        # 读取灰度图像
        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 检查是否成功读取图像
        if mask is None:
            print(f"无法读取图像: {image_path}")
            continue

        # 将像素值为 38 的位置置为 255
        mask[mask == 38] = 255

        # 保存修改后的图像
        modified_image_path = os.path.join('dataset/masks', f'{filename}')
        cv2.imwrite(modified_image_path, mask)

        print(f"处理完毕: {filename}, 已保存为: {modified_image_path}")
