import cv2
import numpy as np
import os

# 设置文件夹路径
folder_path = 'E:/python1/Trans/test_results_M'  # 替换为你的文件夹路径
output_folder = 'E:/python1/Trans/test_results_M_mask'  # 输出文件夹路径

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的所有 PNG 图像
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        # 读取图像
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像

        # 应用阈值处理
        _, thresh_img = cv2.threshold(img, 0.6 * 255, 255, cv2.THRESH_BINARY)

        # 将阈值处理后的图像保存到输出文件夹
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, thresh_img)

print("阈值处理完成！")