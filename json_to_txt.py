import os
import json
import glob
import shutil
import random

# 定义类别名称到索引的映射
class_to_id = {
    'rectangle': 0,
    'line': 1,
}

base_dir = os.getcwd()

# 标注文件所在的文件夹路径
json_folder_path = os.path.join(base_dir, "photo", "images", "in_use")

# 输出标签文件的文件夹路径
output_base_path = os.path.join(base_dir, "YOLOv5-Lite", "data")
# 创建输出子目录
os.makedirs(os.path.join(output_base_path, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(output_base_path, 'train', 'labels'), exist_ok=True)
os.makedirs(os.path.join(output_base_path, 'valid', 'images'), exist_ok=True)
os.makedirs(os.path.join(output_base_path, 'valid', 'labels'), exist_ok=True)

image_files = glob.glob(os.path.join(json_folder_path, "**", "*.jpg"), recursive=True)
print(f"找到 {len(image_files)} 个图片文件")

for image_path in image_files:
    # 获取文件名（不含扩展名）
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    # 根据图片路径推导对应的 JSON 路径
    json_file_path = os.path.join(os.path.dirname(image_path), f"{image_id}.json")

    # 获取图像文件名（不含扩展名）
    image_id = os.path.splitext(os.path.basename(json_file_path))[0]
    image_path = os.path.join(os.path.dirname(json_file_path), f"{image_id}.jpg")

    # 判断是放入训练集还是验证集
    subset = 'train' if random.random() < 0.9 else 'valid'

    # 确定输出文件路径
    output_image_path = os.path.join(output_base_path, subset, 'images', f"{image_id}.jpg")
    output_label_path = os.path.join(output_base_path, subset, 'labels', f"{image_id}.txt")

    # 复制图像文件到输出目录
    shutil.copy(image_path, output_image_path)

    # 写入TXT标签文件
    with open(output_label_path, 'w') as txt_file:
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                data = json.load(f)

            for shape in data['shapes']:
                if shape['shape_type'] in ['polygon', 'rectangle']:
                    # 计算多边形标注的最小和最大点来确定边界框
                    min_x = min(shape['points'], key=lambda x: x[0])[0]
                    max_x = max(shape['points'], key=lambda x: x[0])[0]
                    min_y = min(shape['points'], key=lambda x: x[1])[1]
                    max_y = max(shape['points'], key=lambda x: x[1])[1]

                    # 计算边界框的中心点和宽度、高度
                    x_center = (min_x + max_x) / 2
                    y_center = (min_y + max_y) / 2
                    width = max_x - min_x
                    height = max_y - min_y

                    # 归一化坐标
                    x_center /= data['imageWidth']
                    y_center /= data['imageHeight']
                    width /= data['imageWidth']
                    height /= data['imageHeight']

                    # 获取类别索引
                    class_index = class_to_id[shape['label']]

                    # 写入TXT文件
                    txt_file.write(f'{class_index} {x_center} {y_center} {width} {height}\n')

print("所有JSON文件的转换完成！")