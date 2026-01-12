import cv2
import os
import glob
import json
import shutil
import numpy as np


def save_labelme_format(image_path, bbox, output_dir, label):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return

    h, w = img.shape[:2]
    shapes = []

    points = [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]],
              [bbox[0], bbox[1] + bbox[3]]]
    shape = {
        "label": label,
        "points": points,
        "group_id": None,
        "shape_type": "polygon",
        "flags": {}
    }
    shapes.append(shape)

    labelme_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }

    json_name = os.path.splitext(os.path.basename(image_path))[0] + '.json'
    with open(os.path.join(output_dir, json_name), 'w') as f:
        json.dump(labelme_data, f, indent=4)


def augment_image(image):
    rows, cols = image.shape[:2]

    # 随机旋转
    angle = np.random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows), borderValue=(255, 255, 255))

    # 随机仰角变化
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[50, 50], [200, 50 + np.random.uniform(-50, 50)], [50, 200 + np.random.uniform(-50, 50)]])
    M = cv2.getAffineTransform(pts1, pts2)
    skewed_image = cv2.warpAffine(rotated_image, M, (cols, rows), borderValue=(255, 255, 255))

    return skewed_image


def find_black_bbox(image):
    h, w = image.shape[:2]

    # 假设纸片大致在中心区域（根据你实际拍摄结构可调整）
    x_start = int(w * 0.2)
    x_end = int(w * 0.8)
    y_start = int(h * 0)
    y_end = int(h * 1)

    # 提取ROI区域
    roi = image[y_start:y_end, x_start:x_end]

    # 只在ROI区域内找“很黑”的像素
    black_mask = np.all(roi < 50, axis=2)  # RGB 全都小于 50 为黑色

    coords = np.column_stack(np.where(black_mask))
    if coords.shape[0] == 0:
        return None

    # 注意 coords 是 ROI 内的坐标，我们要映射回全图坐标
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    x = int(x_min + x_start)
    y = int(y_min + y_start)
    w = int(x_max - x_min + 1)
    h = int(y_max - y_min + 1)

    return x, y, w, h



def augment_images(folders, output_dir, target_count=350):
    os.makedirs(output_dir, exist_ok=True)

    for folder_index, folder in enumerate(folders, start=1):
        image_paths = glob.glob(os.path.join(folder, '*.jpg'))
        print(f"Processing folder: {folder}, found {len(image_paths)} images.")

        if len(image_paths) == 0:
            continue

        # 每个文件夹需要生成的增强图片数量
        num_augmented_per_image = target_count // len(image_paths)

        folder_output_dir = os.path.join(output_dir, f"folder_{folder_index}")
        os.makedirs(folder_output_dir, exist_ok=True)

        for image_path in image_paths:
            print(f"Processing image: {image_path}")
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image: {image_path}")
                continue

            shutil.copy(image_path, folder_output_dir)

            for i in range(num_augmented_per_image):
                augmented_image = augment_image(img)
                aug_image_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_aug_{i}.jpg"
                aug_image_path = os.path.join(folder_output_dir, aug_image_name)
                cv2.imwrite(aug_image_path, augmented_image)

        # 检查并处理不足的情况
        current_augmented_images = glob.glob(os.path.join(folder_output_dir, '*.jpg'))
        while len(current_augmented_images) < target_count:
            for image_path in image_paths:
                img = cv2.imread(image_path)
                if img is None:
                    continue
                augmented_image = augment_image(img)
                aug_image_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_aug_extra_{len(current_augmented_images)}.jpg"
                aug_image_path = os.path.join(folder_output_dir, aug_image_name)
                cv2.imwrite(aug_image_path, augmented_image)
                current_augmented_images.append(aug_image_path)
                if len(current_augmented_images) >= target_count:
                    break

        # 标注生成的图像
        annotate_images(folder_output_dir, folder_index)


def annotate_images(output_dir, label):
    image_paths = glob.glob(os.path.join(output_dir, '*.jpg'))
    print(f"Annotating {len(image_paths)} images in folder: {output_dir}.")
    for image_path in image_paths:
        print(f"Annotating image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}")
            continue

        bbox = find_black_bbox(img)
        if bbox:
            save_labelme_format(image_path, bbox, output_dir, str(label))
        else:
            print(f"No valid bounding box found in image {image_path}.")


if __name__ == '__main__':
    base_dir = os.getcwd()
    folders = [os.path.join(base_dir, "photo", "images", str(i)) for i in range(1, 9)]
    output_dir = os.path.join(base_dir, "photo", "images_processed_2")

    augment_images(folders, output_dir)