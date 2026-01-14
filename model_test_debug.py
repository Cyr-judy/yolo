import cv2
import numpy as np
import torch
import time
import random


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def debug_postprocess(predictions, img_shape, conf_threshold=0.6, iou_threshold=0.5):
    """
    调试版后处理函数，添加详细输出
    """
    orig_h, orig_w = img_shape[:2]
    
    # print(f"原始预测数量: {len(predictions)}")
    # print(f"预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # 过滤低置信度的检测
    conf_mask = predictions[..., 4] > conf_threshold
    predictions = predictions[conf_mask]
    
    # print(f"置信度过滤后数量: {len(predictions)}")
    
    if len(predictions) == 0:
        return [], [], []
    
    # 获取类别概率
    class_scores = predictions[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    class_confidences = np.max(class_scores, axis=1)
    
    # print(f"类别ID分布: {np.bincount(class_ids)}")
    # print(f"类别置信度: {class_confidences}")
    
    # 计算边界框坐标
    boxes = predictions[:, :4]
    
    # 转换为xyxy格式
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    # 缩放到原始图像尺寸
    scale_x = orig_w / 320.0
    scale_y = orig_h / 320.0
    
    x1 = x1 * scale_x
    y1 = y1 * scale_y
    x2 = x2 * scale_x
    y2 = y2 * scale_y
    
    # 确保坐标在合理范围内
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)
    
    # 组合边界框
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    
    # print(f"NMS前边界框: {boxes_xyxy}")
    
    # 应用NMS - 使用更低的IOU阈值
    keep_indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        class_confidences.tolist(),
        conf_threshold,
        iou_threshold
    )
    
    # print(f"NMS保留索引: {keep_indices}")
    #
    if len(keep_indices) > 0:
        keep_indices = keep_indices.flatten()
        final_boxes = boxes_xyxy[keep_indices]
        final_scores = class_confidences[keep_indices]
        final_class_ids = class_ids[keep_indices]
        
        print(f"最终检测结果: {len(final_boxes)} 个目标")
        return final_boxes, final_scores, final_class_ids
    else:
        return [], [], []


def infer_img_debug(img0, model, conf_threshold=0.6):
    """
    调试版推理函数
    """
    # 图像预处理
    img = cv2.resize(img0, [320, 320], interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    # 转换为torch tensor
    blob = torch.from_numpy(blob).to(device)

    # 模型推理
    with torch.no_grad():
        predictions = model(blob)[0].cpu().numpy().squeeze(axis=0)

    # print(f"模型输出形状: {predictions.shape}")
    
    # 后处理
    boxes, scores, class_ids = debug_postprocess(predictions, img0.shape, conf_threshold=conf_threshold)

    return boxes, scores, class_ids


if __name__ == "__main__":

    # 模型加载
    weights_path = r"C:\Users\cheny\Desktop\yolo\YOLOv5-Lite-1.4\YOLOv5-Lite\runs\train\exp11\weights\best.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(weights_path, map_location=device, weights_only=False)['model'].float().fuse().eval().to(device)

    # 标签字典
    dic_labels = {
        0: 'rectangle',
        1: 'line',
    }

    video = 0
    cap = cv2.VideoCapture(video)
    flag_det = False
    
    print("按 's' 开始/停止检测，按 'q' 退出")
    print("按 '1' 降低置信度阈值，按 '2' 提高置信度阈值")
    print("按 'd' 开启/关闭详细调试输出")
    
    # 可调节的参数
    current_conf_threshold = 0.6
    debug_mode = False
    
    while True:
        success, img0 = cap.read()

        if success:
            if flag_det:
                t1 = time.time()
                
                if debug_mode:
                    print("\n" + "="*50)
                    print("开始新的检测")
                
                det_boxes, scores, ids = infer_img_debug(img0, model, conf_threshold=current_conf_threshold)
                t2 = time.time()

                # if not debug_mode:
                #     print(f"检测到 {len(det_boxes)} 个目标 (置信度阈值: {current_conf_threshold})")
                
                for i, (box, score, id) in enumerate(zip(det_boxes, scores, ids)):
                    if not debug_mode:
                        print(f"目标 {i+1}: 类别={dic_labels.get(id, 'unknown')}, 置信度={score:.3f}, 边界框={box}")
                    
                    if id in dic_labels:
                        label = '%s:%.2f' % (dic_labels[id], score)
                        plot_one_box(box.astype(np.int16), img0, color=(255, 0, 0), label=label, line_thickness=None)
                print("——————————————————————————————————————————————————")

                str_FPS = "FPS: %.2f" % (1. / (t2 - t1))
                cv2.putText(img0, str_FPS, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img0, f"Conf: {current_conf_threshold}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img0, f"Debug: {'ON' if debug_mode else 'OFF'}", (10, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", img0)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key & 0xFF == ord('s'):
            flag_det = not flag_det
            print(f"检测 {'开启' if flag_det else '关闭'}")
        elif key & 0xFF == ord('1'):
            current_conf_threshold = max(0.1, current_conf_threshold - 0.1)
            print(f"置信度阈值降低到: {current_conf_threshold}")
        elif key & 0xFF == ord('2'):
            current_conf_threshold = min(0.9, current_conf_threshold + 0.1)
            print(f"置信度阈值提高到: {current_conf_threshold}")
        elif key & 0xFF == ord('d'):
            debug_mode = not debug_mode
            print(f"调试模式 {'开启' if debug_mode else '关闭'}")

    cap.release()
    cv2.destroyAllWindows() 