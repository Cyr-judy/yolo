import sys
import time
import torch
import torch.nn as nn

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device


def export_model(weights_path, img_size=(320, 320), batch_size=1, device='cpu', dynamic=False, grid=False, concat=True):
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(device)
    model = attempt_load(weights_path, map_location=device)  # load FP32 model

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    img_size = [check_img_size(x, gs) for x in img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(batch_size, 3, *img_size).to(device)  # image size(1,3,320,320) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, models.yolo.Detect):
            m.forward = m.cat_forward if concat else m.forward  # assign forward (optional)
    model.model[-1].export = not grid  # set Detect() layer grid export
    print(model.model[-1])
    y = model(img)  # dry run

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = weights_path.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                                        'output': {0: 'batch', 2: 'y', 3: 'x'}} if dynamic else None)

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))


if __name__ == '__main__':
    weights_path = r"C:\Users\cheny\Desktop\yolo\YOLOv5-Lite-1.4\YOLOv5-Lite\runs\train\exp10\weights\best.pt"  # 直接指定权重路径
    export_model(weights_path)