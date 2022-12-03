import time
from base_camera import BaseCamera
import numpy as np
import cv2
import os
import sys
import argparse
from models.experimental import attempt_load
from utils.datasets import *
from utils.utils import *
from utils.general import  apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
import torch.backends.cudnn as cudnn
from utils.torch_utils import load_classifier, select_device, time_sync


class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,default='weights\\best.pt',help='model.pt path(s)')  # 模型权重
        #parser.add_argument('--weights', nargs='+', type=str,default='weights\\yolov5s.pt',help='model.pt path(s)')  # 模型权重
        parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='models.yaml path')
        parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # 输出位置，目前不用
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],help='inference size h,w')
        # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')  # 输入的图片大小
        parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')  # 置信域值
        parser.add_argument('--iou-thres', type=float, default=0.2, help='IOU threshold for NMS')  # nms的置信域值
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # 设备，当然是0了
        parser.add_argument('--view-img', default=True, help='display results')  # 是否显示
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')  # 保存预测框
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')  # 预测是否采用数据增强
        parser.add_argument('--update', action='store_true', help='update all models')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print(opt)

        # 导入相关数据：输出位置，资源（视频资源），权重，.....
        out, weights, view_img, save_txt, imgsz = \
            'inference/output','D:\\anaconda\project\\flask_Camera\weights\\insectbest.pt', True,'store_true', 640
        # cfg='models/yolov5s.yaml'
        # 是否使用webcam，网页数据
        source = '0'
        webcam = source == '0'

        project='D:\\anaconda\project\\flask_Camera\\runs\detect'
        name = 'exp'
        visualize=False

        # 加载摄像头
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:

            save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

            device = torch_utils.select_device()  # 用于自动选择本机模型训练的设备（cpu or gpu）
            if os.path.exists(out):
                shutil.rmtree(out)  # delete output folder
            os.makedirs(out)  # make new output folder

            # Load model
            w = str(weights[0] if isinstance(weights, list) else weights)
            classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
            check_suffix(w, suffixes)  # check weights have acceptable suffix
            pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)
            stride=64

            if pt:
                model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
                stride = int(model.stride.max())  # model stride
                names = model.module.names if hasattr(model, 'module') else model.names  # get class names


            # model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
            # stride = int(model.stride.max())
            # # model = torch.load(weights, map_location=device)['model']  # 模型加载
            # model.to(device).eval()

            # Half precision
            # half = False and device.type != 'cpu'

            half = True and device.type != 'cpu'
            print('half = ' + str(half))

            if half:
                model.half()  # 低精度模型加速
            imgsz = check_img_size(imgsz, s=stride)
            # Set Dataloader
            if webcam:
                # view_img = True
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz,stride=stride)
                bs=len(dataset)
            else:
                # save_img = True
                dataset = LoadImages(source, img_size=imgsz)
            vid_path, vid_writer = [None] * bs, [None] * bs

            names = model.modules.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

            # Run inference
            # t0 = time.time()
            # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
            # stride=int(model.stride.max())
            # imgsz=check_img_size(imgsz,stride)
            # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

            # if pt and device.type != 'cpu':
            #     model(torch.zeros(1, 3, imgsz).to(device).type_as(next(model.parameters())))
            dt, seen = [0.0, 0.0, 0.0], 0

            for path, img, im0s, vid_cap in dataset:
                # img = torch.from_numpy(img).to(device)
                # img = img.half() if half else img.float()  # uint8 to fp16/32
                # img /= 255.0  # 0 - 255 to 0.0 - 1.0 归一化
                # if img.ndimension() == 3:
                #     img = img.unsqueeze(0)

                t1 = time_sync()
                if onnx:
                    img = img.astype('float32')
                else:
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                img = img / 255.0  # 0 - 255 to 0.0 - 1.0
                if len(img.shape) == 3:
                    img = img[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1


                # Inference
                if pt:
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    pred = model(img, augment=False, visualize=visualize)[0]


                # t1 = torch_utils.time_synchronized()  # 前向推理
                # pred = model(img, augment=False)[0]
                t3 = time_sync()
                dt[1] += t3 - t2

                # Apply NMS         cesium
                # pred = non_max_suppression(pred, 0.4, 0.5,
                #                            fast=True, classes=None, agnostic=False)
                # t2 = torch_utils.time_synchronized()  # 前向推理

                pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False, max_det=1000)
                dt[2] += time_sync() - t3

                for i, det in enumerate(pred):  # detections per image
                    # p, s, im0 = path, '', im0s
                    seen+=1
                    if webcam:  # batch_size >= 1
                        p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                    else:
                        p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                    p=Path(p)
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        # for c in det[:, -1].unique():  #probably error with torch 1.5
                        for c in det[:, -1].detach().unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %s, ' % (n, names[int(c)])  # add to string
                            print("3")

                        for *xyxy, conf, cls in det:
                            label = '%s %.2f' % (names[int(cls)], conf)
                            print("2")
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    print(det)
                    print('%sDone. (%.3fs)' % (s, t2 - t1))

                yield cv2.imencode('.jpg', im0)[1].tobytes()