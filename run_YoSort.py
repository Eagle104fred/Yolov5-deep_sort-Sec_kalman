import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from socket_function import UDP_connect
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from tools import Tools



# palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
#
# #预测框解码，返回框中点和宽高
# def bbox_rel(*xyxy):
#     """" Calculates the relative bounding box from absolute pixel values. """
#     bbox_left = min([xyxy[0].item(), xyxy[2].item()])
#     bbox_top = min([xyxy[1].item(), xyxy[3].item()])
#     bbox_w = abs(xyxy[0].item() - xyxy[2].item())
#     bbox_h = abs(xyxy[1].item() - xyxy[3].item())
#     x_c = (bbox_left + bbox_w / 2)
#     y_c = (bbox_top + bbox_h / 2)
#     w = bbox_w
#     h = bbox_h
#     return x_c, y_c, w, h
#
#
# def compute_color_for_labels(label):
#     """
#     Simple function that adds fixed color depending on the class
#     """
#     color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
#     return tuple(color)
#
#
# def draw_boxes(img, bbox, clses,confs,names,identities=None,offset=(0, 0)):
#     for i, box in enumerate(bbox):
#         x1, y1, x2, y2 = [int(i) for i in box]
#         x1 += offset[0]
#         x2 += offset[0]
#         y1 += offset[1]
#         y2 += offset[1]
#         # box text and bar
#         id = int(identities[i]) if identities is not None else 0
#         color = compute_color_for_labels(id)
#         label = '{}{:d}'.format("", id)
#         cls = clses[i]
#         conf = confs[i]
#         class_str = f'{names[int(cls)]}'
#         text = label+':'+class_str+':'+str(conf)
#         t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
#         cv2.rectangle(
#             img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
#         cv2.putText(img, text, (x1, y1 +
#                                  t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
#     return img

# def draw_boxes(img, bbox, identities=None,clses=None,confs=None,offset=(0, 0)):
#     for i, box in enumerate(bbox):
#         x1, y1, x2, y2 = [int(i) for i in box]
#         x1 += offset[0]
#         x2 += offset[0]
#         y1 += offset[1]
#         y2 += offset[1]
#         # box text and bar
#         id = int(identities[i]) if identities is not None else 0
#         color = compute_color_for_labels(id)
#         label = '{}{:d}'.format("", id)
#         label +=
#         t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
#         cv2.rectangle(
#             img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
#         cv2.putText(img, label, (x1, y1 +
#                                  t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
#     return img

class Detect:
    def __init__(self):
        self.tool = Tools()
    def detect(opt, save_img=False):
        out, source, weights, view_img, save_txt, imgsz = \
            opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source == '0' or source.startswith(
            'rtsp') or source.startswith('http') or source.endswith('.txt')

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(opt.config_deepsort)
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        # Initialize
        device = select_device(opt.device)
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        half = device.type != 'cpu'  # half precision only supported on CUDA

        #UPD初始化
        udpIpc = UDP_connect(opt.source)
        udpIpc.CleanMessage()

        # Load model
        model = torch.load(weights, map_location=device)[
            'model'].float()  # load to FP32
        model.to(device).eval()
        if half:
            model.half()  # to FP16

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            view_img = True
            save_img = True
            dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        # run once
        _ = model(img.half() if half else img) if device.type != 'cpu' else None

        save_path = str(Path(out))
        txt_path = str(Path(out)) + '/results.txt'

        for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):# dataset存储的内容为：路径，resize+pad的图片，原始图片，视频对象
            img = cv2.rectangle(img,(0,0),(1920,120),(255,255,255),-1)#过滤监控时间和名字,填成白色

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(
                pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Process detections
            udpIpc.SetUdpHead()
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                s += '%gx%g ' % img.shape[2:]  # print string
                save_path = str(Path(out) / Path(p).name)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    bbox_xywh = []#存储每个框的位置信息
                    confs = []

                    labels = []
                    # Adapt detections to deep sort input format#yolov5的检测结果输出
                    for *xyxy, conf, cls in det:
                        x_c, y_c, bbox_w, bbox_h = self.tool.bbox_rel(*xyxy)
                        obj = [x_c, y_c, bbox_w, bbox_h]

                        class_str = f'{names[int(cls)]}'
                        conf_str = f'{conf:.2f}'
                        confint=conf*100
                        #print(class_str + ':' + conf_str)
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])
                        labels.append([int(cls),int(confint)])#增加的conf和cls


                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)

                    # Pass detections to deepsort（deepsort主要功能实现）
                    outputs = deepsort.update(xywhs, confss, im0,labels)#outputs为检测框预测序列


                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:,-3]
                        out_clses=outputs[:,-2]
                        out_confs=outputs[:,-1]
                        #画框
                        self.tool.draw_boxes(im0, bbox_xyxy,out_clses ,out_confs,names,identities)
                        udpIpc.message_concat(bbox_xyxy,out_clses ,out_confs,names,identities)

                    # Write MOT compliant results to file
                    if save_txt and len(outputs) != 0:
                        for j, output in enumerate(outputs):
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2]
                            bbox_h = output[3]
                            identity = output[-1]
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

                else:
                    deepsort.increment_ages()

                # Print time (inference + NMS)
                #print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

            udpIpc.SetUdpTail()
            udpIpc.message_send()
            udpIpc.CleanMessage()



        print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':

    det = Detect()

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/ship.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        det.detect(args)
