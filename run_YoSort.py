import sys

sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
import argparse
import torch
from detect_YoSort import DetectYoSort  # 主功能类

if __name__ == '__main__':
    det = DetectYoSort()  # 实例化

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/2021.5.13_best.pt', help='model.pt path')  # yolov5检测模型
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,  # 修改默认的检测源
                        default='../TrafficFlow/testVideo.mp4', help='source')
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
    parser.add_argument("--udp_ip", type=str,
                        default="127.0.0.1")  # upd发送框的IP地址
    parser.add_argument("--udp_port", type=int,
                        default=9999)  # upd发送框的IP地址
    parser.add_argument("--check_pid", type=int,
                        default='0')  # upd发送框的IP地址
    parser.add_argument('--kalman_predict', action='store_true',
                        help='kalman predict ship only')
    parser.add_argument('--kalmanPred_spacing', type=int,default=8,
                        help='every some frame using kalman_predict function')

    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        det.run(args)  # 调用主功能
