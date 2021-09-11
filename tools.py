import cv2
import psutil
import os
import cmath
import numpy as np


class Tools:
    def __init__(self):
        self.palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

        # 预测框解码，返回框中点和宽高

    def bbox_rel(self, *xyxy):
        """" Calculates the relative bounding box from absolute pixel values. """
        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h

    def compute_color_for_labels(self, label):
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in self.palette]
        return tuple(color)

    def draw_boxes(self, img, bbox, clses, confs, names, identities=None, offset=(0, 0)):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = self.compute_color_for_labels(id)
            label = '{}{:d}'.format("", id)
            cls = clses[i]
            conf = confs[i]
            class_str = f'{names[int(cls)]}'
            text = label + ':' + class_str + ':' + str(conf)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, text, (x1, y1 +
                                    t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img

    def draw_boxes_kalman(self, img, bbox, identities=None):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]

            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = self.compute_color_for_labels(id)
            label = '{}{:d}'.format("", id)

            text = label
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, text, (x1, y1 +
                                    t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img

    # KS: 联动关闭
    def CheckPID(self, pid):
        if (pid != 0):
            if not psutil.pid_exists(pid):
                os.kill(os.getpid(), 0);


"""
KS:控制kalman和yolo刷新的次数 
"""


class Counter:
    def __init__(self, maxAge):
        self.kalmanAge = maxAge
        self.yoloAge = 1
        self.maxAge = maxAge
        self.status = "yolo"
        self.maxTimers=20
        self.minTimers=1
    def Update(self):
        if (self.status == "yolo"):
            self.yoloAge -= 1
            if (self.yoloAge > 0):
                return self.status
            else:
                self.yoloAge = 1  # KS: reset
                if (self.status == "yolo"):  # KS: 切换状态
                    self.status = "kalman"
                else:
                    self.status = "yolo"
                return self.status

        elif (self.status == "kalman"):
            self.kalmanAge -= 1
            if (self.kalmanAge > 0):
                return self.status
            else:
                self.kalmanAge = self.maxAge  # KS: reset
                if (self.status == "yolo"):  # KS: 切换状态
                    self.status = "kalman"
                else:
                    self.status = "yolo"
                return self.status

    """
    KS:动态调整检测频率 
    """
    def AdaptedTimes(self, boxesNumbers):
        tempTimes = cmath.sqrt(self.maxTimers/boxesNumbers)
        if(tempTimes>self.maxTimers):
            tempTimes=self.maxTimers
        elif(tempTimes<self.minTimers):
            tempTimes=self.minTimers
        return tempTimes



class MeanSpeed:
    def __init__(self, time):
        self.oldTime = time

    def Count(self, currentTime, box, x1, y1, x2, y2):
        midX = (x2 + x1) / 2
        midY = (y2 + y1) / 2
        mid = [midX, midY]

        diffTime = currentTime - self.oldTime

        diff_x=mid[0] - box.oldMid[0]
        diff_y=mid[1] - box.oldMid[1]
        meanX = diff_x / diffTime
        meanY = diff_y / diffTime
        #KS: 更新数据
        box.oldMid = mid

        print("{0}meanSpeed:X:{1},Y:{2}".format(box.id, meanX, meanY))
        return meanX,meanY
class IOU:
    def Iou(self,box1, box2, wh=False):
        if wh == False:
            xmin1, ymin1, xmax1, ymax1 = box1
            xmin2, ymin2, xmax2, ymax2 = box2
        else:
            xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
            xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
            xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
            xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)
            # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
        xx1 = np.max([xmin1, xmin2])
        yy1 = np.max([ymin1, ymin2])
        xx2 = np.min([xmax1, xmax2])
        yy2 = np.min([ymax1, ymax2])
        # 计算两个矩形框面积
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))  # 计算交集面积
        iou = inter_area / (area1 + area2 - inter_area + 1e-6)  # 计算交并比

        return iou