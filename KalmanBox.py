from KalmanCV import KalmanFilter
from tools import MeanSpeed,IOU
import time


class KalmanBox:
    """
    KS:kalman滤波器组
    """

    def __init__(self, maxAge):
        self.predList = []
        self.maxAge = maxAge
        self.meanSpeed = MeanSpeed(time.time())
        self.iou = IOU()
    def Filter(self, ids, boxes):
        """
        KS:kalman滤波
        """
        resultBoxes = []
        resultId = []

        #先预测再校准
        for kfBoxPred in self.predList:
            if (kfBoxPred.IsShow == True):  # 未检测到直接上预测
                pred_xyxy = kfBoxPred.Predict()
                # x1, y1, x2, y2 = [i for i in pred_xyxy]

                resultBoxes.append(pred_xyxy)
                resultId.append(kfBoxPred.id)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(i) for i in box]
            id = int(ids[i]) if ids is not None else 0
            flag = True  # KS: 判断是否需要新建卡尔曼box
            filter_xyxy = []
            # KS: 获取当前帧时间
            currentTime = time.time()


            for kfBoxPred in self.predList:  # KS: 遍历卡尔曼list如果有旧的就更新
                if (id == kfBoxPred.id):
                    # KS: 计算速度,用于卡尔曼速度预测
                    meanX, meanY = self.meanSpeed.Count(currentTime, kfBoxPred, x1, y1, x2, y2)
                    # KS: 卡尔曼校准
                    kfBoxPred.Correct(x1, y1, x2, y2, meanX, meanY)
                    # pred_x1, pred_y1, pred_x2, pred_y2 = [i for i in filter_xyxy]
                    if (kfBoxPred.IsShow == True):  # KS: 只有经过一定的时间的卡尔曼才可以输出
                        #resultBoxes.append([pred_x1, pred_y1, pred_x2, pred_y2])
                        #resultId.append(id)
                        #comfirmedBox.append([pred_x1, pred_y1, pred_x2, pred_y2])
                        flag = False
                        break  # 找到后跳出查找循环, 进入下一个id

            """
            KS:新建卡尔曼检测器
            """
            if (flag):  # KS: 新建卡尔曼box
                flagIsNew=True

                for kfBox in self.predList:

                    iou=self.iou.Iou(kfBox.xyxy,[x1,y1,x2,y2])
                    if(iou!=0):
                        print(iou)
                        print(kfBox.xyxy, [x1, y1, x2, y2])
                    if(iou>0.4):
                        flagIsNew=False
                if(flagIsNew==True):
                    self.predList.append(KfBox(id, x1, y1, x2, y2, self.maxAge))





        self.meanSpeed.oldTime = currentTime  # KS: 保存上一帧的时间
        return resultId, resultBoxes

    def Predict(self):
        """
        KS:kalman预测
        """
        resultId = []
        resultBoxes = []
        for box in self.predList:
            pred_xyxy = box.Predict()
            #x1, y1, x2, y2 = [i for i in pred_xyxy]
            box.xyxy=pred_xyxy
            if (box.IsShow == True):  # KS: 只有经过一定的时间的卡尔曼才可以输出
                resultBoxes.append(pred_xyxy)
                resultId.append(box.id)
            self._UpdateAge(box)
        return resultId, resultBoxes

    def _UpdateAge(self,box):
        """
        KS:更新所有检测器的血条
        """
        if (box.age > box.maxAge and len(self.predList)>0):
            self.predList.remove(box)  # KS: 血条耗尽删除box
        box.age += 1  # KS: 更新血条


class KfBox:
    """
    KS:单个卡尔曼追踪器定义
    """

    def __init__(self, id, x1, y1, x2, y2, age):
        self.maxAge = age
        self.age = 0
        self.id = id
        self.conf=20

        self.xyxy=[x1,y1,x2,y2]

        # KS: 注册卡尔曼模型
        self.kf = KalmanFilter()
        # self.kf_p1 = KalmanFilter()
        # self.kf_p2 = KalmanFilter()

        self.oldMid = [0, 0]  # KS: 上一帧的框中点位置

        self.IsShow = False
        self.DetectTimes = 0
        self.DetectMaxTimes = 8#设置每个框过滤多少帧检测才显示

        self.TempCount=0

    def Correct(self, x1, y1, x2, y2, meanX, meanY):

        count=self.TempCount
        count+=1
        self.TempCount=count

        self.kf.Correctx2(x1, y1, x2, y2, meanX, meanY)
        #pred_x1, pred_y1, pred_x2, pred_y2 = self.kf.Predictx2()
        # self.predPoint_x1, self.predPoint_y1=self.kf_p1.Update(x1,y1)
        # self.predPoint_x2, self.predPoint_y2 = self.kf_p2.Update(x2, y2)

        self.age = 0  # KS: 检测到就回满血

        if (self.DetectTimes > self.DetectMaxTimes):  # KS: 屏蔽不稳定检测框
            """
            KS:屏蔽不稳定帧 
            """
            self.IsShow = True
        else:
            self.DetectTimes += 1

        #return [pred_x1, pred_y1, pred_x2, pred_y2]

    def Predict(self):
        x1, y1, x2, y2 = self.kf.Predictx2()
        return [x1,y1,x2,y2]

        # self.predPoint_x1, self.predPoint_y1=self.kf_p1.Predict()
        # self.predPoint_x2, self.predPoint_y2=self.kf_p2.Predict()
