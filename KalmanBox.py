from KalmanCV import KalmanFilter
from tools import MeanSpeed
import time


class KalmanBox:
    """
    KS:kalman滤波器组
    """

    def __init__(self, maxAge):
        self.predList = []
        self.maxAge = maxAge
        self.meanSpeed = MeanSpeed(time.time())

    def Filter(self, ids, boxes):
        """
        KS:kalman滤波
        """
        resultBoxes = []
        resultId = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(i) for i in box]
            id = int(ids[i]) if ids is not None else 0
            flag = True  # KS: 判断是否需要新建卡尔曼box
            pred_xyxy = []
            # KS: 获取当前帧时间
            currentTime = time.time()
            for boxPred in self.predList:  # KS: 遍历卡尔曼list如果有旧的就更新
                if (id == boxPred.id):
                    # KS: 计算速度
                    meanX, meanY = self.meanSpeed.Count(currentTime, boxPred, x1, y1, x2, y2)
                    #KS: 滤波
                    pred_xyxy = boxPred.Update(x1, y1, x2, y2, meanX, meanY)
                    pred_x1, pred_y1, pred_x2, pred_y2 = [i for i in pred_xyxy]

                    if (boxPred.IsShow == True):  # KS: 只有经过一定的时间的卡尔曼才可以输出
                        resultBoxes.append([pred_x1, pred_y1, pred_x2, pred_y2])
                        resultId.append(id)
                    flag = False

                    break

            if (flag):  # KS: 新建卡尔曼box
                """
                KS:!!!!!!!!!!!!!bug!!!!!!!!!!!!!!!!!!!
                    这个地方新建了之后应该马上投入计算, 晚点修复 
                """
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
            box.Predict()
            x1 = box.predPoint_x1
            y1 = box.predPoint_y1
            x2 = box.predPoint_x2
            y2 = box.predPoint_y2

            if (box.IsShow == True):  # KS: 只有经过一定的时间的卡尔曼才可以输出
                resultBoxes.append([x1, y1, x2, y2])
                resultId.append(box.id)
        return resultId, resultBoxes

    def UpdateAllAge(self):
        """
        KS:更新所有检测器的血条
        """
        for box in self.predList:
            if (box.age <= 0):
                self.predList.remove(box)  # KS: 血条耗尽删除box
            box.age -= 1  # KS: 更新血条


class KfBox:
    """
    KS:单个卡尔曼追踪器定义
    """

    def __init__(self, id, x1, y1, x2, y2, age):
        self.maxAge = age
        self.age = age
        self.id = id

        # KS: 注册卡尔曼模型
        self.kf = KalmanFilter()
        # self.kf_p1 = KalmanFilter()
        # self.kf_p2 = KalmanFilter()

        self.oldMid = [0, 0]  # KS: 上一帧的框中点位置

        self.IsShow = False
        self.DetectTimes = 0
        self.DetectMaxTimes = 8#设置每个框过滤多少帧检测才显示

        self.TempCount=0

    def Update(self, x1, y1, x2, y2, meanX, meanY):

        count=self.TempCount
        count+=1
        self.TempCount=count

        self.kf.Correctx2(x1, y1, x2, y2, meanX, meanY)
        pred_x1, pred_y1, pred_x2, pred_y2 = self.kf.Predictx2()
        # self.predPoint_x1, self.predPoint_y1=self.kf_p1.Update(x1,y1)
        # self.predPoint_x2, self.predPoint_y2 = self.kf_p2.Update(x2, y2)

        self.age = self.maxAge  # KS: 检测到就回满血

        if (self.DetectTimes > self.DetectMaxTimes):  # KS: 屏蔽不稳定检测框
            """
            KS:屏蔽不稳定帧 
            """
            self.IsShow = True
        else:
            self.DetectTimes += 1

        return [pred_x1, pred_y1, pred_x2, pred_y2]

    def Predict(self):
        self.predPoint_x1, self.predPoint_y1, self.predPoint_x2, self.predPoint_y2 = self.kf.Predictx2()
        # self.predPoint_x1, self.predPoint_y1=self.kf_p1.Predict()
        # self.predPoint_x2, self.predPoint_y2=self.kf_p2.Predict()
