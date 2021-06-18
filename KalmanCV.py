import numpy as np
import cv2
import time


class KalmanFilter:
    """
    KS:kalman模型
    """

    def __init__(self):
        self.oldTime=0
        self.deltaTime=0
        self.kf = cv2.KalmanFilter(6, 6)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]], np.float32)*8
           # [[1, 0, 0, 0, 0.5, 0], [0, 1, 0, 0, 0, 0.5], [0, 0, 1, 0,0.5, 0], [0, 0, 0, 1, 0,0.5], [0.5, 0,0.5, 0, 3, 0],[0, 0.5, 0, 0.5, 0, 3]], np.float32) *8
        self.kf.measurementNoiseCov = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]], np.float32) * 30
           # [[1, 0, 0, 0, 0.5, 0], [0, 1, 0, 0, 0, 0.5], [0, 0, 1, 0, 0.5, 0], [0, 0, 0, 1, 0, 0.5],[0.5, 0, 0.5, 0, 3, 0],[0, 0.5, 0, 0.5, 0, 3]], np.float32) * 30

        # self.kf = cv2.KalmanFilter(4, 2)
        # self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0,1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.5
        # self.kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.3

    # def Update(self, x, y=0):
    #     """
    #     KS:滤波
    #     """
    #     current_mes = np.array([[np.float32(x)], [np.float32(y)]])
    #     self.kf.correct(current_mes)
    #     current_pred = self.kf.predict()
    #     pred_x, pred_y = current_pred[0], current_pred[1]
    #     # print("kalmanPred: ", pred_x[0],pred_y[0])
    #
    #     return int(pred_x[0]), int(pred_y[0])
    def Correctx2(self, x1, y1, x2, y2, meanX, meanY):
        """
        KS:滤波
        """
        self.GetDeltaTime()
        self.kf.transitionMatrix[0][4]=self.deltaTime
        self.kf.transitionMatrix[1][5]=self.deltaTime
        self.kf.transitionMatrix[2][4]=self.deltaTime
        self.kf.transitionMatrix[3][5]=self.deltaTime
            #

        current_mes = np.array(
            [[np.float32(x1)], [np.float32(y1)], [np.float32(x2)], [np.float32(y2)], [np.float32(meanX)],
             [np.float32(meanY)]])
        self.kf.correct(current_mes)


    def Predictx2(self):
        # """
        # KS:预测
        # """
        self.GetDeltaTime()
        self.kf.transitionMatrix[0][4] = self.deltaTime
        self.kf.transitionMatrix[1][5] = self.deltaTime
        self.kf.transitionMatrix[2][4] = self.deltaTime
        self.kf.transitionMatrix[3][5] = self.deltaTime

        current_pred = self.kf.predict()
        pred_x1, pred_y1, pred_x2, pred_y2 = current_pred[0], current_pred[1], current_pred[2], current_pred[3]

        # print("kalmanPred: ", pred_x[0],pred_y[0])
        return int(pred_x1[0]), int(pred_y1[0]), int(pred_x2[0]), int(pred_y2[0])

    # def Predict(self):
    #     current_pred = self.kf.predict()
    #     pred_x, pred_y = current_pred[0], current_pred[1]
    #     # print("kalmanPred: ", pred_x[0],pred_y[0])
    #     return int(pred_x[0]), int(pred_y[0])

    def GetDeltaTime(self):
        tempTime=time.time()
        self.deltaTime=tempTime-self.oldTime
        self.oldTime=tempTime
