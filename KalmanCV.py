import numpy as np
import cv2


class KalmanFilter:
    """
    KS:kalman模型
    """
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.003
        self.kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1

    def Update(self, x, y=0):
        """
        KS:滤波
        """
        current_mes = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(current_mes)
        current_pred = self.kf.predict()
        pred_x, pred_y = current_pred[0], current_pred[1]
        # print("kalmanPred: ", pred_x[0],pred_y[0])

        return int(pred_x[0]), int(pred_y[0])

    def Predict(self):
        """
        KS:预测
        """
        current_pred = self.kf.predict()
        pred_x, pred_y = current_pred[0], current_pred[1]
        # print("kalmanPred: ", pred_x[0],pred_y[0])

        return int(pred_x[0]), int(pred_y[0])