import numpy as np
import cv2

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.003
        self.kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1

    def UpdateList(self,xList,yList):
        predList = []
        for i in range(0,len(xList)-1):
            if(i==0):

                #print("kalmanOrigin: ", xList[i], yList[i])

                current_mes=np.array([[np.float32(xList[i])], [np.float32(yList[i])]])
                self.kf.correct(current_mes)
                current_pred=self.kf.predict()
                pred_x,pred_y = current_pred[0],current_pred[1]

                predList.append((pred_x[0],pred_y[0]))
                #print("kalmanPred: ", pred_x[0],pred_y[0])
                #print("kalmanPred: ", predList)
        return predList

    def Update(self,x,y=0):
        current_mes=np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(current_mes)
        current_pred=self.kf.predict()
        pred_x,pred_y = current_pred[0],current_pred[1]
        #print("kalmanPred: ", pred_x[0],pred_y[0])

        return int(pred_x[0]),int(pred_y[0])