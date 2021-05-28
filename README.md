# GetStarted
- 安装pytorch 1.8.1版本
- > pip install -r requirements.txt
- 下载YOLOv5权重文件 https://github.com/ultralytics/yolov5/releases. 放在 yolov5/weights/目录下
- 下载deep_sort权重文件 https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6. 放在deep_sort/deep/checkpoint/目录下；备份路径: https://pan.baidu.com/s/1b0WbHNk2ir2r6iEpeB665A  密码: makv
- 运行run_YoSort.py根据自己需要来修改启动参数
- > python3 run_YoSort.py --source rtsp://xxx(./image/xxx.jpg) --weights yolo_weight.pt --view-img(显示opencv检测画面)
# 功能更新列表
- 增加显示检测框的置信度和类别信息
- 增加udp发送框信息功能
- 更新id池功能，可复用之前抛弃的id号
- 更新发送MD5
- 屏蔽视频的时间和设备名
- 使用DetectYoSort类封装了原来的检测函数，方便项目嵌入
- 增加参数可与上位机联动关闭
- 更新检测框的卡尔曼滤波使得框跳动减缓(NEW!)
 
