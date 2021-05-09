from socket import *
import hashlib

class UDP_connect:
    def __init__(self,rtspUrl,udpIp,udpPort):
        self.udp_socket = socket(AF_INET, SOCK_DGRAM)
        self.server_addr = (udpIp, udpPort)
        # dest_addr = ('192.168.1.114', 8080)
        print("ip:port",udpIp,udpPort)
        camBG_port = "7502"
        camCS_port = "7702"
        self.cam_port = camBG_port
        cam_user = "admin"
        cam_psw = "SMUwm_007"

        self.message_head = 'ShipTrack'+':'
        self.message_tail = 'END'+'\n'
        self.send_message = ""

        self.md5_message = ""
        self.CountRtspMD5(rtspUrl)

    def CountRtspMD5(self,rtspUrl):
        md5_rtspurl = hashlib.md5(rtspUrl.encode(encoding='UTF-8')).hexdigest()
        self.md5_message = md5_rtspurl[:8]+ '|'
    #设置数据头
    def SetUdpHead(self):
        self.send_message +=self.message_head
        self.send_message += self.md5_message
    #清除数据
    def CleanMessage(self):
        self.send_message=""
    #设置数据尾
    def SetUdpTail(self):
        self.send_message += self.message_tail
    #发送数据主体
    def message_concat(self,bbox, clses, confs, names,identities=None, offset=(0, 0)):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0

            label = '{}{:d}'.format("", id)
            cls = clses[i]
            class_str = f'{names[int(cls)]}'
            conf = confs[i]
            text = str(id)+':'+class_str + ':' + str(conf)+':'+\
                   str(x1)+':'+str(y1)+':'+str(x2)+':'+str(y2)+'|'
            self.send_message+=text
    #发送函数
    def message_send(self):
        self.udp_socket.sendto(self.send_message.encode('utf-8'), self.server_addr)


