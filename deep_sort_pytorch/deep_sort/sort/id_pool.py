class IdPool:
    def __init__(self):
        self.id_dict = {}#字典用于存储每个id
        self.hp=100#每个id血条为100帧, deepsort的id抛弃血条为70帧
    def init(self):#检测这个id是否存活
        for i in range(1,9999):
            if(i in self.id_dict.keys()):
                continue
            else:
                self.id_dict[i]=self.hp
                return i
        return 0
    def updat_times(self):#刷新每一帧所有id的血条
        for k in list(self.id_dict.keys()):
            self.id_dict[k]-=1
            if(self.id_dict[k]<=0):
                print("release id :",k)
                del self.id_dict[k]


    def update(self,id):
        self.id_dict[id] = self.hp