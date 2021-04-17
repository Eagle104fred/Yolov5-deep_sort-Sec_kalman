
class IdPool:
    def __init__(self):
        self.id_dict = {}
    def id_init(self):#检测这个id是否存活
        for i in range(1,9999):
            if(i in self.id_dict.keys()):
                continue
            else:
                self.id_dict[i]=None
                return i
    def id_delect(self,id):
        self.id_dict.pop(id)