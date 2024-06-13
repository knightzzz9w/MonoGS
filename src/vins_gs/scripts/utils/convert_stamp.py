import time


class TicToc:
    def __init__(self):
        self.start_time = None
    def tic(self):
        self.start_time = time.time()

    def toc(self):
        temp_time = self.start_time
        self.start_time = time.time()
        return (time.time() - temp_time)*1e3



def stamp2seconds(msg):
    # 访问时间戳
    timestamp = msg.header.stamp
    
    # 转换为秒
    return timestamp.secs + timestamp.nsecs * 1e-9