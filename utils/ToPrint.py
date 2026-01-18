import all
import sys
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def ToPrint(b,path):
    sys.stdout = Logger(path+"print.txt")
    # Proposed输出
    print("我们的算法最终一共有", all.real_index_number, "个点")
    print("                                                                                                  ")
    print("RT剔点后平均残差", all.RT_residuals)
    print("A剃点后平均残差", all.A_residuals)
    print("H剃点后平均残差", all.H_residuals)
    print("H1剃点后平均残差", all.H1_residuals)
    print("H2剃点后平均残差", all.H2_residuals)
    print("H3剃点后平均残差", all.H3_residuals)
    print("我们的算法最终平均残差",all.final_residuals)
    print("                                                                                                  ")
    # RANSAC输出
    print("RANSAC最终得到", all.cnt, "个")
    print("RANSAC最终平均残差是", all.Ransac_residuals)
    print("                                                                                                  ")

    # 判断candidate图在那边
    # if b == 0:
    #     print("candidate图在右侧")
    # else:
    #     print("candidate图在左侧")
    # print("                                                                                                  ")




