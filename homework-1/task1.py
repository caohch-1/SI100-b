#=============================================================================#
#                       Homework 5: SKY PRIORITY                              #
#       SI 100B: Introduction to Information Science and Technology           #
#                     Spring 2020, ShanghaiTech University                    #
#                     Author: Diao Zihao <hi@ericdiao.com>                    #
#                         Last motified: 03/11/2020                           #
#=============================================================================#
# task1.py - write your code for task 1 here.
import sys
# 读取文件
lines = sys.stdin.readlines()

# 处理为二维列表
info = list()
for line in lines:
    temp = line.strip('\n').split()
    if temp != []:
        info.append(temp)

# 获取航班号码 总人数 开始时间 需要运载次数
flight_num = info[0][0]
passenger_num = int(info[0][1])
boarding_time = int(info[0][2])
turn_num = passenger_num//20+1

# 清理脏数据
# 去除第一行
info.pop(0)
# 去除两个队列人数
for ls in info:
    if (len(ls) == 1):
        info.remove(ls)
# 去除来错地方的纯种大哈皮
wrong_passenger = []
for i in range(len(info)):
    if info[i][2] != flight_num:
        wrong_passenger.append(info[i])
for sb in wrong_passenger:
    info.remove(sb)
# 将时间转为int
for ls in info:
    ls[0] = int(ls[0])
passenger_num = len(info)
turn_num = passenger_num//20+1

# 按照到达时间排序
#info.sort(key=(lambda x: x[0]))

"""重写"""
time = boarding_time
bus = list()
for turn in range(turn_num):
    # Debug:轮次为20倍数 会多一轮空轮 直接break
    if len(info) == 0:
        break
    # 没时间解释了，快下车
    bus = list()
    # 没时间解释了，快上车
    for i in range(20):
        if len(info) != 0:
            bus.append(info[0])
            time = max(time, info[0][0])
            info.pop(0)
        else:
            break
    # 输出信息
    sys.stdout.write(str(time))
    sys.stdout.write(': ')
    for passenger in bus[:-1]:
        sys.stdout.write(passenger[1]+' ')
    sys.stdout.write(bus[-1][1]+'\n')
    time += 600

# # 每20人分为一组
# counter = 1
# time = boarding_time
# for turn in range(0, turn_num-1):
#         # 判断第一次出发时间是否比班车到达时间早
#     if turn == 0:
#         if info[counter*20-1][0] >= boarding_time:
#             sys.stdout.write(str(info[counter*20-1][0]))
#             sys.stdout.write(':')
#             time = info[counter*20-1][0]
#         elif info[counter*20-1][0] < boarding_time:
#             sys.stdout.write(str(boarding_time))
#             sys.stdout.write(':')
#     else:
#         # 判断两次发车间隔是否大于10分钟
#         if info[counter*20-1][0] < time:
#             sys.stdout.write(str(time))
#             sys.stdout.write(':')
#         else:
#             sys.stdout.write(str(info[counter*20-1][0]))
#             sys.stdout.write(':')
#             time = info[counter*20-1][0]
#     # 写入乘客名字
#     for i in range((counter-1)*20, counter*20):
#         sys.stdout.write(' ')
#         sys.stdout.write(info[i][1])
#     sys.stdout.write('\n')
#     # 更新时间
#     time += 600
#     # 更新班次
#     counter += 1

# # 剩余的不满20的人
# if passenger_num % 20 != 0:
#     # 判断两次发车间隔是否大于10分钟
#     if info[-1][0] < time:
#         sys.stdout.write(str(time))
#         sys.stdout.write(':')
#     else:
#         sys.stdout.write(str(info[-1][0]))
#         sys.stdout.write(':')
#     for ls in info[::-1][:passenger_num % 20][::-1]:
#         sys.stdout.write(' ')
#         sys.stdout.write(ls[1])
#     sys.stdout.write('\n')
