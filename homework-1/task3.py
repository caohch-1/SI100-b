#=============================================================================#
#                       Homework 5: SKY PRIORITY                              #
#       SI 100B: Introduction to Information Science and Technology           #
#                     Spring 2020, ShanghaiTech University                    #
#                     Author: Diao Zihao <hi@ericdiao.com>                    #
#                         Last motified: 03/11/2020                           #
#=============================================================================#
# task3.py - write your code for task 3 here.
import sys

"""读取文件"""
lines = sys.stdin.readlines()

"""获取所有哈皮信息"""
info = list()
for line in lines:
    temp = line.strip('\n').split()
    if temp != []:
        info.append(temp)

"""获取航班号码 总哈皮数 开始时间 需要运猪次数 普通哈皮数 纯种哈皮数"""
flight_num = info[0][0]
boarding_time = int(info[0][2])
common_num = int(info[1][0])
vip_num = int(info[common_num+2][0])
turn_num=(common_num+vip_num)//20+1

"""处理哈皮数据"""
# 去除第一行
info.pop(0)

# 去除两个队列人数
for ls in info:
    if (len(ls) == 1):
        info.remove(ls)

# 将时间转为int
for ls in info:
    ls[0]=int(ls[0])

"""分别创建普通哈皮列表和纯种哈皮列表"""
common = list()
for i in range(common_num):
    common.append(info[i])
vip = info[::-1][:vip_num][::-1]

"""去除来错地方的纯种大哈皮"""
sb_common = list()
for i in range(len(common)):
    if common[i][2] != flight_num:
        sb_common.append(common[i])
for sb in sb_common:
    common.remove(sb)

sb_vips = list()
for i in range(len(vip)):
    if vip[i][2] != flight_num:
        sb_vips.append(vip[i])
for sb in sb_vips:
    vip.remove(sb)

"""按照到达时间排序 ???可能没用"""
common.sort(key=(lambda x: x[0]))
vip.sort(key=(lambda x: x[0]))

"""我不想Debug了，求求你了"""
"""我认输！我重写 我第一题写的什么垃圾"""
time = boarding_time
bus = list()
while len(common) != 0 or len(vip) !=0:
    # Debug:轮次为20倍数 会多一轮空轮 直接break
    if len(vip) == 0 and len(common) == 0:
        break
    else:
    # 没时间解释了，快下车 ???这行是不是该放到最后
        bus = list()
        first_person_time=int()
        arrive_time=time
        # 没时间解释了，快上车
        for i in range(20):         
            # 先将等候中的哈皮赶上车
            # 再等别的哈皮来直到满载
            # Debug：防止越界判断 列表是否空
            if len(common) != 0 and len(vip) != 0:
                # 时刻准备着的乘客
                # VIP优先级高于普通人
                if vip[0][0] <= time:
                    bus.append(vip[0])
                    vip.pop(0)
                    if i==0:
                        first_person_time=max(bus[0][0],arrive_time)
                elif common[0][0] <= time:
                    bus.append(common[0])
                    common.pop(0)
                    if i==0:
                        first_person_time=max(bus[0][0],arrive_time)
                # 磨磨唧唧刚来的
                # VIP优先级高于普通人
                # Debug：注意更新时间啊喂！
                elif vip[0][0] <= common[0][0]:
                    if i!=0:
                        if vip[0][0]-first_person_time<=600:
                            bus.append(vip[0])
                            time = vip[0][0]
                            vip.pop(0)
                        else:
                            time=first_person_time+600
                            break
                    else:
                        bus.append(vip[0])
                        time = vip[0][0]
                        vip.pop(0)
                        first_person_time=max(bus[0][0],arrive_time)
                else:
                    if i!=0:
                        if common[0][0]-first_person_time<=600:
                            bus.append(common[0])
                            time=common[0][0]
                            common.pop(0)
                        else:
                            time=first_person_time+600
                            break
                    else:
                        bus.append(common[0])
                        time = common[0][0]
                        common.pop(0)
                        first_person_time=max(bus[0][0],arrive_time)
            elif len(vip) == 0 and len(common) != 0:
                if i!=0:
                    if common[0][0]-first_person_time<=600:
                        bus.append(common[0])
                        time=max(common[0][0],time)
                        common.pop(0)
                    else:
                        time=first_person_time+600
                        break
                else:
                    bus.append(common[0])
                    time = max(common[0][0], time)
                    common.pop(0)
                    first_person_time=max(bus[0][0],arrive_time)
            elif len(vip) != 0 and len(common) == 0:
                if i != 0:
                    if vip[0][0]-first_person_time<=600:
                        bus.append(vip[0])
                        time=max(vip[0][0],time)
                    else:
                        time=first_person_time+600
                        break
                else:
                    bus.append(vip[0])
                    time = max(vip[0][0], time)
                    vip.pop(0)
                    first_person_time=max(bus[0][0],arrive_time)
            else:
                pass
            # 乖乖排好
            #bus.sort(key=(lambda x: x[0]))
        # 输出信息
        sys.stdout.write(str(time))
        sys.stdout.write(':')
        for passenger in bus:
            sys.stdout.write(' '+passenger[1])
        sys.stdout.write('\n')
        time += 600
