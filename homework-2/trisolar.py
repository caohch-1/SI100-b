from math import sqrt, pi, sin, cos,tan

from decimal import Decimal as dc

ERROR = 0.01


def AlmostEqual(first, second):
    delta = abs(ERROR * first)
    diff = abs(first - second)
    assert diff <= delta


def task1(planets_num: int, check_time: int, bodys: list):
    res=list()
    for time in range(check_time):
        delta_speed_x_list=list()
        delta_speed_y_list=list()
        for i in range(planets_num):
            delta_speed_x=0
            delta_speed_y=0
            m_body=bodys[i]
            for j in range(planets_num):
                if j==i:
                    continue
                else:
                    body=bodys[j]
                    dis=sqrt( ( body[1] - m_body[1])**2 + ( body[2] - m_body[2] )**2 )
                    delta_speed_x+=body[0]*(body[1]-m_body[1])/(dis**3)
                    delta_speed_y+=body[0]*(body[2]-m_body[2])/(dis**3)                   
            delta_speed_x_list.append(delta_speed_x)
            delta_speed_y_list.append(delta_speed_y)
        for i in range(planets_num):
            bodys[i][3]+=delta_speed_x_list[i]
            bodys[i][4]+=delta_speed_y_list[i]
            bodys[i][1]+=bodys[i][3]
            bodys[i][2]+=bodys[i][4]

    for body in bodys:
        answer=list()
        answer.append(body[1])
        answer.append(body[2])
        res.append(answer)
    return res


def task2(check_time: int, bodys: list):
    sun = 0
    distance_list=list()
    distance_list=task1(4,check_time,bodys)
    for i in range(3):
        distance=sqrt((distance_list[i][0]-distance_list[3][0])**2 + (distance_list[i][1]-distance_list[3][1])**2)
        if distance<=200:
            sun+=1
    if sun == 0:
        return "Eternal Night"
    elif sun == 1:
        return "Stable Era"
    elif sun == 2:
        return "Double-Solar Day"
    elif sun == 3:
        return "Tri-Solar Day"


def task3(check_time: int, bodys: list):
    S = 0
    for time in range(check_time):
        sun=0   
        res=list()
        delta_speed_x_list=list()
        delta_speed_y_list=list()
        for i in range(4):
            delta_speed_x=0
            delta_speed_y=0
            m_body=bodys[i]
            for j in range(4):
                if j==i:
                    continue
                else:
                    body=bodys[j]
                    dis=sqrt( ( body[1] - m_body[1])**2 + ( body[2] - m_body[2] )**2 )
                    delta_speed_x+=body[0]*(body[1]-m_body[1])/(dis**3)
                    delta_speed_y+=body[0]*(body[2]-m_body[2])/(dis**3)                   
            delta_speed_x_list.append(delta_speed_x)
            delta_speed_y_list.append(delta_speed_y)
        for i in range(4):
            bodys[i][3]+=delta_speed_x_list[i]
            bodys[i][4]+=delta_speed_y_list[i]
            bodys[i][1]+=bodys[i][3]
            bodys[i][2]+=bodys[i][4]
            
        for body in bodys:
            answer=list()
            answer.append(body[1])
            answer.append(body[2])
            res.append(answer)        
             
        for i in range(3):
            distance=sqrt((res[i][0]-res[3][0])**2 + (res[i][1]-res[3][1])**2)
            if distance<=200:
                sun+=1
        
        if sun==1:
            S+=2
        else:
            S-=1
            
        if S<0:
            return "No civilization"
        
    if S < 0:
        return "No civilization"
    elif S < 400:
        return "level 1 civilization"
    elif S < 1200:
        return "level 2 civilization"
    else:
        return "level 3 civilization"

def judge_same_side_one(k,b,x,y):
    res=k*x+b
    if y>=res:
        return True
    else:
        return False
    
def judge_same_side_two(k,b,x,y):
    res=k*x+b
    if y<=res:
        return True
    else:
        return False

def task_bonus(check_time: int, bodys: list):
    day = 0
    time=0
    while time <check_time:
        sun=0   
        res=list()
        delta_speed_x_list=list()
        delta_speed_y_list=list()
        for i in range(4):
            delta_speed_x=0
            delta_speed_y=0
            m_body=bodys[i]
            for j in range(4):
                if j==i:
                    continue
                else:
                    body=bodys[j]
                    dis=sqrt( ( body[1] - m_body[1])**2 + ( body[2] - m_body[2] )**2 )
                    delta_speed_x+=body[0]*(body[1]-m_body[1])/(dis**3)
                    delta_speed_y+=body[0]*(body[2]-m_body[2])/(dis**3)                   
            delta_speed_x_list.append(delta_speed_x)
            delta_speed_y_list.append(delta_speed_y)
        for i in range(4):
            bodys[i][3]+=delta_speed_x_list[i]
            bodys[i][4]+=delta_speed_y_list[i]
            bodys[i][1]+=bodys[i][3]
            bodys[i][2]+=bodys[i][4]
            
        for body in bodys:
            answer=list()
            answer.append(body[1])
            answer.append(body[2])
            res.append(answer)  
            
        for i in range(3):
            distance=sqrt((res[i][0]-res[3][0])**2 + (res[i][1]-res[3][1])**2)
            if distance<=200:
                sun+=1
                
        if sun ==1:
            if time%360==90 :
                if res[0][0]>=res[3][0] and res[1][0]>=res[3][0] and res[2][0]>=res[3][0]:
                    day+=1
            elif time%360==270 :
                if res[0][0]<=res[3][0] and res[1][0]<=res[3][0] and res[2][0]<=res[3][0]:
                    day+=1
            elif time%360==180 and res[0][1]<=res[3][1] and res[1][1]<=res[3][1] and res[2][1]<=res[3][1]:
                day+=1
            elif time%360==0 and res[0][1]>=res[3][1] and res[1][1]>=res[3][1] and res[2][1]>=res[3][1]:
                day+=1
            else:
                if 0<time%360<90 or 270<time%360<360:
                    k_of_line=-tan(time*pi/180)
                    b_of_line=res[3][1]-k_of_line*res[3][0]
                    if judge_same_side_one(k_of_line,b_of_line,res[0][0],res[0][1]) and judge_same_side_one(k_of_line,b_of_line,res[1][0],res[1][1]) and judge_same_side_one(k_of_line,b_of_line,res[2][0],res[2][1]):
                        day+=1
                elif 90<time%360<270:
                    k_of_line=-tan(time*pi/180)
                    b_of_line=res[3][1]-k_of_line*res[3][0]
                    if judge_same_side_two(k_of_line,b_of_line,res[0][0],res[0][1]) and judge_same_side_two(k_of_line,b_of_line,res[1][0],res[1][1]) and judge_same_side_two(k_of_line,b_of_line,res[2][0],res[2][1]):
                        day+=1  
        time+=1   
    return day


if __name__ == "__main__":
    # '''
    # Task 1 Example 1
    # <planets-num> = 2, <check-time>  = 1986
    # <planet1-mass> = 10000, <planet1-coordinate-x> = 0, <planet1-coordinate-y> = 0, <planet1-speed-x> = 0, <planet1-speed-y> = 0
    # <planet2-mass> = 0.1, <planet2-coordinate-x> = 1000, <planet2-coordinate-y> = 0, <planet2-speed-x> = 0, <planet2-speed-y> = sqrt(10)
    # '''
    # output = task1(
    #     2,
    #     1986,
    #     [
    #         [10000, 0, 0, 0, 0],
    #         [0.1, 1000, 0, 0, sqrt(10)]
    #     ]
    # )
    # answer = [(-4.568800204932483e-09, 0.06283041322543657), (1000.0004568800274, -2.757889449272592)]
    # for i in range(len(answer)):
    #     for j in (0, 1):
    #         ans = answer[i][j]
    #         out = output[i][j]
    #         AlmostEqual(ans, out)

    # '''
    # Task 2 Example 1
    # <check-time>  = 1986
    # <sun1-mass> = 1000, <sun1-coordinate-x> = 0, <sun1-coordinate-y> = 0, <sun1-speed-x> = 0, <sun1-speed-y> = 0
    # <sun2-mass> = 1, <sun2-coordinate-x> = 1000000, <sun2-coordinate-y> = 0, <sun2-speed-x> = 0, <sun2-speed-y> = 0
    # <sun3-mass> = 1, <sun3-coordinate-x> = -1000000, <sun3-coordinate-y> = 0, <sun3-speed-x> = 0, <sun3-speed-y> = 0
    # <planet-mass> = 0.1, <planet-coordinate-x> = 100, <planet-coordinate-y> = 0, <planet-speed-x> = 0, <planet-speed-y> = sqrt(10)
    # '''
    # output = task2(
    #     1986,
    #     [
    #         [1000, 0, 0, 0, 0],
    #         [1, 1000000, 0, 0, 0],
    #         [1, -1000000, 0, 0, 0],
    #         [0.1, 100, 0, 0, sqrt(10)]
    #     ]
    # )
    # assert output == "Stable Era"

    # '''
    # Task 3 Example 1
    # <check-time>  = 600
    # <sun1-mass> = 1000, <sun1-coordinate-x> = 0, <sun1-coordinate-y> = 0, <sun1-speed-x> = 0, <sun1-speed-y> = 0
    # <sun2-mass> = 0.001, <sun2-coordinate-x> = 148.6, <sun2-coordinate-y> = 0, <sun2-speed-x> = 0, <sun2-speed-y> = -2.59
    # <sun3-mass> = 0.001, <sun3-coordinate-x> = 0, <sun3-coordinate-y> = 148.6, <sun3-speed-x> = 2.59, <sun3-speed-y> = 0
    # <planet-mass> = 0.001, <planet-coordinate-x> = 0, <planet-coordinate-y> = -148.6, <planet-speed-x> = -2.59, <planet-speed-y> = sqrt(10)
    # '''
    # omega = 2*pi/360
    # R = (1000/omega**2)**(1/3)
    # output = task3(
    #     600,
    #     [
    #         [1000, 0, 0, 0, 0],
    #         [0.001, R, 0, 0, -omega*R],
    #         [0.001, 0, R, omega*R, 0],
    #         [0.001, 0, -R, -omega*R, 0]
    #     ]
    # )
    # assert output == "level 3 civilization"

    '''
    Task BONUS Example 1
    <check-time>  = 6000
    <sun1-mass> = 1000, <sun1-coordinate-x> = 0, <sun1-coordinate-y> = 0, <sun1-speed-x> = 0, <sun1-speed-y> = 0
    <sun2-mass> = 0.001, <sun2-coordinate-x> = 148.6, <sun2-coordinate-y> = 0, <sun2-speed-x> = 0, <sun2-speed-y> = -2.59
    <sun3-mass> = 0.001, <sun3-coordinate-x> = 0, <sun3-coordinate-y> = 148.6, <sun3-speed-x> = 2.59, <sun3-speed-y> = 0
    <planet-mass> = 0.001, <planet-coordinate-x> = 0, <planet-coordinate-y> = -148.6, <planet-speed-x> = -2.59, <planet-speed-y> = sqrt(10)
    '''
    omega = 2*pi/360
    R = (1000/omega**2)**(1/3)
    output = task_bonus(
        6000,
        [
            [1000, 0, 6, 0, 0],
            [0.001, R, 0, 0, -omega*R],
            [0.001, 0, R, omega*R, 0],
            [0.001, 0, -R, -omega*R, 0]
        ]
    )
    print(str(output))
