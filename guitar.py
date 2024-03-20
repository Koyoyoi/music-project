import cv2
import math

from pythonosc import udp_client
client = udp_client.SimpleUDPClient('127.0.0.1', 4560)

class mpHands:
    import mediapipe as mp

    def __init__(self, maxHands = 2, detection = 0.5, tracking = 0.5):
        self.hands= self.mp.solutions.hands.Hands(model_complexity = 0, max_num_hands = maxHands, min_detection_confidence = detection, min_tracking_confidence = tracking)
        self.handsDrawing = self.mp.solutions.drawing_utils

    def Marks(self, frame, width, height, handColor = (0, 0, 0)):
        myHands = []
        handsType = []
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frameRGB)
        if results.multi_hand_landmarks != None:
            for hand in results.multi_handedness:
                handType = hand.classification[0].label
                handsType.append(handType)
            for handLandMarks in results.multi_hand_landmarks:
                myHand = []
                for landMark in handLandMarks.landmark:
                    myHand.append((int(landMark.x * width), int(landMark.y * height)))
                myHands.append(myHand)
                self.handsDrawing.draw_landmarks(
                    frame, handLandMarks, self.mp.solutions.hands.HAND_CONNECTIONS, 
                    self.handsDrawing.DrawingSpec(color = handColor, thickness = 2, circle_radius = 4),
                    self.handsDrawing.DrawingSpec(color = handColor, thickness = 2, circle_radius = 2),
                )
         
        return myHands, handsType
                   
# 根據兩點的座標，計算角度
def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ = 180
    return angle_

# 根據傳入的 21 個節點座標，得到該手指的角度
def hand_angle(hand_):
    angle_list = []
    # thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    # index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    # middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    # ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    # pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

# 根據各個手指角度得出手勢
def hand_pos(finger_angle):
    f1 = finger_angle[0]   # 大拇指角度
    f2 = finger_angle[1]   # 食指角度
    f3 = finger_angle[2]   # 中指角度
    f4 = finger_angle[3]   # 無名指角度
    f5 = finger_angle[4]   # 小拇指角度

    # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮
    if f1 >= 40 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return note[0], ["/play", [0]]
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 >= 50 and f5 >= 50:
        return note[1], ["/play", [1]]
    elif f1 >= 30 and f2 >= 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return note[2], ["/play", [2]]
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return note[3], ["/play", [3]]
    elif f1 < 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return note[4], ["/play", [4]]
    elif f1 < 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 < 50:
        return note[5], ["/play", [5]]
    elif f1 < 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return note[6], ["/play", [6]]
    else:
        return '', None

# 右手手勢播放
def hand_play(finger_angle, hand_):
    f1 = finger_angle[0]   # 大拇指角度
    f2 = finger_angle[1]   # 食指角度
    f3 = finger_angle[2]   # 中指角度
    f4 = finger_angle[3]   # 無名指角度
    f5 = finger_angle[4]   # 小拇指角度
    
    # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮
    if f1 < 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return ''
    elif f2 > 50 and f3 > 50 and f4 > 50 and f5 > 50:
        return 'play'

note = ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
rootNum = {'C':0, 'C#':1, 'Db':1, 'D':2, 'D#':3, 'Eb':3, 'E':4, 'F':5, 'F#':6,
           'Gb':6, 'G':7, 'G#':8, 'Ab':8, 'A':9, 'A#':10, 'Bb':10, 'B':11}
rootTable = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
fx = ["pluck", "piano", "blade"]
transpose = 0
sync = 0

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 60)

prepose = False

findHands = mpHands(2)
font = cv2.FONT_HERSHEY_SIMPLEX  # 印出文字的字型
Type = cv2.LINE_AA               # 印出文字的邊框
cWHITE = (255, 255, 255)
cGREEN = (0, 255, 255)
cRED = (0, 0, 255)

def changeChords():
    client.send_message("/chord", [n[:1], n[1:]])

while cam.isOpened():
    text, message = None, None
    width, height= 600, 420
    rh_pose = ''
    
    ignore, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))
    handData, handsType = findHands.Marks(frame, width, height)

    for hand, handType in zip(handData, handsType):
        
        if handType == 'Left':
            finger_angle = hand_angle(hand)
            text, message = hand_pos(finger_angle) 
            
            cv2.putText(frame, "Left", hand[0], font, 1, cRED, 2, Type) 
            cv2.putText(frame, text, (30,60), font, 2, cRED, 2, Type) # 印出文字
        
        if handType == 'Right':
            finger_angle = hand_angle(hand)
            rh_pose = hand_play(finger_angle, hand) 
            
            if prepose == rh_pose:
                rh_pose = ''
            elif rh_pose == '':
                prepose = ''
            cv2.putText(frame, "Right", hand[0], font, 1, cRED, 2, Type) 
    
    xp = int(width / 9)
    cv2.rectangle(frame, (0, height - 80), (width, height), (0, 255, 0), 3)
    for i, n in enumerate(note):
        cv2.putText(frame, str(i + 1), ((5 + (i + 1) * xp + i * 10), height - 50), font, 1, cWHITE, 2, Type)
        cv2.putText(frame, n, ((5 + (i + 1) * xp + i * 10), height - 20), font, 0.7, cWHITE, 2, Type)
    cv2.putText(frame, "Capo : " + str(transpose), (width - 100, 20), font, 0.5, cRED, 1, Type)
    cv2.putText(frame, "Synth: " + str(fx[sync]), (width - 100, 40), font, 0.5, cRED, 1, Type)
    cv2.imshow('guitar', frame)

    key = cv2.waitKey(1)

    if (key == ord(' ') or rh_pose == 'play') and message != None:
        message[1].append('d')
        prepose = rh_pose
        client.send_message(*message)
   
    if key == ord('q'):
        break
    elif key == ord('+') or rh_pose == 'thumb_up':
        transpose += 1
        prepose = rh_pose
        client.send_message("/transpose", transpose)
    elif key == ord('-') or rh_pose == 'thumb_down':
        transpose -= 1
        prepose = rh_pose
        client.send_message("/transpose", transpose)
    elif ord('0') <= key and key <= ord('2'):
        sync = int(key) - 48
        client.send_message("/synth", fx[sync])
     
cam.release()
cv2.destroyAllWindows()
