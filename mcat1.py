import numpy as np
import math
import cv2
import mediapipe as mp
import pyautogui as pg

pg.PAUSE = 0
pg.FAILSAFE = 0

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

from pynput.mouse import Button, Controller

class RemoteMouse:
    def __init__(self):
        self.mouse = Controller()

    def getPosition(self):
        return self.mouse.position

    def setPos(self, xPos, yPos):
        self.mouse.position = (xPos, yPos)
    def movePos(self, xPos, yPos):
        self.mouse.move(xPos, yPos)

    def click(self):
        self.mouse.click(Button.left)
    def doubleClick(self):
        self.mouse.click(Button.left, 2)
    def clickRight(self):
        self.mouse.click(Button.right)
    
    def down(self):
        self.mouse.press(Button.left)
    def up(self):
        self.mouse.release(Button.left)

def create_dimage(h, w, d):
    image = np.zeros((h, w,  d), np.uint8)
    color = tuple(reversed((0,0,0)))
    image[:] = color
    return image

class CAT:
    SENSE = 2.3
    hands = mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)
    
    def __init__(self,i_shape):
        self.cus_bef = [-1,-1]
        self.cus_cur = [-1,-1]
        self.finger = [[0,0],[0,0],[0,0],[0,0],[0,0]]
        self.shape = 0
        self.stdp_bef = [-1,-1]
        self.stdp = [-1,-1]
        self.Rclicking = False
        self.Dclicking = False
        self.image_shape = i_shape
        self.mouse = RemoteMouse()

    def act_move(self):
        self.cus_cur = [self.finger[1][0], self.finger[1][1]]
        if self.cus_bef == [-1,-1]: self.cus_bef = self.cus_cur
        self.cus_dif = [self.cus_cur[0]-self.cus_bef[0], self.cus_cur[1]-self.cus_bef[1]]
        if abs(self.cus_dif[0]) < 0.3 : self.cus_dif[0] = 0
        if abs(self.cus_dif[1]) < 0.3 : self.cus_dif[1] = 0
        
        moveX = math.sqrt(pow(abs(self.cus_dif[0]*3),3))*(1 if self.cus_dif[0]>0 else -1)
        moveY = math.sqrt(pow(abs(self.cus_dif[1]*3),3))*(1 if self.cus_dif[1]>0 else -1)
        
        return (moveX,moveY)

    def act_subMove(self):
        dX,dY = self.act_move()
        self.mouse.movePos(CAT.SENSE * dX,CAT.SENSE * dY)
        self.cus_bef = self.cus_cur
        
    def act_Rclick(self):
        if not self.Rclicking :
            self.Rclicking = True
            self.mouse.clickRight()

    def act_Dclick(self):
        if not self.Dclicking :
            self.Dclicking = True
            self.mouse.doubleClick()

    def act_scroll(self):
        stdp_ydif = self.stdp[1]-self.stdp_bef[1]
        if abs(stdp_ydif) < 0.3 : stdp_ydif = 0
        
        moveY = CAT.SENSE * math.sqrt(pow(abs(stdp_ydif*3),3))*(1 if stdp_ydif>0 else -1)
        
        pg.scroll((-1)*int(moveY))

    def action(self,sh):
        if sh == 19 :
            self.act_subMove()
            self.mouse.down()
        else :
            self.mouse.up()
            if sh == 3 :
                self.act_subMove()
                return
            else : self.cus_bef = [-1,-1]
        
        if sh == 24 : self.act_Dclick()
        else : self.Dclicking = False
        
        if sh == 18 : self.act_Rclick()
        else : self.Rclicking = False
        
        if sh == 6 : self.act_scroll()

    def operate(self,video):
        image = cv2.cvtColor(cv2.flip(video, 1), cv2.COLOR_BGR2RGB)
        results = CAT.hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        dimage = create_dimage(self.image_shape[0],self.image_shape[1],self.image_shape[2])

        if results.multi_hand_landmarks:
            for hls in results.multi_hand_landmarks:
                self.stdp = (hls.landmark[0].x * 100, hls.landmark[0].y * 100)
                
                self.finger = [(hls.landmark[4].x * 100, hls.landmark[4].y * 100),
                          (hls.landmark[8].x * 100, hls.landmark[8].y * 100),
                          (hls.landmark[12].x * 100, hls.landmark[12].y * 100),
                          (hls.landmark[16].x * 100, hls.landmark[16].y * 100),
                          (hls.landmark[20].x * 100, hls.landmark[20].y * 100)]
                discr = [(hls.landmark[2].x * 100, hls.landmark[2].y * 100),
                          (hls.landmark[6].x * 100, hls.landmark[6].y * 100),
                          (hls.landmark[10].x * 100, hls.landmark[10].y * 100),
                          (hls.landmark[14].x * 100, hls.landmark[14].y * 100),
                          (hls.landmark[17].x * 100, hls.landmark[17].y * 100)]
                
                is_folded = [(self.finger[0][0] > discr[0][0]),
                             (self.finger[1][1] > discr[1][1]),
                             (self.finger[2][1] > discr[2][1]),
                             (self.finger[3][1] > discr[3][1]),
                             (self.finger[4][1] > discr[4][1])]
                self.shape = is_folded[0]*16 + is_folded[1]*8 + is_folded[2]*4 + is_folded[3]*2 + is_folded[4]*1

                self.action(self.shape)
                self.stdp_bef = self.stdp

                cv2.putText(
                    dimage, text='stdp=(%d,%d) f : %d %d %d %d %d' % (self.stdp[0],self.stdp[1],
                                                                      is_folded[0],
                                                                      is_folded[1],
                                                                      is_folded[2],
                                                                      is_folded[3],
                                                                      is_folded[4]),
                    org=(10, 30),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=255, thickness=3)
        
                mp_drawing.draw_landmarks(dimage, hls, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('CAT', dimage)

    def stop(self):
        cv2.destroyWindow('CAT')
        del self

    def __del__(self):
        print("Thank you for using CAT!!!\n개발자 : 김주완\n버전 1.0.1-beta")
