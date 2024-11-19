import random
import threading
from pyrfuniverse.envs.base_env import RFUniverseBaseEnv
import keyboard
from SerialDataReader import SerialDataReader
from Eyedata import EyeData
import pickle
import time
        
class EyeAnnotation(RFUniverseBaseEnv):
    def __init__(self):
        super().__init__()
        # 屏幕物理尺寸
        self.screen_size = []
        # 屏幕像素尺寸
        self.screen_pixel_size = [2560, 1440]
        self.AddListenerObject("Pos", self.Pos)
        self.AddListenerObject("MoveDone", self.MoveDone)
        self.move_done = False
        self.blink = False
        self.listener_thread = threading.Thread(target=self.listen_key)
        self.listener_thread.start()

    def listen_key(self):
        # while 1:
        #     keyboard.wait('x')
        #     time.sleep(0.1)
        #     _, blink = S.read()
        #     new_data.blink_time.append(blink)
        #     print(f"c_pressed: {blink}")
        
        pass

    def Pos(self, obj):
        self.pos = obj

    def MoveDone(self, obj):
        self.move_done = True

    def SetMovePos(self, screen_pos):
        self.step(count=30)
        screen_pos = [self.screen_pixel_size[0] * screen_pos[0], self.screen_pixel_size[1] * screen_pos[1]]
        self.move_done = False
        print(f"新目标点：{screen_pos}")
        self.SendObject("SetPos", screen_pos, 500)
        while not self.move_done:
            self.step()
            new_data.pos_x.append(self.pos[0])
            new_data.pos_y.append(self.pos[1])
            print(self.pos)

if __name__ == "__main__":
    idx = 0
    process = EyeAnnotation()
    S = SerialDataReader("COM7", 1000000) 
    data = []
    while 1:
        new_data = EyeData()
        keyboard.wait('space')
        new_data.start_cnt, _ = S.read()
        process.SetMovePos([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)])
        _, new_data.end_cnt = S.read()
        print(f"{idx} - start: {new_data.start_cnt}, end: {new_data.end_cnt}\n")
        idx+=1
        print(new_data.blink_time)
        data.append(new_data)
        with open('note_data/Eyedata_29.pkl', 'wb') as f:
            pickle.dump(data, f) 