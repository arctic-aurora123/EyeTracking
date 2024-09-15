import random
from pyrfuniverse.envs.base_env import RFUniverseBaseEnv
import keyboard
import pandas as pd
from SerialDataReader import SerialDataReader

class EyeAnnotation(RFUniverseBaseEnv):
    def __init__(self):
        super().__init__()
        # 屏幕物理尺寸
        self.screen_size = []
        # 屏幕像素尺寸
        self.screen_pixel_size = [2560, 1440]
        self.AddListenerObject("MoveDone", self.MoveDone)
        self.move_done = False

    def SetPos(self, screen_pos):
        screen_pos = [self.screen_pixel_size[0] * screen_pos[0], self.screen_pixel_size[1] * screen_pos[1]]
        self.move_done = False
        self.SendObject("SetPos", screen_pos)
        while not self.move_done:
            self.step(simulate=False)
        print("屏幕目标点变成红色时，看向目标点并按空格键")
        start, end = Serial.read()
        df.loc[len(df)] = [screen_pos[0],screen_pos[1], start, end]
        df.to_csv("C:/Users/Arctic/Documents/ads1299/saved/pos_count.csv", index=False)
        print(df)
        # keyboard.wait('space')
        # To Do
        # 保存当前pos和肌电数据

    def MoveDone(self, obj):
        self.move_done = True


if __name__ == "__main__":
    i = 0
    
    df = pd.DataFrame(columns=['pos_x', 'pos_y', 'start_cnt', 'end_cnt'])
    Serial = SerialDataReader("COM4", 1000000)
    process = EyeAnnotation()

    pos_start = [0.5, 0.5]
    pos = [[0.2, 0.8], [0.5, 0.8], [0.8, 0.8], [0.2, 0.5], [0.5, 0.5], [0.8, 0.5], [0.2, 0.2], [0.5, 0.2], [0.8, 0.2]]

    # 初始屏幕中心
    process.SetPos(pos_start)
    # 9点标定
    for p in pos:
        i += 1
        process.SetPos(p)
    # 随机位置
    while 1:
        i += 1
        process.SetPos([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)])
        if keyboard.is_pressed('q'):
            print(df)
            exit()
         