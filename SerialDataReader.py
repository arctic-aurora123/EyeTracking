import serial
import keyboard
class SerialDataReader():
    def __init__(self, port, baudrate):
        self.ser = serial.Serial(port, baudrate)
        
    def read(self,):
        flag = True
        while not(keyboard.is_pressed('space')): pass
        
        data = self.ser.read(self.ser.in_waiting).decode('utf-8').split('\n')
        start_str = data[1].split(', Timestamp')[0].split(':')[-1]
        start = int(start_str)
        end_str = data[-2].split(', Timestamp')[0].split(':')[-1]
        end = int(end_str)
        
        self.ser.flushInput()  
        self.ser.flushOutput()
        return start, end