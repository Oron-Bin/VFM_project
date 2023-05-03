import serial
import time


def jsonize(key,data): #how the data is tansform in the packet
    packet = 'json:{"'+str(key)+'":'+str(data)+'}'+'\x0d'+'\x0a'
    return packet

class Card:
    def __init__(self, x_d: object, y_d: object, a_d: object, x: object, y: object, a: object, baud: object, port: object) -> object:
        self.x = x
        self.y = y
        self.a = a
        self.x_d = x_d
        self.y_d = y_d
        self.a_d = a_d
        self.motor_a = 0
        self.encoder = 0
        self.vibrate = 0
        self.start = 0
        self.key_dict = {'x_des': self.x_d,
                         'y_des': self.y_d,
                         'a_des': self.a_d,
                         'x_act': self.x,
                         'y_act':self.y,
                         'a_act': self.a,
                         'motor': self.motor_a,
                         'encoder': self.encoder,
                         'vibrate': self.vibrate,
                         'st': self.start
                         }
        self.usb = serial.Serial(timeout=0.000001) ## Create serial port
        self.usb.port = port ## Name of the port
        self.usb.baudrate = baud ## Defined baudrate

    def set_x(self,x): ## Set desired x location
        self.x_d = x
        return None

    def set_y(self,y): ## set desired y location
        self.y_d = y
        return None

    def set_actual(self,x,y,a):
        self.x = x
        self.y = y
        self.a = a
        print(x)
        print(y)
        print(a)
        return None

    def set_angle(self): ## set desired orientation
        print('Input the desired orientation: ')
        self.a_d = int(input())
        return None

    def send_data(self,key): ## This function send data to controller
        data = jsonize(key,self.key_dict[key]) ## convert data to json package
        self.usb.open()
        rx_byte = ''
        if self.usb.writable():
            # self.usb.write(data.encode()) ## Oron try this to write full string and not byte by byte
            for tx_byte in data:
                self.usb.write(tx_byte.encode()) ## Write a full string
                # rx_byte = rx_byte + self.usb.read().decode('utf-8')
            # print('Sended data :' + rx_byte)

            self.usb.close()
            # self.usb.open()
            # print(self.usb.readline())
            # self.usb.close()

            # print('Length of bytes sended: ' + len(data))
            # print('Length of bytes recieved: ' + len(rx_byte))
            if (len(rx_byte) != len(data)):
                # print('o')
                return 0

        else:
            self.usb.close()
            print('Serial Bus is not writable')

        return 1
    def stop_initial(self):
        byte = 'a'
        self.usb.open()
        self.usb.write(byte.encode())
        # print(self.usb.readline())
        self.usb.close()
        return None
    def set_motor_angle(self,angle): ## Defines which angle the motor need to move
        self.key_dict['motor'] = angle
        # print(self.motor_a)
        return None
    def set_encoder_angle(self,angle):
        self.key_dict['encoder'] = angle
        # print(self.motor_a)
        return None
    def vibrate_on(self): ## set desired orientation
        self.key_dict['vibrate'] = 1
        return None
    def vibrate_off(self): ## set desired orientation
        self.key_dict['vibrate'] = 0
        return None






# Format: json:{"name": "dsp", "mode": "off"}\r\n

# def int_to_bytes(number: int) -> bytes:
#     return number.to_bytes(length=(8 + (number + (number < 0)).bit_length()) // 8, byteorder='big', signed=True)
#
# def int_from_bytes(binary_data: bytes) -> [int]:
#     return int.from_bytes(binary_data, byteorder='big', signed=True)

