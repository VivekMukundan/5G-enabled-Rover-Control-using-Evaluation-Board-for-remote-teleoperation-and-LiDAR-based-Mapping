import serial
import struct
import numpy as np
import time


class YDLidarX2:
    START_FLAG = 0xFA
    INDEX_MIN = 0xA0
    INDEX_MAX = 0xDB
    PACKET_SIZE = 22

    def __init__(self, port, chunk_size=360):
        self.port = port
        self.chunk_size = chunk_size
        self.serial = None
        self.buffer_angle = np.zeros(360, dtype=np.int32)
        self.last_ts = 0
        self._available = False

    def connect(self, baudrate=115200):
        try:
            self.serial = serial.Serial(
                self.port,
                baudrate=baudrate,
                timeout=1
            )
            return True
        except Exception as e:
            print("Connection error:", e)
            return False

    @property
    def available(self):
        return self._available

    def disconnect(self):
        if self.serial is not None:
            self.serial.close()
            self.serial = None

    def start_scan(self):
        # X2 starts scanning automatically when powered
        pass

    def stop_scan(self):
        # No stop command in simple protocol
        pass

    def checksum(self, data):
        # Lidar checksum rule
        chk32 = 0
        for i in range(0, 20, 2):
            chk32 = (chk32 << 1) + (data[i] + (data[i+1] << 8))
        checksum = (chk32 & 0x7FFF) + (chk32 >> 15)
        checksum &= 0x7FFF
        return checksum

    def read_packet(self):
        if self.serial.readable():
            start = self.serial.read(1)

            if len(start) == 0:
                return None

            if start[0] != self.START_FLAG:
                return None

            header = self.serial.read(21)
            if len(header) != 21:
                return None

            data = bytes([self.START_FLAG]) + header
            return data
        return None

    def parse_packet(self, data):
        if len(data) != self.PACKET_SIZE:
            return None

        index = data[1]
        if index < self.INDEX_MIN or index > self.INDEX_MAX:
            return None

        speed = struct.unpack("<H", data[2:4])[0]
        angles_base = (index - self.INDEX_MIN) * 4

        for i in range(4):
            offset = 4 + i * 4
            dist = struct.unpack("<H", data[offset:offset+2])[0]

            if dist == 0:
                dist = 32768  # invalid reading

            angle = angles_base + i
            if 0 <= angle < 360:
                self.buffer_angle[angle] = dist

        # check valid chunk
        if angles_base == 356:
            self._available = True

    def get_data(self):
        self._available = False
        return np.copy(self.buffer_angle)

    def loop(self):
        raw = self.read_packet()
        if raw:
            self.parse_packet(raw)
