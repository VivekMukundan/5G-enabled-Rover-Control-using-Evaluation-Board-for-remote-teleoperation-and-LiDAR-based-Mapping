#!/usr/bin/env python3
# rover_serial_forwarder_client.py  -- run on Rover (client mode)
import argparse, socket, threading, sys, time, serial

BUF_SIZE = 4096

def serial_reader(ser, conn, stop_event):
    try:
        while not stop_event.is_set():
            if ser.in_waiting:
                data = ser.read(ser.in_waiting)
                if data:
                    try:
                        conn.sendall(data)
                    except:
                        stop_event.set()
                        break
            else:
                time.sleep(0.002)
    except Exception as e:
        stop_event.set()

def socket_reader(ser, conn, stop_event):
    try:
        while not stop_event.is_set():
            data = conn.recv(BUF_SIZE)
            if not data:
                stop_event.set()
                break
            ser.write(data)
    except Exception as e:
        stop_event.set()

def connect_and_forward(serial_port, baudrate, eval_host, eval_port):
    print(f"[rover-client] Opening serial {serial_port} @ {baudrate}")
    ser = serial.Serial(serial_port, baudrate, timeout=0)
    time.sleep(0.2)
    while True:
        try:
            print(f"[rover-client] Connecting to eval {eval_host}:{eval_port} ...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(6)
            sock.connect((eval_host, eval_port))
            sock.settimeout(None)
            print("[rover-client] Connected. Starting forwarding.")
            stop_event = threading.Event()
            t1 = threading.Thread(target=serial_reader, args=(ser, sock, stop_event), daemon=True)
            t2 = threading.Thread(target=socket_reader, args=(ser, sock, stop_event), daemon=True)
            t1.start(); t2.start()
            while not stop_event.is_set():
                time.sleep(0.1)
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except:
                pass
            sock.close()
            print("[rover-client] Connection closed. Reconnecting in 2s...")
        except Exception as e:
            print("[rover-client] Connect failed:", e, "retry in 2s")
            time.sleep(2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--serial', default='/dev/ttyACM0', help='serial device')
    p.add_argument('--baud', type=int, default=115200)
    p.add_argument('--host', required=True, help='eval board IP')
    p.add_argument('--port', type=int, default=25001)
    args = p.parse_args()
    try:
        connect_and_forward(args.serial, args.baud, args.host, args.port)
    except KeyboardInterrupt:
        print("Exiting")
        sys.exit(0)
