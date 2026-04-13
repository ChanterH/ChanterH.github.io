# 此程序用于多线程采集三种生理信号（EDA、EEG、PPG）及同步视频录制

import serial
import csv
import threading
import time
import os
import cv2
import sys
from datetime import datetime

# ================= 配置参数 =================
PORTS = {
    'EEG': 'COM9',
    'PPG': 'COM11',
    'EDA': 'COM10'
}

BAUDRATES = {
    'EDA': 9600,
    'EEG': 9600,
    'PPG': 115200
}

VIDEO_CONFIG = {
    'CAMERA_INDEX': 0,
    'WIDTH': 1280,
    'HEIGHT': 720,
    'FPS': 30.0
}

# 全局超时设置（秒）
TIMEOUT_STARTUP = 3.0  # 启动后多久没数据算失败
TIMEOUT_RUNTIME = 3.0  # 运行中途多久没数据算断开


# ============================================

class VideoCollector:
    def __init__(self, stop_event, error_event, data_folder, start_time):
        self.stop_event = stop_event
        self.error_event = error_event
        self.timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(data_folder, f"Video_data_{self.timestamp}.avi")
        self.cam_index = VIDEO_CONFIG['CAMERA_INDEX']
        self.fps = VIDEO_CONFIG['FPS']

    def run(self):
        try:
            cap = cv2.VideoCapture(self.cam_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_CONFIG['WIDTH'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_CONFIG['HEIGHT'])
            cap.set(cv2.CAP_PROP_FPS, self.fps)

            # 故障检测 1：摄像头设备是否成功调用
            if not cap.isOpened():
                print("\n❌ [致命错误 - Video] 无法打开摄像头，请检查USB连接或权限！")
                self.error_event.set()
                return

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.filename, fourcc, self.fps, (VIDEO_CONFIG['WIDTH'], VIDEO_CONFIG['HEIGHT']))

            print(f"Video: 摄像头已启动 ({VIDEO_CONFIG['WIDTH']}x{VIDEO_CONFIG['HEIGHT']} @ {self.fps}fps)...")
            time.sleep(1.0)  # 预热

            consecutive_dropped_frames = 0

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                    consecutive_dropped_frames = 0
                else:
                    consecutive_dropped_frames += 1
                    # 故障检测 2：运行时是否连续丢帧 (15帧大约0.5秒)
                    if consecutive_dropped_frames > 15:
                        print("\n❌ [致命错误 - Video] 连续丢失视频帧，摄像头可能已意外断开！")
                        self.error_event.set()
                        break

                time.sleep(0.001)

        except Exception as e:
            print(f"\n❌ [致命错误 - Video] 发生异常: {e}")
            self.error_event.set()
        finally:
            if 'cap' in locals() and cap.isOpened(): cap.release()
            if 'out' in locals(): out.release()
            print("Video: 线程已安全退出")


class EDACollector:
    def __init__(self, port, baudrate, stop_event, error_event, data_folder, start_time):
        self.port = port
        self.baudrate = baudrate
        self.stop_event = stop_event
        self.error_event = error_event
        self.timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(data_folder, f"EDA_data_{self.timestamp}.csv")

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"EDA: 已连接 {self.port}，开始采集数据...")

            with open(self.filename, 'w') as f:
                f.write("Timestamp,EDA_Value,Abs_Value\n")

                last_data_time = time.monotonic()

                while not self.stop_event.is_set():
                    # 故障检测：超时无数据判定
                    if time.monotonic() - last_data_time > TIMEOUT_RUNTIME:
                        print(f"\n❌ [致命错误 - EDA] 超过 {TIMEOUT_RUNTIME} 秒未收到数据，设备可能脱落！")
                        self.error_event.set()
                        break

                    if ser.in_waiting:
                        line = ser.readline().decode('utf-8').strip()
                        if line:
                            timestamp = datetime.now().isoformat(timespec='milliseconds')
                            f.write(f"{timestamp},{line}\n")
                            last_data_time = time.monotonic()  # 更新最后收到数据的时间

                    time.sleep(0.01)

        except serial.SerialException:
            print(f"\n❌ [致命错误 - EDA] 串口 {self.port} 被占用或不存在！")
            self.error_event.set()
        except Exception as e:
            print(f"\n❌ [致命错误 - EDA] 发生异常: {e}")
            self.error_event.set()
        finally:
            if 'ser' in locals() and ser.is_open: ser.close()
            print("EDA: 线程已安全退出")


class EEGCollector:
    def __init__(self, port, baudrate, stop_event, error_event, data_folder, start_time):
        self.port = port
        self.baudrate = baudrate
        self.stop_event = stop_event
        self.error_event = error_event
        self.timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(data_folder, f"EEG_data_{self.timestamp}.csv")

    def calculate_ratio(self, low_beta, high_beta, theta, low_alpha, high_alpha):
        denominator = theta + low_alpha + high_alpha
        return (low_beta + high_beta) / denominator if denominator != 0 else 0

    def run(self):
        try:
            mSerial = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"EEG: 已连接 {self.port}，开始采集数据...")

            with open(self.filename, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                header = ['Timestamp', 'Delta', 'Theta', 'LowAlpha', 'HighAlpha', 'LowBeta', 'HighBeta', 'LowGamma',
                          'MiddleGamma', 'Ratio']
                csv_writer.writerow(header)

                last_data_time = time.monotonic()
                data_received = False

                while not self.stop_event.is_set():
                    now = time.monotonic()
                    # 故障检测：严格处理警告，不再是只打印，而是直接触发熔断
                    if not data_received and (now - last_data_time > TIMEOUT_STARTUP):
                        print(f"\n❌ [致命错误 - EEG] 启动后 {TIMEOUT_STARTUP} 秒内未收到有效数据！")
                        self.error_event.set()
                        break
                    elif data_received and (now - last_data_time > TIMEOUT_RUNTIME):
                        print(f"\n❌ [致命错误 - EEG] 数据中断超过 {TIMEOUT_RUNTIME} 秒，蓝牙或电极片可能断开！")
                        self.error_event.set()
                        break

                    if mSerial.in_waiting > 0:
                        b = mSerial.read(1)
                        if b.hex() == 'aa':
                            b = mSerial.read(1)
                            if b.hex() == 'aa':
                                data = mSerial.read(34)
                                if len(data) == 34:
                                    Delta, Theta = data[6] * 256 + data[7], data[9] * 256 + data[10]
                                    LowAlpha, HighAlpha = data[12] * 256 + data[13], data[15] * 256 + data[16]
                                    LowBeta, HighBeta = data[18] * 256 + data[19], data[21] * 256 + data[22]
                                    LowGamma, MiddleGamma = data[24] * 256 + data[25], data[27] * 256 + data[28]

                                    total = Delta + Theta + LowAlpha + HighAlpha + LowBeta + HighBeta + LowGamma + MiddleGamma
                                    if total == 0:
                                        print("\n❌ [致命错误 - EEG] 收到全零包数据，设备工作异常！")
                                        self.error_event.set()
                                        break

                                    ratio = self.calculate_ratio(LowBeta, HighBeta, Theta, LowAlpha, HighAlpha)
                                    timestamp = datetime.now().isoformat(timespec='milliseconds')
                                    csv_writer.writerow(
                                        [timestamp, Delta, Theta, LowAlpha, HighAlpha, LowBeta, HighBeta, LowGamma,
                                         MiddleGamma, ratio])

                                    data_received = True
                                    last_data_time = time.monotonic()
                    time.sleep(0.01)

        except serial.SerialException:
            print(f"\n❌ [致命错误 - EEG] 串口 {self.port} 连接失败！")
            self.error_event.set()
        except Exception as e:
            print(f"\n❌ [致命错误 - EEG] 发生异常: {e}")
            self.error_event.set()
        finally:
            if 'mSerial' in locals() and mSerial.is_open: mSerial.close()
            print("EEG: 线程已安全退出")


class PPGCollector:
    def __init__(self, port, baudrate, stop_event, error_event, data_folder, start_time):
        self.port = port
        self.baudrate = baudrate
        self.stop_event = stop_event
        self.error_event = error_event
        self.start_time = start_time
        self.timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        self.ppg_filename = os.path.join(data_folder, f"PPG_data_{self.timestamp}.csv")
        self.vital_filename = os.path.join(data_folder, f"SPO2_HR_data_{self.timestamp}.csv")
        # 用于超时检测
        self.last_data_time = time.monotonic()

    def run(self):
        try:
            ser = serial.Serial(port=self.port, baudrate=self.baudrate, parity=serial.PARITY_NONE,
                                stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS, timeout=1)

            with open(self.ppg_filename, 'w', newline='') as f1, open(self.vital_filename, 'w', newline='') as f2:
                writer_ppg = csv.writer(f1)
                writer_vital = csv.writer(f2)
                writer_ppg.writerow(['Timestamp', 'PPG_wave'])
                writer_vital.writerow(['Timestamp', 'SPO2', 'Heart rate'])

                print(f"PPG: 已连接 {self.port}，开始采集数据...")

                while not self.stop_event.is_set():
                    # 故障检测：超时无数据判定
                    if time.monotonic() - self.last_data_time > TIMEOUT_RUNTIME:
                        print(f"\n❌ [致命错误 - PPG] 超过 {TIMEOUT_RUNTIME} 秒未收到数据，光电传感器可能脱落！")
                        self.error_event.set()
                        break

                    if ser.in_waiting:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            self.process_line(line, writer_ppg, writer_vital)
                            self.last_data_time = time.monotonic()  # 更新状态

                    time.sleep(0.001)

        except serial.SerialException:
            print(f"\n❌ [致命错误 - PPG] 串口 {self.port} 连接失败！")
            self.error_event.set()
        except Exception as e:
            print(f"\n❌ [致命错误 - PPG] 发生异常: {e}")
            self.error_event.set()
        finally:
            if 'ser' in locals() and ser.is_open: ser.close()
            print("PPG: 线程已安全退出")

    def process_line(self, line, writer_ppg, writer_vital):
        if "探头未接触" in line:
            # 你可以在这里决定：探头未接触是警告还是致命错误。目前暂定严重警告。
            print("\n⚠️ [警告 - PPG] 探头未接触手指！")
            return

        iso_time = datetime.now().isoformat(timespec='milliseconds')
        if line.startswith("PPG_wave:"):
            try:
                writer_ppg.writerow([iso_time, int(line.split(':')[1])])
            except:
                pass
        elif line.startswith("SpO2:"):
            try:
                parts = line.split(',')
                writer_vital.writerow([iso_time, int(parts[0].split(':')[1]), int(parts[1].split(':')[1])])
            except:
                pass


def main():
    print("🚀 启动实验数据采集终端")
    print("⚠️ 实验过程中任何传感器故障，都将触发全局停止！")
    print("=====================================================")

    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    data_folder = f"StudentData_{timestamp}"

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # 两个事件开关：
    # stop_event: 用于正常停止（比如按Ctrl+C，或者因为故障要停止所有子线程）
    # error_event: 子线程遇到致命问题时主动拉响的“警报器”
    stop_event = threading.Event()
    error_event = threading.Event()

    video_col = VideoCollector(stop_event, error_event, data_folder, start_time)
    eda_col = EDACollector(PORTS['EDA'], BAUDRATES['EDA'], stop_event, error_event, data_folder, start_time)
    eeg_col = EEGCollector(PORTS['EEG'], BAUDRATES['EEG'], stop_event, error_event, data_folder, start_time)
    ppg_col = PPGCollector(PORTS['PPG'], BAUDRATES['PPG'], stop_event, error_event, data_folder, start_time)

    threads = [
        threading.Thread(target=video_col.run),
        threading.Thread(target=eda_col.run),
        threading.Thread(target=eeg_col.run),
        threading.Thread(target=ppg_col.run)
    ]

    for thread in threads:
        thread.daemon = True
        thread.start()

    try:
        # 主线程担任“监工”角色
        while any(thread.is_alive() for thread in threads):
            # 不断检查是否有子线程拉响了错误警报
            if error_event.is_set():
                print("\nWarning: 探测到模态采集中断！正在停止采集···")
                stop_event.set()  # 向所有健康线程广播停止指令
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n收到用户人工终止指令，正在安全关闭所有硬件引擎...")
        stop_event.set()

    # 主线程等待所有子线程处理后事（释放端口、关闭相机等）
    for thread in threads:
        thread.join(timeout=3)

    if error_event.is_set():
        print(f"\n实验异常终止。本次采集失败。数据残片位于: {os.path.abspath(data_folder)}")
        sys.exit(1)  # 返回错误状态码，如果被外部批处理调用可以知道失败了
    else:
        print(f"\n多模态数据集已完整保存至: {os.path.abspath(data_folder)}")
        sys.exit(0)


if __name__ == "__main__":
    main()