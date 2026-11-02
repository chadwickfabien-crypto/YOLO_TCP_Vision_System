# -*- coding: utf-8 -*-
"""
=====================================================================
 自动装卸机器人 - YOLO 视觉检测界面（适配 Python 3.8+ / PyQt5）
---------------------------------------------------------------------
 新版 YOLO 接口（ultralytics，YOLOv8 / YOLOv11 通用）：
     from ultralytics import YOLO
     model = YOLO("best0408.pt")
     results = model(frame, conf=0.5, iou=0.46)

 两个 TCP 通信连接（按老师要求）：
     TCP1  机械臂/下位机 —— 检测到杯子自动下发坐标 CUP,x,y,w,h,conf
     TCP2  上位机/PLC    —— 上报状态，可接收 START/STOP/PAUSE 指令

 依赖：pip install ultralytics pyqt5 opencv-python numpy
=====================================================================
"""

import os
import sys
import time
import socket
import threading

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QLinearGradient
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QComboBox, QLineEdit, QCheckBox, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFileDialog, QGroupBox, QFrame, QMessageBox
)

# ---------- YOLO 新版接口 ----------
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception as e:
    HAS_ULTRALYTICS = False
    _IMPORT_ERR = e


# =====================================================================
#  通用 TCP 通道（后台线程，自动重连）
# =====================================================================
class TcpChannel(QObject):
    connected   = pyqtSignal(str)            # 状态文案
    received    = pyqtSignal(str)            # 收到的文本
    box_sent    = pyqtSignal(str)            # 已下发的坐标日志

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.name = name
        self.sock = None
        self.connected_flag = False
        self._want_connect = False
        self._lock = threading.Lock()
        self._thread = None

    # ---- 主动连接 ----
    def connect_to(self, ip, port):
        self.disconnect()
        self._want_connect = True
        self._thread = threading.Thread(
            target=self._reconnect_loop, args=(ip, int(port)), daemon=True)
        self._thread.start()

    def disconnect(self):
        self._want_connect = False
        self.connected_flag = False
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None

    def _reconnect_loop(self, ip, port):
        while self._want_connect:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2.0)
                s.connect((ip, port))
                s.settimeout(0.5)
                self.sock = s
                self.connected_flag = True
                self.connected.emit(f"[{self.name}] 已连接 {ip}:{port}")
                self._recv_loop(s)
            except Exception as e:
                self.connected.emit(f"[{self.name}] 连接失败，3s 后重连... ({e})")
                self.connected_flag = False
                time.sleep(3.0)

    def _recv_loop(self, s):
        buf = b""
        while self._want_connect and self.connected_flag:
            try:
                chunk = s.recv(1024)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self.received.emit(line.decode("utf-8", errors="ignore").strip())
            except socket.timeout:
                continue
            except Exception:
                break
        self.connected_flag = False
        self.connected.emit(f"[{self.name}] 连接断开")

    def send(self, text: str):
        """线程安全地发文本，末尾自动加 \\n"""
        if not self.connected_flag or self.sock is None:
            return False
        try:
            with self._lock:
                self.sock.sendall((text + "\n").encode("utf-8"))
            self.box_sent.emit(text)
            return True
        except Exception:
            self.connected_flag = False
            return False


# =====================================================================
#  视频读取 + YOLO 推理线程
# =====================================================================
class DetectWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray, list, dict)  # (BGR图, boxes, stats)
    fps_ready   = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.paused = False
        self.cap = None
        self.source = None

        self.model = None
        self.model_name = "best0408.pt"
        self.conf_thres = 0.50
        self.iou_thres = 0.46
        self.show_label = True
        self.show_point = False

        self._last_t = time.time()
        self._fps = 0.0

    def load_model(self, pt_path: str):
        if not HAS_ULTRALYTICS:
            return False, "未安装 ultralytics，请先 pip install ultralytics"
        if not os.path.isfile(pt_path):
            return False, f"找不到模型文件: {pt_path}"
        try:
            self.model = YOLO(pt_path)
            self.model_name = os.path.basename(pt_path)
            _ = self.model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
            return True, f"模型加载成功: {self.model_name}"
        except Exception as e:
            return False, f"模型加载失败: {e}"

    def open_source(self, source):
        self.release()
        self.source = source
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)
        if self.cap and self.cap.isOpened():
            self.running = True
            if not self.isRunning():
                self.start()
            return True
        return False

    def release(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def run(self):
        while self.running:
            if self.paused:
                time.sleep(0.03); continue
            if self.cap is None:
                time.sleep(0.05); continue
            ok, frame = self.cap.read()
            if not ok:
                if isinstance(self.source, str) and not self.source.lower().startswith("rtsp"):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                time.sleep(0.05); continue

            stats = {"classes": 0, "boxes": 0}
            boxes_out = []
            if self.model is not None:
                try:
                    results = self.model.predict(
                        frame, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
                    frame, boxes_out, stats = self._draw(frame, results[0])
                except Exception as e:
                    cv2.putText(frame, f"infer error: {e}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            now = time.time()
            dt = now - self._last_t
            self._last_t = now
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            self.fps_ready.emit(self._fps)
            self.frame_ready.emit(frame, boxes_out, stats)
            time.sleep(0.005)

    def _draw(self, frame, result):
        names = result.names
        boxes = result.boxes
        cls_set = set()
        cup_count = 0
        out = []

        if boxes is not None and len(boxes):
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                conf = float(b.conf[0])
                cls_id = int(b.cls[0])
                cls_name = names.get(cls_id, str(cls_id))
                cls_set.add(cls_name)
                color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if self.show_label:
                    text = f"{cls_name} CF:{conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if "cup" in cls_name.lower() or "杯" in cls_name:
                    cup_count += 1
                    out.append({
                        "cls": cls_name, "conf": conf,
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                        "cx": int((x1 + x2) / 2), "cy": int((y1 + y2) / 2),
                    })
        return frame, out, {"classes": len(cls_set), "boxes": cup_count}


# =====================================================================
#  顶部统计卡片
# =====================================================================
class StatCard(QWidget):
    def __init__(self, title, color1, color2, parent=None):
        super().__init__(parent)
        self.setFixedHeight(90)
        self._title = title
        self._value = "0"
        self._c1 = QColor(*color1)
        self._c2 = QColor(*color2)

    def setValue(self, v):
        self._value = str(v); self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        grad = QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0, self._c1); grad.setColorAt(1, self._c2)
        p.setBrush(grad); p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect(), 12, 12)
        p.setPen(QColor(255, 255, 255))
        p.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        p.drawText(QRect(0, 12, self.width(), 24), Qt.AlignHCenter, self._title)
        p.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        p.drawText(QRect(0, 42, self.width(), 36), Qt.AlignHCenter, self._value)
        p.end()


# =====================================================================
#  主窗口
# =====================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("自动装卸机器人")
        self.resize(1440, 820)

        # 两个 TCP 通道
        self.tcp_arm  = TcpChannel("机械臂")    # 下发坐标
        self.tcp_host = TcpChannel("上位机")    # 上行状态
        self.tcp_arm.connected.connect(self._toast)
        self.tcp_host.connected.connect(self._toast)
        self.tcp_host.received.connect(self._on_host_command)
        self._last_send_t = 0.0

        self.worker = DetectWorker()
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.fps_ready.connect(lambda f: self.card_fps.setValue(f"{f:.0f}"))

        self._build_ui()
        self._init_models()

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        root.addWidget(self._build_left_bar())
        root.addWidget(self._build_center())
        root.addWidget(self._build_right_panel())

    def _build_left_bar(self):
        bar = QFrame(); bar.setFixedWidth(180)
        bar.setStyleSheet("background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
                          "stop:0 #b088ff, stop:1 #7c5cff);")
        lay = QVBoxLayout(bar); lay.setContentsMargins(10, 20, 10, 10); lay.setSpacing(6)
        title = QLabel("自动装卸\nBy Li")
        title.setStyleSheet("color:white; font-size:16px; font-weight:bold;")
        title.setAlignment(Qt.AlignHCenter)
        lay.addWidget(title); lay.addSpacing(15)
        for text, slot in [
            ("隐藏",        self.toggle_left_bar),
            ("检测文件",    self.open_file),
            ("调用摄像头",  self.open_camera),
            ("调用RTSP监控",self.open_rtsp),
            ("计算内参",    lambda: self._toast("计算内参（按上课代码补棋盘格标定）")),
            ("点云投影",    self.toggle_point),
            ("发送货箱位置",self.send_now),
        ]:
            b = QPushButton(text); b.setFixedHeight(38)
            b.setStyleSheet("QPushButton{color:white;background:transparent;border:none;"
                            "text-align:left;padding-left:14px;font-size:14px;}"
                            "QPushButton:hover{background:rgba(255,255,255,0.18);border-radius:6px;}")
            b.clicked.connect(slot); lay.addWidget(b)
        lay.addStretch()
        self.status_label = QLabel("未连接相机")
        self.status_label.setStyleSheet("color:white; font-size:11px;")
        lay.addWidget(self.status_label)
        self._left_bar = bar
        return bar

    def _build_center(self):
        w = QFrame(); w.setStyleSheet("background:#f4f4fb;")
        v = QVBoxLayout(w); v.setContentsMargins(15, 10, 15, 10); v.setSpacing(10)
        head = QLabel("自动装卸机器人")
        head.setAlignment(Qt.AlignHCenter)
        head.setStyleSheet("font-size:16px; font-weight:bold;")
        v.addWidget(head)

        cards = QHBoxLayout()
        self.card_cls  = StatCard("总类数",   (140,120,255),(90,80,220))
        self.card_box  = StatCard("杯子数量", (255,150,150),(255,110,110))
        self.card_fps  = StatCard("帧率",     (180,130,220),(140,90,200))
        self.card_model= StatCard("使用模型", (80,220,200),(40,190,180))
        for c in (self.card_cls,self.card_box,self.card_fps,self.card_model):
            cards.addWidget(c)
        v.addLayout(cards)

        self.video = QLabel("未开始")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background:#111;color:#aaa;border-radius:8px;")
        self.video.setMinimumHeight(440)
        v.addWidget(self.video, 1)

        bottom = QHBoxLayout()
        self.btn_play = QPushButton("▶"); self.btn_play.setFixedWidth(36)
        self.btn_play.clicked.connect(self.toggle_pause)
        self.progress = QSlider(Qt.Horizontal)
        self.lbl_state = QLabel("实时视频流检测中...")
        self.lbl_state.setStyleSheet("color:#ff6b6b;")
        self.chk_label = QCheckBox("显示标签"); self.chk_label.setChecked(True)
        self.chk_label.stateChanged.connect(lambda s: setattr(self.worker,"show_label",bool(s)))
        self.chk_point = QCheckBox("显示点云")
        self.chk_point.stateChanged.connect(lambda s: setattr(self.worker,"show_point",bool(s)))
        bottom.addWidget(self.btn_play); bottom.addWidget(self.progress,1)
        bottom.addWidget(self.lbl_state); bottom.addWidget(self.chk_label); bottom.addWidget(self.chk_point)
        v.addLayout(bottom)
        return w

    def _build_right_panel(self):
        panel = QFrame(); panel.setFixedWidth(260)
        panel.setStyleSheet("background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
                            "stop:0 #f3c6ff, stop:1 #d8a8ff); border-radius:10px;")
        v = QVBoxLayout(panel); v.setContentsMargins(12,15,12,12); v.setSpacing(8)

        head = QLabel("设置"); head.setAlignment(Qt.AlignHCenter)
        head.setStyleSheet("font-size:16px;font-weight:bold;"); v.addWidget(head)

        v.addWidget(self._gt("模型"))
        self.cmb_model = QComboBox()
        self.cmb_model.setStyleSheet("background:white;padding:4px;")
        self.cmb_model.currentTextChanged.connect(self._on_model_change)
        v.addWidget(self.cmb_model)

        v.addWidget(self._gt("交互比"))
        row = QHBoxLayout()
        self.spin_iou = QLineEdit("0.46"); self.spin_iou.setFixedWidth(50); self.spin_iou.setStyleSheet("background:white;")
        self.spin_iou.returnPressed.connect(self._sync_iou)
        self.sld_iou = QSlider(Qt.Horizontal); self.sld_iou.setRange(1,95); self.sld_iou.setValue(46)
        self.sld_iou.valueChanged.connect(lambda x:(self.spin_iou.setText(f"{x/100:.2f}"),
                                                    setattr(self.worker,"iou_thres",x/100.0)))
        row.addWidget(self.spin_iou); row.addWidget(self.sld_iou); v.addLayout(row)

        v.addWidget(self._gt("置信度"))
        row = QHBoxLayout()
        self.spin_conf = QLineEdit("0.50"); self.spin_conf.setFixedWidth(50); self.spin_conf.setStyleSheet("background:white;")
        self.spin_conf.returnPressed.connect(self._sync_conf)
        self.sld_conf = QSlider(Qt.Horizontal); self.sld_conf.setRange(5,95); self.sld_conf.setValue(50)
        self.sld_conf.valueChanged.connect(lambda x:(self.spin_conf.setText(f"{x/100:.2f}"),
                                                     setattr(self.worker,"conf_thres",x/100.0)))
        row.addWidget(self.spin_conf); row.addWidget(self.sld_conf); v.addLayout(row)

        v.addWidget(self._gt("保存"))
        self.chk_save_video = QCheckBox("保存检测视频或图像")
        self.chk_save_info  = QCheckBox("保存检测信息(.txt)")
        v.addWidget(self.chk_save_video); v.addWidget(self.chk_save_info)

        # ===== TCP1：机械臂（下发坐标） =====
        v.addWidget(self._gt("TCP1 机械臂(下发坐标)"))
        g1 = QGridLayout()
        self.edt_arm_ip   = QLineEdit("127.0.0.1"); self.edt_arm_port = QLineEdit("14000")
        for e in (self.edt_arm_ip, self.edt_arm_port): e.setStyleSheet("background:white;")
        g1.addWidget(QLabel("IP"),0,0); g1.addWidget(self.edt_arm_ip,0,1)
        g1.addWidget(QLabel("端口"),0,2); g1.addWidget(self.edt_arm_port,0,3)
        self.btn_arm = QPushButton("连接")
        self.btn_arm.setCheckable(True)
        self.btn_arm.toggled.connect(self._toggle_arm)
        g1.addWidget(self.btn_arm,1,0,1,4)
        v.addLayout(g1)

        # ===== TCP2：上位机（上行状态/收指令） =====
        v.addWidget(self._gt("TCP2 上位机(状态/指令)"))
        g2 = QGridLayout()
        self.edt_host_ip   = QLineEdit("127.0.0.1"); self.edt_host_port = QLineEdit("14001")
        for e in (self.edt_host_ip, self.edt_host_port): e.setStyleSheet("background:white;")
        g2.addWidget(QLabel("IP"),0,0); g2.addWidget(self.edt_host_ip,0,1)
        g2.addWidget(QLabel("端口"),0,2); g2.addWidget(self.edt_host_port,0,3)
        self.btn_host = QPushButton("连接")
        self.btn_host.setCheckable(True)
        self.btn_host.toggled.connect(self._toggle_host)
        g2.addWidget(self.btn_host,1,0,1,4)
        v.addLayout(g2)

        v.addStretch()
        return panel

    def _gt(self, t):
        l = QLabel(t); l.setStyleSheet("font-weight:bold;margin-top:4px;"); return l

    # ---------------- 业务 ----------------
    def _init_models(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.card_model.setValue("best0408.pt")
        local = [f for f in os.listdir(script_dir) if f.lower().endswith(".pt")]
        if not local: local = ["best0408.pt"]
        self.cmb_model.addItems(local)
        pt = os.path.join(script_dir, local[0])
        ok, msg = self.worker.load_model(pt)
        self.status_label.setText(msg if not ok else "模型已就绪")

    def _on_model_change(self, name):
        if not name: return
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pt = os.path.join(script_dir, name)
        ok, msg = self.worker.load_model(pt)
        self.card_model.setValue(name); self.status_label.setText(msg)

    def _sync_iou(self):
        try: self.sld_iou.setValue(int(float(self.spin_iou.text())*100))
        except: pass
    def _sync_conf(self):
        try: self.sld_conf.setValue(int(float(self.spin_conf.text())*100))
        except: pass

    def toggle_left_bar(self): self._left_bar.setVisible(not self._left_bar.isVisible())
    def toggle_point(self):
        self.worker.show_point = not self.worker.show_point
        self.chk_point.setChecked(self.worker.show_point)
    def toggle_pause(self):
        self.worker.paused = not self.worker.paused
        self.btn_play.setText("▶" if self.worker.paused else "⏸")
        self.lbl_state.setText("已暂停" if self.worker.paused else "实时视频流检测中...")

    def open_file(self):
        p,_ = QFileDialog.getOpenFileName(self,"选择检测文件","","视频/图片 (*.mp4 *.avi *.jpg *.png)")
        if p and self.worker.open_source(p):
            self.status_label.setText(f"已打开: {os.path.basename(p)}")
    def open_camera(self):
        from PyQt5.QtWidgets import QInputDialog
        idx, ok = QInputDialog.getInt(self, "摄像头", "摄像头索引\n(0=内置摄像头, 1/2/3=外接USB摄像头)", 0, 0, 10)
        if not ok: return
        if self.worker.open_source(idx):
            self.status_label.setText(f"调用摄像头成功 (索引 {idx})")
        else:
            self.status_label.setText(f"打不开索引 {idx}，换 1 或 2 试试")
    def open_rtsp(self):
        from PyQt5.QtWidgets import QInputDialog
        url,ok = QInputDialog.getText(self,"RTSP","输入RTSP地址:",text="rtsp://169.254.249.253:554/")
        if ok and url and self.worker.open_source(url):
            self.status_label.setText(f"RTSP已连接: {url}")

    # ---- TCP 开关 ----
    def _toggle_arm(self, checked):
        if checked:
            self.tcp_arm.connect_to(self.edt_arm_ip.text(), self.edt_arm_port.text())
            self.btn_arm.setText("断开")
        else:
            self.tcp_arm.disconnect()
            self.btn_arm.setText("连接")

    def _toggle_host(self, checked):
        if checked:
            self.tcp_host.connect_to(self.edt_host_ip.text(), self.edt_host_port.text())
            self.btn_host.setText("断开")
        else:
            self.tcp_host.disconnect()
            self.btn_host.setText("连接")

    def _on_host_command(self, line: str):
        """上位机下发的指令：START / STOP / PAUSE / RESUME"""
        cmd = line.upper().strip()
        self._toast(f"收到上位机指令: {cmd}")
        if cmd == "START" or cmd == "RESUME":
            self.worker.paused = False
        elif cmd == "STOP":
            self.worker.paused = True
        elif cmd == "PAUSE":
            self.worker.paused = True

    def send_now(self):
        """手动发一次当前画面里的杯子坐标"""
        if not hasattr(self, "_last_boxes") or not self._last_boxes:
            self._toast("画面里没检测到杯子"); return
        for b in self._last_boxes:
            msg = f"CUP,{b['cx']},{b['cy']},{b['x2']-b['x1']},{b['y2']-b['y1']},{b['conf']:.2f}"
            self.tcp_arm.send(msg)

    def _toast(self, msg): self.status_label.setText(msg)

    # ---------------- 收帧 ----------------
    _last_boxes = []

    def _on_frame(self, frame, boxes, stats):
        self.card_cls.setValue(stats["classes"])
        self.card_box.setValue(stats["boxes"])
        self._last_boxes = boxes

        # 自动下发坐标（节流 0.2s 一条，避免刷屏）
        now = time.time()
        if boxes and (now - self._last_send_t) > 0.2:
            self._last_send_t = now
            # 取最大那个杯子（最近的）
            biggest = max(boxes, key=lambda b:(b["x2"]-b["x1"])*(b["y2"]-b["y1"]))
            msg = (f"CUP,{biggest['cx']},{biggest['cy']},"
                   f"{biggest['x2']-biggest['x1']},{biggest['y2']-biggest['y1']},"
                   f"{biggest['conf']:.2f}")
            self.tcp_arm.send(msg)

        # 每 1s 上报一次状态给上位机
        if not hasattr(self, "_last_report_t"): self._last_report_t = 0
        if now - self._last_report_t > 1.0:
            self._last_report_t = now
            self.tcp_host.send(f"STATUS,boxes={stats['boxes']},fps={self.card_fps._value}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        img = QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888).copy()
        self.video.setPixmap(QPixmap.fromImage(img).scaled(
            self.video.width(), self.video.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, e):
        self.worker.release(); self.worker.wait(2000)
        self.tcp_arm.disconnect(); self.tcp_host.disconnect()
        super().closeEvent(e)


if __name__ == "__main__":
    if not HAS_ULTRALYTICS:
        print("[!] 未安装 ultralytics: pip install ultralytics")
        print("   ", _IMPORT_ERR)
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    win = MainWindow(); win.show()
    sys.exit(app.exec_())
