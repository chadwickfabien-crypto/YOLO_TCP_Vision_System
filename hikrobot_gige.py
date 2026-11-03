# -*- coding: utf-8 -*-
"""
=====================================================================
 海康机器人 GigE 工业相机 - MVS SDK 网络取流封装
---------------------------------------------------------------------
 相机：MV-CS060-10GC 等海康机器人（Hikrobot）GigE Vision 工业相机。
 这类相机只支持 GigE 网络协议，不支持 RTSP / 网页登录。
 本模块用 ctypes 直调 MVS 运行时 MvCameraControl.dll：
     枚举(GigE) → 创建句柄 → 打开 → 开始取流 → 抓帧 → 转 BGR numpy
 使用方式（在 auto_loading_robot.py 中）：
     from hikrobot_gige import HikRobotGigeCamera
     cam = HikRobotGigeCamera()
     ok, msg = cam.open(ip="169.254.154.100")   # 或 cam.open() 自动找第一台
     while ...:
         frame = cam.read(1000)                  # BGR ndarray 或 None
     cam.release()
=====================================================================
"""

import ctypes
import os
import socket
import threading

# 注意导入顺序：必须先 numpy 后 cv2（cv2 内部会导入 numpy，顺序颠倒时
# 在部分环境下 numpy 会随机导入失败，导致 np=None 与后续崩溃）。
import numpy as np  # noqa: E402
try:
    import cv2
except Exception:
    cv2 = None

# MVS 运行时 DLL 常见安装路径（优先用已装版本）
_DLL_CANDIDATES = [
    r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64\MvCameraControl.dll",
    r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64\MvCameraControl.dll",
]

# 模块级 DLL 单例：整个进程只加载一次 MvCameraControl.dll。
# 实测同一进程二次 ctypes.CDLL 加载该 DLL 会触发访问冲突（SDK 缺陷）。
_DLL_SINGLETON = None
_DLL_SINGLETON_LOCK = threading.Lock()


def _get_dll():
    """进程内共享的 MvCameraControl.dll 句柄（仅加载一次）"""
    global _DLL_SINGLETON
    if _DLL_SINGLETON is not None:
        return _DLL_SINGLETON
    with _DLL_SINGLETON_LOCK:
        if _DLL_SINGLETON is not None:
            return _DLL_SINGLETON
        for p in _DLL_CANDIDATES:
            if os.path.isfile(p):
                try:
                    _DLL_SINGLETON = ctypes.CDLL(p)
                    return _DLL_SINGLETON
                except Exception:
                    _DLL_SINGLETON = None
    return None

MV_GIGE_DEVICE = 1            # nTLayerType：GigE
MV_ACCESS_EXCLUSIVE = 0       # 独占
MV_ACCESS_CONTROL = 1         # 控制
MV_ACCESS_READ = 2            # 只读

# GigE 常见像素格式
MV_PIXEL_MONO8     = 0x01080001
MV_PIXEL_BAYERBG8  = 0x01080008
MV_PIXEL_BAYERRG8  = 0x01080009
MV_PIXEL_BAYERGB8  = 0x0108000A
MV_PIXEL_BAYERGR8  = 0x0108000B
MV_PIXEL_BGR8      = 0x0210000C
MV_PIXEL_RGB8      = 0x02100014

_BAYER2CV = {
    MV_PIXEL_BAYERBG8: cv2.COLOR_BayerBG2BGR if cv2 else None,
    MV_PIXEL_BAYERRG8: cv2.COLOR_BayerRG2BGR if cv2 else None,
    MV_PIXEL_BAYERGB8: cv2.COLOR_BayerGB2BGR if cv2 else None,
    MV_PIXEL_BAYERGR8: cv2.COLOR_BayerGR2BGR if cv2 else None,
}


MV_MAX_DEVICE_NUM = 256  # MVS 官方上限（MV_CC_DEVICE_INFO_LIST 的指针数组长度）


class _DevList(ctypes.Structure):
    # 官方定义：unsigned int nDeviceNum;
    #          MV_CC_DEVICE_INFO* pDeviceInfo[MV_MAX_DEVICE_NUM];
    # 必须按 256 指针分配，否则 SDK 初始化整个数组时会越界写坏 Python 堆。
    _fields_ = [("nDeviceNum", ctypes.c_uint),
                ("pDeviceInfo", ctypes.c_void_p * MV_MAX_DEVICE_NUM)]


class _FrameInfoEx(ctypes.Structure):
    """MV_FRAME_OUT_INFO_EX（MVS SDK 4.x / 本机 67698691）。

    官方布局（前 104 字节）+ MVS 4.x 追加的 GigE 参数段（216 字节），
    共 320 字节。实测：GetImageBuffer 会写入完整的 320 字节 stFrameInfo，
    旧版只定义到 168 字节导致 SDK 越界写坏 Python 堆（堆损坏→GC 崩溃）。
    这里额外保留 16 字节余量以防 SDK 微调。
    """
    _fields_ = [
        ("nWidth", ctypes.c_ushort), ("nHeight", ctypes.c_ushort),
        ("nPixelType", ctypes.c_uint), ("nFrameNum", ctypes.c_uint),
        ("nDevTimeStampHigh", ctypes.c_uint), ("nDevTimeStampLow", ctypes.c_uint),
        ("nHostTimeStamp", ctypes.c_uint), ("nFrameLen", ctypes.c_uint),
        ("nLostPacket", ctypes.c_uint), ("nFrameRate", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 8),
        ("nCRC16", ctypes.c_ushort), ("nReserved2", ctypes.c_ushort),
        ("nReserved3", ctypes.c_ubyte * 32),
        ("nGevSCPD", ctypes.c_uint), ("nGevSCDA", ctypes.c_uint),
        ("nGevSCPH", ctypes.c_uint), ("nGevSCDO", ctypes.c_uint),
        ("nGevSCPR", ctypes.c_uint),
        ("nReserved4", ctypes.c_uint * 53),   # 填满 320 字节并留 16 字节余量
    ]


class _FrameOut(ctypes.Structure):
    _fields_ = [("pBufAddr", ctypes.POINTER(ctypes.c_ubyte)),
                ("stFrameInfo", _FrameInfoEx)]


class _StringValue(ctypes.Structure):
    _fields_ = [("nCurValueLen", ctypes.c_uint),
                ("chCurValue", ctypes.c_char * 256)]


class HikRobotGigeCamera:
    """海康机器人 GigE 工业相机（MVS SDK 网络取流）"""

    def __init__(self, dll_path=None):
        self._dll = None
        self._handle = ctypes.c_void_p()
        self._open = False
        self._grabbing = False
        self._lock = threading.Lock()
        self.devices = []
        self.opened_device = None
        self.access_mode = ""
        self.last_error = ""
        self._load(dll_path)

    # ---------------- SDK 加载 ----------------
    def _load(self, dll_path=None):
        if dll_path:
            # 显式指定路径：仍尽量复用进程内已加载的 DLL
            dll = _get_dll()
            if dll is None and os.path.isfile(dll_path):
                try:
                    dll = ctypes.CDLL(dll_path)
                except Exception:
                    dll = None
        else:
            dll = _get_dll()
        if dll is not None:
            if not getattr(dll, "_mvs_bound", False):
                try:
                    self._bind(dll)
                    dll._mvs_bound = True
                except Exception:
                    dll = None
            if dll is not None:
                self._dll = dll
                self.last_error = ""
                return
        self.last_error = ("未找到 MVS 运行时 DLL（MvCameraControl.dll）。"
                           "请先安装海康 MVS 客户端/运行时："
                           "C:\\Program Files (x86)\\Common Files\\MVS\\...")

    def _bind(self, d):
        d.MV_CC_EnumDevices.argtypes = [ctypes.c_uint, ctypes.POINTER(_DevList)]
        d.MV_CC_EnumDevices.restype = ctypes.c_int
        d.MV_CC_CreateHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        d.MV_CC_CreateHandle.restype = ctypes.c_int
        d.MV_CC_OpenDevice.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        d.MV_CC_OpenDevice.restype = ctypes.c_int
        d.MV_CC_StartGrabbing.argtypes = [ctypes.c_void_p]
        d.MV_CC_StartGrabbing.restype = ctypes.c_int
        d.MV_CC_GetImageBuffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(_FrameOut), ctypes.c_uint]
        d.MV_CC_GetImageBuffer.restype = ctypes.c_int
        d.MV_CC_FreeImageBuffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(_FrameOut)]
        d.MV_CC_FreeImageBuffer.restype = ctypes.c_int
        d.MV_CC_StopGrabbing.argtypes = [ctypes.c_void_p]
        d.MV_CC_StopGrabbing.restype = ctypes.c_int
        d.MV_CC_CloseDevice.argtypes = [ctypes.c_void_p]
        d.MV_CC_CloseDevice.restype = ctypes.c_int
        d.MV_CC_DestroyHandle.argtypes = [ctypes.c_void_p]
        d.MV_CC_DestroyHandle.restype = ctypes.c_int
        d.MV_CC_GetStringValue.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(_StringValue)]
        d.MV_CC_GetStringValue.restype = ctypes.c_int

    # ---------------- 设备枚举 ----------------
    @staticmethod
    def _parse_info(ptr):
        """从 MV_CC_DEVICE_INFO 原始内存解析设备信息。
        字符串偏移按 MVS SDK(67698691) 实际内存布局核对过：
        model@0x54, fw@0x74, mfr@0x94, sn@0xC4, 相机IP(u32)@0x2C。"""
        d = {"model": "", "sn": "", "fw": "", "mfr": "", "ip": "", "ptr": ptr}
        try:
            buf = ctypes.string_at(ptr, 520)

            def s(off, n=32):
                end = buf.find(b"\x00", off)
                if end == -1 or end > off + n:
                    end = off + n
                return buf[off:end].decode("ascii", errors="ignore").strip()

            d["model"] = s(0x54, 32)
            d["fw"] = s(0x74, 32)
            d["mfr"] = s(0x94, 32)
            d["sn"] = s(0xC4, 32)
            v = int.from_bytes(buf[0x2C:0x30], "little")
            ip = socket.inet_ntoa(bytes([(v >> 24) & 0xFF, (v >> 16) & 0xFF,
                                         (v >> 8) & 0xFF, v & 0xFF]))
            if ip.split(".")[0] in ("10", "172", "192", "169"):
                d["ip"] = ip
        except Exception:
            pass
        return d

    def list_devices(self):
        """枚举 GigE 相机，返回 [{model, sn, fw, mfr, ip, ptr}]"""
        self.devices = []
        if self._dll is None:
            return []
        dl = _DevList()
        try:
            ret = self._dll.MV_CC_EnumDevices(MV_GIGE_DEVICE, ctypes.byref(dl))
        except Exception:
            return []
        if ret != 0 or dl.nDeviceNum < 1:
            return []
        for i in range(dl.nDeviceNum):
            try:
                ptr = dl.pDeviceInfo[i]
            except Exception:
                continue
            if not ptr:
                continue
            self.devices.append(self._parse_info(ptr))
        return self.devices

    # ---------------- 打开 / 取流 ----------------
    def open(self, serial=None, ip=None):
        """打开相机并开始取流。
        serial / ip 任一可用来在多个相机里挑目标；都为空时取第一台。
        返回 (ok, msg)。
        整个 CreateHandle→OpenDevice→StartGrabbing 序列全程持锁，
        与 read()/release() 互斥，可安全重复调用（先释放旧句柄）。"""
        if self._dll is None:
            return False, self.last_error or "MVS SDK 未加载"
        # 已有句柄先完整释放（release 自带锁），保证 open() 可重复安全调用
        self.release()
        devs = self.list_devices()
        if not devs:
            return False, "未找到海康GigE相机（检查网线/电源/MVS驱动，本机直连时IP应为169.254.x.x）"
        target = None
        if serial:
            target = next((x for x in devs if x.get("sn") == serial), None)
        if target is None and ip:
            target = next((x for x in devs if x.get("ip") == ip), None)
        if target is None:
            target = devs[0]

        with self._lock:
            h = ctypes.c_void_p()
            r = self._dll.MV_CC_CreateHandle(ctypes.byref(h), target["ptr"])
            if r != 0:
                return False, "创建相机句柄失败 (0x%08X)" % (r & 0xFFFFFFFF)
            self._handle = h

            # 优先独占；被 MVS 客户端等占用时退回控制模式（同样可取流）
            opened = False
            for mode, mname in ((MV_ACCESS_EXCLUSIVE, "独占"),
                                (MV_ACCESS_CONTROL, "控制")):
                r = self._dll.MV_CC_OpenDevice(h, mode)
                if r == 0:
                    self.access_mode = mname
                    opened = True
                    break
            if not opened:
                try:
                    self._dll.MV_CC_DestroyHandle(h)
                except Exception:
                    pass
                self._handle = ctypes.c_void_p()
                return False, ("打开相机失败(0x%08X)，可能被 MVS 客户端占用，"
                               "请先关闭 MVS 客户端再重试" % (r & 0xFFFFFFFF))
            self._open = True

            # 打开成功后，若枚举未取到 IP，再从相机参数读取（GigE 当前地址节点）
            if not target.get("ip"):
                try:
                    sv = _StringValue()
                    if self._dll.MV_CC_GetStringValue(h, b"GevCurrentIPAddress",
                                                     ctypes.byref(sv)) == 0:
                        s = sv.chCurValue.decode("ascii", errors="ignore").strip("\x00 ")
                        if s:
                            target["ip"] = s
                except Exception:
                    pass

            r = self._dll.MV_CC_StartGrabbing(h)
            if r != 0:
                # 持锁内联清理（不能调 release/close_device，否则死锁）
                try:
                    self._dll.MV_CC_StopGrabbing(h)
                except Exception:
                    pass
                try:
                    self._dll.MV_CC_CloseDevice(h)
                except Exception:
                    pass
                try:
                    self._dll.MV_CC_DestroyHandle(h)
                except Exception:
                    pass
                self._handle = ctypes.c_void_p()
                self._open = False
                return False, "启动取流失败 (0x%08X)" % (r & 0xFFFFFFFF)
            self._grabbing = True
        self.opened_device = target
        model = target.get("model") or "GigE相机"
        ip_txt = target.get("ip") or ip or "自动"
        return True, "%s 已连接(%s访问, IP %s)" % (
            model, self.access_mode, ip_txt)

    def read(self, timeout_ms=1000):
        """取一帧，返回 BGR ndarray；超时/失败返回 None。
        与 close_device/release 全程互斥（同一把锁），锁内二次校验
        句柄有效性，避免对已释放句柄取流。"""
        if self._dll is None:
            return None
        with self._lock:
            # 锁内二次校验：主线程可能已 close/release 本相机
            if not self._open or not self._handle:
                return None
            fo = _FrameOut()
            r = self._dll.MV_CC_GetImageBuffer(self._handle, ctypes.byref(fo), timeout_ms)
            if r != 0:
                # 超时/失败：SDK 规定此时没有缓冲区，不得调用 FreeImageBuffer
                return None
            try:
                info = fo.stFrameInfo
                w, h, pt = int(info.nWidth), int(info.nHeight), int(info.nPixelType)
                if w <= 0 or h <= 0:
                    return None
                if pt in (MV_PIXEL_BGR8, MV_PIXEL_RGB8):
                    nbytes = w * h * 3
                else:
                    nbytes = w * h
                # 立即拷为 Python 自有内存（string_at），不再让 numpy 视图
                # 引用 SDK 内部缓冲区，避免 SDK 回收缓冲区时访问已释放内存
                buf = ctypes.string_at(fo.pBufAddr, nbytes)
                raw = np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
                if pt == MV_PIXEL_BGR8:
                    img = raw.reshape(h, w, 3).copy()
                elif pt == MV_PIXEL_RGB8:
                    img = raw.reshape(h, w, 3)[:, :, ::-1].copy()
                elif pt == MV_PIXEL_MONO8:
                    img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                elif pt in _BAYER2CV:
                    img = cv2.cvtColor(raw, _BAYER2CV[pt])
                else:
                    img = None
            finally:
                self._dll.MV_CC_FreeImageBuffer(self._handle, ctypes.byref(fo))
        return img

    def close_device(self):
        """停止取流并关闭相机（句柄保留可重开）。"""
        with self._lock:
            h = self._handle
            if not h:
                return
            if self._grabbing:
                try:
                    self._dll.MV_CC_StopGrabbing(h)
                except Exception:
                    pass
                self._grabbing = False
            if self._open:
                try:
                    self._dll.MV_CC_CloseDevice(h)
                except Exception:
                    pass
                self._open = False

    def release(self):
        """完全释放：停止取流 + 关闭设备 + 销毁句柄，全程持锁，
        与 read() 严格互斥，杜绝"read 进行中销毁句柄"的竞态。"""
        with self._lock:
            h = self._handle
            if h:
                if self._grabbing:
                    try:
                        self._dll.MV_CC_StopGrabbing(h)
                    except Exception:
                        pass
                    self._grabbing = False
                if self._open:
                    try:
                        self._dll.MV_CC_CloseDevice(h)
                    except Exception:
                        pass
                    self._open = False
                try:
                    self._dll.MV_CC_DestroyHandle(h)
                except Exception:
                    pass
            self._handle = ctypes.c_void_p()
        self.opened_device = None
