"""枚举 libva.so.2 暴露的 profile + entrypoint 矩阵（不需要 root）。"""
import ctypes
import os

os.environ.setdefault("LIBVA_DRIVER_NAME", "iHD")

va = ctypes.CDLL("libva.so.2")
va_drm = ctypes.CDLL("libva-drm.so.2")

va_drm.vaGetDisplayDRM.restype = ctypes.c_void_p
va_drm.vaGetDisplayDRM.argtypes = [ctypes.c_int]
va.vaInitialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
va.vaQueryVendorString.restype = ctypes.c_char_p
va.vaQueryVendorString.argtypes = [ctypes.c_void_p]
va.vaMaxNumProfiles.argtypes = [ctypes.c_void_p]
va.vaMaxNumEntrypoints.argtypes = [ctypes.c_void_p]
va.vaQueryConfigProfiles.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
]
va.vaQueryConfigEntrypoints.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
]

PROFILES = {
    -1: "None",
    0: "MPEG2Simple", 1: "MPEG2Main",
    3: "MPEG4Simple", 4: "MPEG4AdvSimple",
    5: "H264Baseline", 6: "H264Main", 7: "H264High",
    8: "VC1Simple", 9: "VC1Main", 10: "VC1Advanced",
    11: "H263Baseline", 12: "JPEGBaseline",
    13: "H264ConstrainedBaseline", 14: "VP8Version0_3", 15: "H264MultiviewHigh",
    16: "H264StereoHigh", 17: "HEVCMain", 18: "HEVCMain10",
    19: "VP9Profile0", 20: "VP9Profile1", 21: "VP9Profile2", 22: "VP9Profile3",
    23: "HEVCMain12", 24: "HEVCMain422_10", 25: "HEVCMain422_12",
    26: "HEVCMain444", 27: "HEVCMain444_10", 28: "HEVCMain444_12",
    29: "HEVCSccMain", 30: "HEVCSccMain10", 31: "HEVCSccMain444",
    32: "AV1Profile0", 33: "AV1Profile1", 34: "HEVCSccMain444_10",
    35: "VVCMain", 36: "VVCMultilayerMain",
}
EP = {
    1: "VLD", 2: "IZZ", 3: "IDCT", 4: "MoComp", 5: "Deblocking",
    6: "EncSlice", 7: "EncPicture", 8: "EncSliceLP",
    9: "VideoProc", 10: "FEI", 11: "Stats",
    12: "EncPackedHeader", 13: "ProtectedTEEComm", 14: "ProtectedContent",
}

fd = os.open("/dev/dri/renderD128", os.O_RDWR)
dpy = va_drm.vaGetDisplayDRM(fd)
print(f"display=0x{dpy:x}")

mj, mn = ctypes.c_int(0), ctypes.c_int(0)
ret = va.vaInitialize(dpy, ctypes.byref(mj), ctypes.byref(mn))
print(f"vaInitialize ret={ret}, VA-API {mj.value}.{mn.value}")
print(f"Vendor: {va.vaQueryVendorString(dpy).decode()}")

np_max = va.vaMaxNumProfiles(dpy)
ne_max = va.vaMaxNumEntrypoints(dpy)
print(f"max profiles={np_max}, max entrypoints={ne_max}")

prof_buf = (ctypes.c_int * np_max)()
np_actual = ctypes.c_int(0)
va.vaQueryConfigProfiles(dpy, prof_buf, ctypes.byref(np_actual))
print(f"\n=== Driver 实际暴露 {np_actual.value} 个 profile ===")
print(f"{'ID':>4}  {'Profile':<24}  Entrypoints")
print("-" * 78)
for i in range(np_actual.value):
    p = prof_buf[i]
    eps_buf = (ctypes.c_int * ne_max)()
    ne_actual = ctypes.c_int(0)
    va.vaQueryConfigEntrypoints(dpy, p, eps_buf, ctypes.byref(ne_actual))
    eps = [EP.get(eps_buf[j], f"EP{eps_buf[j]}") for j in range(ne_actual.value)]
    pname = PROFILES.get(p, f"Profile{p}")
    print(f"{p:>4}  {pname:<24}  {eps}")
