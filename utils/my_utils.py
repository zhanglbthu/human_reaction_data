import torch
import numpy as np
import subprocess

def cartesian_to_spherical(x, y, z):
    """
    将笛卡尔坐标 (x,y,z) 转换为球坐标 (theta, phi, r)
    motion坐标系: z 前, y 上
    - r: 距离
    - theta: 上下角度 (绕x轴)，弧度转角度
    - phi: 左右角度 (绕y轴)，弧度转角度
    """
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-6
    theta = np.degrees(np.arcsin(y / r))    # y控制仰角
    phi = np.degrees(np.arctan2(x, z))      # x,z控制水平角
    return theta, phi, r

def crop_to_aspect_ratio(frame, target_ratio=(16, 9)):
    h, w = frame.shape[:2]
    target_w, target_h = target_ratio
    if w / h > target_w / target_h:
        new_w = int(h * target_w / target_h)
        x0 = (w - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    else:
        new_h = int(w * target_h / target_w)
        y0 = (h - new_h) // 2
        frame = frame[y0:y0 + new_h, :]
    return frame

def interpolate_frames(frames, target_num=98):
    n = len(frames)
    if n == 0:
        return []
    idxs = np.linspace(0, n - 1, target_num).astype(int)
    return [frames[i] for i in idxs]

def convert_to_h264(input_path, output_path):
    """调用 ffmpeg 把 mp4v 转成 H.264"""
    cmd = [
        "ffmpeg", "-y",  # 覆盖已存在文件
        "-i", input_path,
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion failed: {e.stderr.decode()}")