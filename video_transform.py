import cv2
import numpy as np
import os
import torch
from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from os.path import join as pjoin, basename, splitext
from utils import paramUtil
import subprocess

kinematic_chain = paramUtil.t2m_kinematic_chain
fps = 20
radius = 4
data_root = '/root/autodl-tmp/VIMO'
mean = np.load(pjoin(data_root, 'Mean.npy'))
std = np.load(pjoin(data_root, 'Std.npy'))

def plot_t2m(data, save_dir):
    # data = data * std + mean
    joint_data = data
    joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

    save_path = pjoin(save_dir, 'motion.mp4')
    plot_3d_motion(save_path, kinematic_chain, joint, title="", fps=30, radius=radius)

def get_head_traj(data):
    joint_data = data
    joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
    joint = joint.reshape(-1, 22, 3)
    MINS = joint.min(axis=0).min(axis=0)
    
    height_offset = MINS[1]
    joint[:, :, 1] -= height_offset  # 把头部高度归零
    
    head_traj = joint[:, 15]
    
    return head_traj

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

def save_traj_txt(head_traj, save_path):
    """
    将头部轨迹 [T,3] 转换成 traj.txt 格式
    head_traj: global trajectory
    """
    radius = 5.0
    # 定义pivot点为head起始点，z轴+radius
    head_start = head_traj[0]
    
    thetas, phis, xs, ys, zs = [], [], [], [], []
    for (x, y, z) in head_traj:
        # 计算相对于pivot点的偏移   
        rx, ry, rz = x - head_start[0], y - head_start[1], z - head_start[2]

        theta, phi, r = cartesian_to_spherical(- rx * radius, ry * radius, radius - rz * radius)
        thetas.append(theta)
        phis.append(phi)
        xs.append(- rx)  # x偏移量 TODO: think why?
        ys.append(ry)    # y偏移量
        zs.append(rz)    # z偏移量

    with open(save_path, "w") as f:
        f.write(" ".join([f"{t:.4f}" for t in thetas]) + "\n")
        f.write(" ".join([f"{p:.4f}" for p in phis]) + "\n")
        f.write(" ".join([f"{x:.4f}" for x in xs]) + "\n")
        f.write(" ".join([f"{y:.4f}" for y in ys]) + "\n")
        f.write(" ".join([f"{z:.4f}" for z in zs]) + "\n")

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

def load_video_and_motion(video_path, motion_path, num_frames=98):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = crop_to_aspect_ratio(frame)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {video_path}")
    
    frames = interpolate_frames(frames, target_num=num_frames)
    frames = np.array(frames).astype("float32") / 255.0  # [N,H,W,C]
    
    motion = np.load(motion_path)  # shape: (T,D)
    return frames, motion

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

class VideoMotionDataset:
    def __init__(self, split_file, num_frames=98, root_dir="."):
        self.samples = []
        self.num_frames = num_frames
        self.root_dir = root_dir
        split_path = os.path.join(root_dir, split_file)
        with open(split_path, "r") as f:
            for line in f:
                video_path, motion_path = line.strip().split()
                video_path = os.path.join(root_dir, video_path)
                motion_path = os.path.join(root_dir, motion_path)
                self.samples.append((video_path, motion_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, motion_path = self.samples[idx]
        frames, motion = load_video_and_motion(video_path, motion_path, self.num_frames)
        return frames, motion, video_path, motion_path

if __name__ == "__main__":
    dataset = VideoMotionDataset("train_debug.txt", num_frames=98, root_dir="/root/autodl-tmp/VIMO")
    print(f"Dataset size: {len(dataset)}")

    out_dir = "/root/autodl-tmp/VIMO/processed/automobile_rush_towards"
    subprocess.run(f"rm -rf {out_dir}/*", shell=True)
    os.makedirs(out_dir, exist_ok=True)
    # 清空输出目录

    for i in range(len(dataset)):
        frames, motion, video_path, motion_path = dataset[i]

        # 用视频文件名作为子文件夹名
        video_name = splitext(basename(video_path))[0]  # e.g. dodge-016-len113
        sample_dir = pjoin(out_dir, video_name)
        os.makedirs(sample_dir, exist_ok=True)

        # Step 1: 保存成临时 mp4v
        h, w = frames[0].shape[:2]
        tmp_path = pjoin(sample_dir, "video_tmp.mp4")
        final_path = pjoin(sample_dir, "video.mp4")
        out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for f in (frames * 255).astype("uint8"):
            out.write(f)
        out.release()

        # Step 2: 转换为 H.264
        convert_to_h264(tmp_path, final_path)
        
        # Step 3: 删除临时文件
        os.remove(tmp_path)

        # 保存 motion 可视化 (可能有多段)
        plot_t2m(motion, sample_dir)
        
        head_traj = get_head_traj(motion) # global head trajectory
        traj_txt_path = pjoin(sample_dir, "traj.txt")
        save_traj_txt(head_traj, traj_txt_path)

        print(f"Saved sample {video_name} to {sample_dir}")
