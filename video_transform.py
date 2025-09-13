import cv2
import numpy as np
import os
import torch
from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from os.path import join as pjoin, basename, splitext
from utils import paramUtil
from utils.my_utils import cartesian_to_spherical, crop_to_aspect_ratio, interpolate_frames, convert_to_h264
import subprocess

kinematic_chain = paramUtil.t2m_kinematic_chain
fps = 20
radius = 4
data_root = '/root/autodl-tmp/VIMO'
mean = np.load(pjoin(data_root, 'Mean.npy'))
std = np.load(pjoin(data_root, 'Std.npy'))

def plot_t2m(data, save_dir, head_traj_smooth=None):
    # data = data * std + mean
    joint_data = data
    joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

    save_path = pjoin(save_dir, 'motion.mp4')
    plot_3d_motion(save_path, kinematic_chain, joint, title="", fps=30, radius=radius, smooth_traj=head_traj_smooth)

def get_head_traj(data):
    joint_data = data
    joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
    joint = joint.reshape(-1, 22, 3)
    MINS = joint.min(axis=0).min(axis=0)
    
    height_offset = MINS[1]
    joint[:, :, 1] -= height_offset
    
    head_traj = joint[:, 15]
    
    return head_traj

def smooth_head_traj(head_traj, n_segments=5):
    """
    head_traj: numpy array [T, 3]
    n_segments: 拟合的分段数
    """
    from scipy.optimize import least_squares
    
    def bezier(t, P0, P1, P2, P3):
        return (1-t)**3 * P0 + 3*(1-t)**2*t * P1 + 3*(1-t)*t**2 * P2 + t**3 * P3

    def residual(control_points, t, target_points, P0, P3):
        P1, P2 = control_points.reshape(2, -1)
        curve = np.array([bezier(tt, P0, P1, P2, P3) for tt in t])
        return (curve - target_points).ravel()

    seg_len = len(head_traj) // n_segments
    curves = []
    for i in range(n_segments):
        start = i * seg_len
        end = (i+1) * seg_len if i < n_segments-1 else len(head_traj)-1
        seg_points = head_traj[start:end+1]
        P0, P3 = seg_points[0], seg_points[-1]
        t = np.linspace(0, 1, len(seg_points))
        init_control = np.array([seg_points[1], seg_points[-2]]).ravel()
        res = least_squares(residual, init_control, args=(t, seg_points, P0, P3))
        P1, P2 = res.x.reshape(2, -1)
        curves.append((P0, P1, P2, P3))

    # 重建平滑轨迹
    smooth_traj = []
    for (P0, P1, P2, P3) in curves:
        t_vals = np.linspace(0, 1, 50)  # 每段采样50点
        seg_curve = np.array([bezier(tt, P0, P1, P2, P3) for tt in t_vals])
        smooth_traj.append(seg_curve)
    smooth_traj = np.vstack(smooth_traj)
    
    from scipy.interpolate import interp1d
    x = np.linspace(0, 1, len(smooth_traj))
    f = interp1d(x, smooth_traj, axis=0)
    smooth_traj = f(np.linspace(0, 1, len(head_traj)))  # 插值到原始长度
    return smooth_traj

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

class VideoMotionDataset:
    def __init__(self, split_file, num_frames=98, root_dir="."):
        self.samples = []
        self.num_frames = num_frames
        self.root_dir = root_dir
        split_path = os.path.join(root_dir, 'split', split_file)
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
    category = "automobile_rush_towards"
    txt_name = f"{category}.txt"
    
    dataset = VideoMotionDataset(txt_name, num_frames=98, root_dir="/root/autodl-tmp/VIMO")
    print(f"Dataset size: {len(dataset)}")

    out_dir = os.path.join(data_root, "processed", category)
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
        
        head_traj = get_head_traj(motion) # global head trajectory
        head_traj_smooth = smooth_head_traj(head_traj, n_segments=3)
        
        plot_t2m(motion, sample_dir, head_traj_smooth=head_traj_smooth)
        
        traj_txt_path = pjoin(sample_dir, "traj.txt")
        save_traj_txt(head_traj_smooth, traj_txt_path)

        print(f"Saved sample {video_name} to {sample_dir}")
