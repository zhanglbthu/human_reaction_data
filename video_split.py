import os
from collections import defaultdict

# 指定保存目录
root_dir = "/root/autodl-tmp/VIMO/split"
os.makedirs(root_dir, exist_ok=True)

# 需要处理的文件
splits = ["/root/autodl-tmp/VIMO/train.txt", "/root/autodl-tmp/VIMO/test.txt"]

# 保存类别 -> 行 的映射
category_lines = defaultdict(list)

for split in splits:
    with open(split, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 提取类别，比如 videos/baby_crawl_towards/baby_crawl_towards-026.mp4
        # 就用第二级目录作为类别
        video_path = line.split()[0]  # 第一个是视频路径
        parts = video_path.split("/")
        if len(parts) >= 2:
            category = parts[1]   # e.g. baby_crawl_towards
        else:
            category = "unknown"

        category_lines[category].append(line)

# 将结果写入到各个类别文件
for category, lines in category_lines.items():
    save_path = os.path.join(root_dir, f"{category}.txt")
    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"保存 {save_path}，共 {len(lines)} 行")
