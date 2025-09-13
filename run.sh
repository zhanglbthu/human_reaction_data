# # gradual mode
# python inference.py \
#     --video_path '/root/autodl-tmp/VIMO/baby_crawl_towards/baby_crawl_towards-012/video.mp4' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'traj' \
#     --mode 'gradual' \
#     --mask \
#     --traj_txt '/root/autodl-tmp/VIMO/baby_crawl_towards/baby_crawl_towards-012/traj.txt'

#!/bin/bash

# root_dir="/root/autodl-tmp/VIMO/processed/automobile_rush_towards"
# out_dir="/root/autodl-tmp/VIMO/cam_align/automobile_rush_towards"
category="automobile_rush_towards"
root_dir="/root/autodl-tmp/VIMO/processed/$category"
out_dir="/root/autodl-tmp/VIMO/cam_align/$category"

for sub_dir in "$root_dir"/*; do
    if [ -d "$sub_dir" ]; then
        video_path="$sub_dir/video.mp4"
        traj_txt="$sub_dir/traj.txt"

        # 提取子目录名
        sub_name=$(basename "$sub_dir")
        exp_out_dir="$out_dir/$sub_name"

        if [ -f "$video_path" ] && [ -f "$traj_txt" ]; then
            echo "Processing: $sub_dir -> $exp_out_dir"

            python inference.py \
                --video_path "$video_path" \
                --stride 2 \
                --out_dir "$exp_out_dir" \
                --radius_scale 1 \
                --camera 'traj' \
                --mode 'gradual' \
                --mask \
                --traj_txt "$traj_txt"
        else
            echo "Missing video.mp4 or traj.txt in $sub_dir, skipped."
        fi
    fi
done
