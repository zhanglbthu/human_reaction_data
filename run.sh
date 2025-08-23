# gradual mode
python inference.py \
    --video_path '/root/autodl-tmp/VIMO/baby_crawl_towards/baby_crawl_towards-012/video.mp4' \
    --stride 2 \
    --out_dir experiments \
    --radius_scale 1 \
    --camera 'traj' \
    --mode 'gradual' \
    --mask \
    --traj_txt '/root/autodl-tmp/VIMO/baby_crawl_towards/baby_crawl_towards-012/traj.txt'