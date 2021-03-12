CUDA_VISIBLE_DEVICES=6 python main_semseg.py --base_dir /data/sunzhiyi/pc --test_area 6 --exp_name 3levels_6
wait
CUDA_VISIBLE_DEVICES=6 python main_semseg.py --base_dir /data/sunzhiyi/pc --test_area 5 --exp_name 3levels_5
wait
CUDA_VISIBLE_DEVICES=6 python main_semseg.py --base_dir /data/sunzhiyi/pc --test_area 4 --exp_name 3levels_4
wait
CUDA_VISIBLE_DEVICES=6 python main_semseg.py --base_dir /data/sunzhiyi/pc --test_area 3 --exp_name 3levels_3
wait
CUDA_VISIBLE_DEVICES=6 python main_semseg.py --base_dir /data/sunzhiyi/pc --test_area 2 --exp_name 3levels_2
wait
CUDA_VISIBLE_DEVICES=6 python main_semseg.py --base_dir /data/sunzhiyi/pc --test_area 1 --exp_name 3levels_1
wait

