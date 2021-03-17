CUDA_VISIBLE_DEVICES=0 python main_semseg.py --base_dir /data/sunzhiyi/pc --test_area 6 --exp_name res0317_6
wait
CUDA_VISIBLE_DEVICES=0 python main_semseg.py --base_dir /data/sunzhiyi/pc --test_area 5 --exp_name res0317_5
wait
CUDA_VISIBLE_DEVICES=0 python main_semseg.py --base_dir /data/sunzhiyi/pc --test_area 4 --exp_name res0317_4
wait
CUDA_VISIBLE_DEVICES=0 python main_semseg.py --base_dir /data/sunzhiyi/pc --test_area 3 --exp_name res0317_3
wait
CUDA_VISIBLE_DEVICES=0 python main_semseg.py --base_dir /data/sunzhiyi/pc --test_area 2 --exp_name res0317_2
wait
CUDA_VISIBLE_DEVICES=0 python main_semseg.py --base_dir /data/sunzhiyi/pc --test_area 1 --exp_name res0317_1
