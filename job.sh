CUDA_VISIBLE_DEVICES=4 python main_cls.py --num_points 2048 --base_dir /data/sunzhiyi/pc --exp_name pn.mi4.res.v7.2048
wait
CUDA_VISIBLE_DEVICES=4 python main_cls.py --num_points 2048 --model dgcnn --base_dir /data/sunzhiyi/pc --exp_name ec.mi4.res.v7.2048
