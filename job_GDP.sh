##train
#CUDA_VISIBLE_DEVICES=0 python main_cls.py --model pointnet --pool GDP --exp_name pn.gdp.cd-2.sam384 --cd_weights 1e-2 --num_agg 10 --num_sample 384 --base_dir /data4/jiahuawang
## inference
CUDA_VISIBLE_DEVICES=0 python main_cls.py --model pointnet --pool GDP --exp_name pn.gdp.cd-2.sam384.eval --cd_weights 1e-2 --num_agg 10 --num_sample 384 --base_dir /data4/jiahuawang --eval True --model_path /data4/jiahuawang/ckpt/cls/pn.gdp.cd-2.sam384/models/model.t7
