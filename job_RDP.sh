#CUDA_VISIBLE_DEVICES=0 python main_cls.py --model pointnet --pool RDP --exp_name pn.rdp --base_dir /data4/jiahuawang
CUDA_VISIBLE_DEVICES=0 python main_cls.py --model pointnet --pool RDP --exp_name pn.rdp.agg20.sam512.eval --num_agg 20 --num_sample 512 --base_dir /data4/jiahuawang --eval True --model_path /data4/jiahuawang/ckpt/cls/pn.rdp.agg20.sam512/models/model.t7
