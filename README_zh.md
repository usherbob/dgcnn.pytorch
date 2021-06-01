# 3D Pool
本仓库是硕士毕业论文《面向3D点云的深层神经网络池化研究》代码. 代码的主体框架借鉴自 [Antao97/dgcnn.pytorch](https://github.com/WangYueFt/dgcnn/tree/master/pytorch).

&nbsp;论文的主体架构
<p float="left">
    <img src="image/frame.svg"/>
</p>

&nbsp;

## Requirements
- Python 3.7
- PyTorch 1.2
- CUDA 10.0
- Package: glob, h5py, sklearn

&nbsp;
## Contents
- [Point Cloud Classification](#point-cloud-classification)
- [Point Cloud Part Segmentation](#point-cloud-part-segmentation)
- [Point Cloud Semantic Segmentation](#point-cloud-sementic-segmentation)

&nbsp;
## Global Description Guided Pooling 
### Classification

#### ModelNet40

- train

``` 
python main_cls.py --exp_name=GDP_M40 --model pointnet/dgcnn --base_dir /path/to/data --pool GDP --cd_weights 0.01
```

- eval

``` 
python main_cls.py --eval True --exp_name=GDP_M40.eval --model pointnet/dgcnn --base_dir /path/to/data --pool GDP --cd_weights 0.01 --model_path /path/to/model
```

#### ScanObjectNN

- train

```
python main_scan.py --exp_name=GDP_scan --base_dir /path/to/data --pool GDP --cd_weights 0.01
```

- eval

```
python main_scan.py --eval True --exp_name=GDP_scan.eval --base_dir /path/to/data --pool GDP --cd_weights 0.01 --model_path /path/to/model
```

### Segmentation

#### ShapeNetPart

- train

```
python main_part.py --exp_name=GDP_part --base_dir /path/to/data --pool GDP --cd_weights 0.01
```

- eval

```
python main_part.py --eval True --exp_name=GDP_part.eval --base_dir /path/to/data --pool GDP --cd_weights 0.01 --model_path /path/to/model
```

#### S3DIS

You have to download `Stanford3dDataset_v1.2_Aligned_Version.zip` manually from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 .

- train

```
python main_sem.py --exp_name=GDP_sem --base_dir /path/to/data --pool GDP --cd_weights 0.01
```

- eval

```
python main_sem.py --eval True --exp_name=GDP_sem.eval --base_dir /path/to/data --pool GDP --cd_weights 0.01 --model_path /path/to/model
```

## Random Pooling 

### Classification

#### ModelNet40

- train

```
python main_cls.py --exp_name=RDP_M40 --model pointnet/dgcnn --base_dir /path/to/data --pool RDP 
```

- eval

```
python main_cls.py --eval True --exp_name=RDP_M40.eval --model pointnet/dgcnn --base_dir /path/to/data --pool RDP --model_path /path/to/model
```

#### ScanObjectNN

- train

```
python main_scan.py --exp_name=RDP_scan --base_dir /path/to/data --pool RDP 
```

- eval

```
python main_scan.py --eval True --exp_name=RDP_scan.eval --base_dir /path/to/data --pool RDP --model_path /path/to/model
```

### Segmentation

#### ShapeNetPart

- train

```
python main_part.py --exp_name=RDP_part --base_dir /path/to/data --pool RDP 
```

- eval

```
python main_part.py --eval True --exp_name=RDP_part.eval --base_dir /path/to/data --pool RDP --model_path /path/to/model
```

#### S3DIS

- train

```
python main_sem.py --exp_name=RDP_sem --base_dir /path/to/data --pool RDP
```

- eval

```
python main_sem.py --eval True --exp_name=RDP_sem.eval --base_dir /path/to/data --pool RDP --model_path /path/to/model
```

## Mutual Infomax Pooling 

### Classification

#### ModelNet40

- train

```
python main_cls.py --exp_name=MIP_M40 --model pointnet/dgcnn --base_dir /path/to/data --pool MIP --mi_weights 1 
```

- eval

```
python main_cls.py --eval True --exp_name=MIP_M40.eval --model pointnet/dgcnn --base_dir /path/to/data --pool MIP --model_path /path/to/model --mi_weights 1
```

#### ScanObjectNN

- train

```
python main_scan.py --exp_name=MIP_scan --base_dir /path/to/data --pool MIP --mi_weights 1 
```

- eval

```
python main_scan.py --eval True --exp_name=MIP_scan.eval --base_dir /path/to/data --pool MIP --mi_weights 1 --model_path /path/to/model
```

### Segmentation

#### ShapeNetPart

- train

```
python main_part.py --exp_name=MIP_part --base_dir /path/to/data --pool MIP --mi_weights 1
```

- eval

```
python main_part.py --eval True --exp_name=MIP_part.eval --base_dir /path/to/data --pool MIP --mi_weights 1 --model_path /path/to/model
```

#### S3DIS

- train

```
python main_sem.py --exp_name=MIP_sem --base_dir /path/to/data --pool MIP --mi_weights 1
```

- eval

```
python main_sem.py --eval True --exp_name=MIP_sem.eval --base_dir /path/to/data --pool MIP --mi_weights 1 --model_path /path/to/model
```



