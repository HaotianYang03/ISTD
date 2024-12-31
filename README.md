# It's an assignment for ISTD

## 1. Dateset

Under the repo root, create Dataset subdirectory and order your dataset like this:

```plaintext
Dataset
    └─数据集
        ├─测试集
        │  ├─MDvsFA
        │  │  ├─image
        │  │  └─mask
        │  │    
        │  └─SIRST
        │      ├─image
        │      └─mask
        └─训练集
            ├─image
            └─mask

```
百度网盘: https://pan.baidu.com/s/1HQX9FjJwRHZY5xx0z2DZRA
提取码：7ey5

## 2. Environment Configuration

```plaintext
$ conda env create -f environment.yaml
```

```plaintext
$ pip install wandb
```

```plaintext
$ wandb login
```

## 2. Train

```plaintext
$ python train.py
```

