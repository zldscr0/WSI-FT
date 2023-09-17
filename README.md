## README

[TOC]

TO-DO（Last update：20230917）

- [x] 环境配置
- [x] 数据集准备
- [x] 代码复现
- [ ] 整理代码结构（部分阶段未整理）
- [ ] 补充部分结果
- [ ] 绘图

---

#### Introduction

Paper：https://openaccess.thecvf.com/content/CVPR2023/html/Li_Task-Specific_Fine-Tuning_via_Variational_Information_Bottleneck_for_Weakly-Supervised_Pathology_Whole_CVPR_2023_paper.html

Code：https://github.com/invoker-LL/WSI-finetuning

```
@InProceedings{Li_2023_CVPR,
    author    = {Li, Honglin and Zhu, Chenglu and Zhang, Yunlong and Sun, Yuxuan and Shui, Zhongyi and Kuang, Wenwei and Zheng, Sunyi and Yang, Lin},
    title     = {Task-Specific Fine-Tuning via Variational Information Bottleneck for Weakly-Supervised Pathology Whole Slide Image Classification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {7454-7463}
}
```



```
Project
    checkpoints # 文件夹，放置最优模型，或者写.sh文件说明最优模型的下载地址
    common      # 文件夹，放置utilizes.py等文件
    datasets    # 文件夹，放置数据集的读取方式，写清楚各个数据集的.py文件
    models      # 文件夹，放置网络模型.py文件
    scripts     # 文件夹，放置train.sh, test.sh脚本文件
    README.md  
    environment.yml
    train.py
    test.py
```

---

#### 数据集准备

##### 本地数据集说明

数据集使用的是省肿瘤项目当中的病理数据集(.svs文件)。（也可用公开的camyon16数据集）

.svs图像存储位置：

```
/data/Colon/svs
```

切割并提取特征后的存储位置：

```
data/data_seg_patch  
data/feat_dir
```

> 注：实际处理时切割并提取特征后的存储位置存在了整个code的上一级目录中。

##### 数据集预处理

Mostly folked from [CLAM](https://github.com/mahmoodlab/CLAM), with minor modification. So just follow the [docs](https://github.com/mahmoodlab/CLAM/tree/master/docs) to perform baseline, or with following steps:

1.Preparing grid patches of WSI without overlap.

```
bash create_patches.sh
```

2.Preparing pretrained patch features fow WSI head training.

```
bash create_feature.sh
```

此处原始代码仓直接搬用CLAM的预处理病理图像和提取特征的代码（写了两个脚本），但并没有指明要修改哪些参数，可参照http://gitlab.nju.rl/nju/szl_pathology中的README.md文件准备数据集并修改超参数。

针对省肿瘤病理数据集，有备份的已经过CLAM提取特征后的图像（存在nju云盘中，13.4G），可直接下载用来处理。

原始代码仓还漏写了划分数据集的步骤，本代码中补充了这一步骤：

```bash
python3 create_splits.py
```



```bash
CUDA_VISIBLE_DEVICES=2 python3 extract_topK_ROIs.py --data_h5_dir ../data_seg_patch --csv_path ../data_seg_patch/process_list_autogen.csv --feat_dir ../feat_dir --patch_dir ../data_seg_patch --data_slide_dir data --batch_size 512 --slide_ext .svs
```

---

#### 环境配置

CUDA:11.6

创建新环境

```
conda create --name p39 python=3.9
```

激活

```
conda activate p39
```

安装依赖项

```
pip install -r ./requirments.txt
```

下载GPU版本的pytorch(CUDA版本=11.6)，2G左右，较快。

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
```

验证torch是否安装成功

```python
import torch
print(torch.__version__)
print(torch.version.cuda)
```

如果遇到`topk`的`module found error`报错：

```bash
git clone https://github.com/oval-group/smooth-topk.git
cd smooth-topk
python setup.py install#此处报错则替换为 pip install .
```

base中装载了2.0.1的cpu版本的pytorch，new_env中安装GPU版pytorch成功

![image-20230916183128012](C:\Users\hanabi\AppData\Roaming\Typora\typora-user-images\image-20230916183128012.png)

注：环境里用`pip`安装，如果安装到了外面的`python`里，在`pip`指令前加一个`python -m`

```python
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
```

---

#### train

##### Stage1

```
bash scripts/train.sh
```

训练过程中截图：

![image-20230916182958798](C:\Users\hanabi\AppData\Roaming\Typora\typora-user-images\image-20230916182958798.png)

最优模型：

训练集上的准确率达到了97.62%

```
Epoch: 47, train_loss: 0.2090, train_clustering_loss:  0.0592, train_error: 0.0238
```

##### Stage-1b (variational IB training):

```
bash scripts/vib_train.sh
```

##### Stage-2 (wsi-finetuning with topK):

1. Collecting top-k patches of WSI by inference vib model, save in pt form.

```
bash scripts/extract_topk_rois.sh
```



1. Perform end-to-end training.

```
bash scripts/e2e_train.sh
```

##### Stage-3 (training wsi head with fine-tuned patch backbone):

Now you can use finetuned patch bakcbone in stage-2 to generate patch features, then run stage-1 again with the new features.

---

#### test

##### stage-1

同CLAM框架的结果格式，此图显示了5折交叉的测试结果

![image-20230917103319888](C:\Users\hanabi\AppData\Roaming\Typora\typora-user-images\image-20230917103319888.png)

验证集和测试集的最高准确率达到了**1**和**0.733**。

---

 Contribution: Code completed by Zhixin Bai.