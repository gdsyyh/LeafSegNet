## README

环境：本项目在autodl云服务器上可以直接使用

项目结构：

```
.
├── NewData
├── NewDatalog
├── README.md
├── __pycache__
├── convertSegToAim.py
├── dataset
├── log
├── loss.py
├── metrics.py
├── model
├── results
├── split_trainandtest.py
├── test.py
├── train.py
├── train_aux.py
├── content
├── Leaf Only SAM.ipynb
└── utils.py
```

其中，`dataset/` ` log/` ` NewData/` ` NewDatalog/` 四个文件夹未上传至github，里面分别放自建数据集、自建数据集实验结果、公开数据集、公开数据集实验结果



`content/`中的是所有实验的结果数据对应的excel表



`Leaf Only SAM.ipynb`是使用sam做自监督数据集构建的脚本。使用时需要创建一个安装了segment anything的环境，autodl上有预下载好的镜像，可以直接使用。





