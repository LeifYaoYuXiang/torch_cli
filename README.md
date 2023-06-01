# 0. Description 项目介绍
（在本章节中介绍该项目的主要内容）

# 1. Environment 运行环境
```shell
pip install numpy
pip install scikit-learn
pip install tensorboard
```
针对 PyTorch, 实验人员可以选择自己合适的版本加以下载。具体的下载地址为：https://pytorch.org/

# 2. Structure 项目结构
```text
.
├── README.md
├── argument_parser.py：指定模型、数据集、训练过程的详细参数
├── dataloader.py：数据加载器
├── dataset.py：数据集，负责数据集的预处理与加载等工作
├── inference.py：针对训练完成的模型进行调用
├── main.py：主函数
├── metrics.py：模型评价指标
├── model.py：模型定义
├── model_configuration：预定义的模型参数
├── optimizer_loss_func_scheduler.py：生成优化器、损失函数、Scheduler
├── runs：记录模型训练结果
│   └── README.md
├── server_run_shell.sh：用于命令行式的调用main函数
├── test：用于整个项目的单元测试
│   ├── __init__.py
│   ├── test_dataloader.py
│   ├── test_dataset.py
│   └── test_model.py
├── train_eval_test.py：模型的训练、验证、测试过程
└── utils.py：辅助函数
```

# 3. Result 实验结果
