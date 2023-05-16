# 本文用于生成常见的优化函数 与 损失函数
import torch
import torch.optim as optim
# 实现的优化器
IMPLEMENTED_OPTIMIZER = []
# 实现的损失函数
IMPLEMENTED_LOSS_FCN = []
# 实现的


# 生成优化器
def build_optimizer(train_test_config, model):
    optimizer_name = train_test_config['optimizer_name']
    optimizer_network2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005, betas=(0.5, 0.999))
    pass


# 生成损失函数
def build_loss_function(train_test_config):
    loss_fcn_name = train_test_config['loss_fcn_name']
    pass


# 生成scheduler
def build_scheduler(train_test_config):
    scheduler_name = train_test_config['scheduler_name']
    pass
