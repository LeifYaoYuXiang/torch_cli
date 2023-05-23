# 本文用于生成常见的优化函数 与 损失函数
import torch
IMPLEMENTED_OPTIMIZER = ['SGD', 'ASGD', 'Rprop', 'Adagrad', 'Adadelta', 'RMSprop', 'Adam', 'Adamax', 'SparseAdam', 'LBFGS']
IMPLEMENTED_LOSS_FCN = ['MSELoss', 'L1Loss', 'CrossEntropyLoss', 'NLLLoss', 'NLLLoss2d', 'KLDivLoss', 'BCELoss',
                        'MarginRankingLoss', 'MultiLabelMarginLoss', 'SmoothL1Loss', 'SoftMarginLoss']
IMPLEMENTED_SCHEDULER = ['StepLR']


# 生成优化器
def build_optimizer(train_test_config, model):
    optimizer_name = train_test_config['optimizer_name']
    lr = train_test_config['lr']
    assert optimizer_name in IMPLEMENTED_OPTIMIZER
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif optimizer_name == 'ASGD':
        optimizer = torch.optim.ASGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif optimizer_name == 'Rprop':
        optimizer = torch.optim.Rprop(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif optimizer_name == 'Adadelta':
        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif optimizer_name == 'Adamax':
        optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif optimizer_name == 'SparseAdam':
        optimizer = torch.optim.SparseAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif optimizer_name == 'LBFGS':
        optimizer = torch.optim.LBFGS(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        raise NotImplementedError
    return optimizer


# 生成损失函数
def build_loss_function(train_test_config):
    loss_fcn_name = train_test_config['loss_fcn_name']
    assert loss_fcn_name in IMPLEMENTED_LOSS_FCN
    if loss_fcn_name == 'MSELoss':
        loss_fcn = torch.nn.MSELoss()
    elif loss_fcn_name == 'L1Loss':
        loss_fcn = torch.nn.L1Loss()
    elif loss_fcn_name == 'CrossEntropyLoss':
        loss_fcn = torch.nn.CrossEntropyLoss()
    elif loss_fcn_name == 'NLLLoss':
        loss_fcn = torch.nn.NLLLoss()
    elif loss_fcn_name == 'NLLLoss2d':
        loss_fcn = torch.nn.NLLLoss2d()
    elif loss_fcn_name == 'KLDivLoss':
        loss_fcn = torch.nn.KLDivLoss()
    elif loss_fcn_name == 'BCELoss':
        loss_fcn = torch.nn.BCELoss()
    elif loss_fcn_name == 'MarginRankingLoss':
        loss_fcn = torch.nn.MarginRankingLoss()
    elif loss_fcn_name == 'MultiLabelMarginLoss':
        loss_fcn = torch.nn.MultiLabelMarginLoss()
    else:
        raise NotImplementedError
    return loss_fcn


# 生成scheduler
def build_scheduler(train_test_config, optimizer):
    scheduler_name = train_test_config['scheduler_name']
    assert scheduler_name in IMPLEMENTED_SCHEDULER
    step_size = train_test_config['step_size']
    gamma = train_test_config['gamma']
    if scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma, last_epoch=-1)
    else:
        raise NotImplementedError
    return scheduler
