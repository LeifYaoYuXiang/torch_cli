import torch
import torch.nn as nn
from utils import classification_softmax


def train_eval_test_v1(model, optimizer, loss_function, scheduler,
                    train_loader, eval_dataloader, test_loader, summary_writer, train_test_config):
    n_epoch = train_test_config['n_epoch']
    device = train_test_config['device']

    # 训练与验证阶段
    for each_epoch in range(n_epoch):
        # 训练阶段
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            # 以下为训练范式
            x_batch, y_batch = batch[0], batch[1]
            # 将数据移动到设备上
            outputs = model(x_batch)
            optimizer.zero_grad()
            loss = loss_function(outputs, y_batch)
            loss.backward()
            optimizer.step()
            pass
        if scheduler is not None:
            scheduler.step()

        # 验证阶段
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                # 以下为验证范式
                x_batch, y_batch = batch[0], batch[1]
                # 将数据移动到设备上
                outputs = model(x_batch)
                # 针对分类问题
                y_pred = classification_softmax(outputs)
                # 针对回归问题
                y_pred = outputs
        # Add metrics into SummaryWriter
        summary_writer.add_scaler()

     # 测试阶段
    if test_loader is not None:
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                # 测试范式
                x_batch, y_batch = batch[0], batch[1]
                # # Move Data to Device
                outputs = model(x_batch)
                y_pred = classification_softmax(outputs)

    return model



