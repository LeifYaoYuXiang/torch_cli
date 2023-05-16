import torch
from argument_parser import parser_args
from dataloader import build_dataloader
from model import build_model
from optimizer_loss_func_scheduler import build_loss_function, build_optimizer, build_scheduler
from train_eval_test import train_eval_test_v1
from utils import seed_setting, get_summary_writer, record_configuration


def main(args):
    # 设置实验的随机性
    seed = args.seed
    seed_setting(seed)

    # 设置显卡
    gpu_index = args.gpu_index
    device = args.device
    if device == 'gpu':
        device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    # 设置
    log_filepath = args.log_filepath
    dataset_config = {
        'dataset_name': args.dataset_name,
        'test_ratio': args.test_ratio,
        'seed': seed,
        'batch_size': args.batch_size,
    }
    model_config = {
        'model_name': args.model_name,
        'device': device,
    }
    train_test_config = {
        'n_epoch': args.n_epoch,
        'device': device,
        'batch_size': args.batch_size,
        'comment': args.comment,
    }
    summary_writer, log_dir = get_summary_writer(log_filepath)
    record_configuration(save_dir=log_dir, configuration_dict={
        'DATASET': dataset_config,
        'MODEL': model_config,
        'TRAIN': train_test_config,
    })
    print(model_config, '\n', dataset_config, '\n', train_test_config)

    model = build_model(model_config)
    print('model build finish')

    train_dataloader, eval_dataloader, test_dataloader = build_dataloader(dataset_config)
    print('dataloader build finish')

    loss_fcn = build_loss_function(train_test_config)
    optimizer = build_optimizer(train_test_config, model)
    scheduler = build_scheduler(train_test_config)
    print('loss_fcn optimizer scheduler build finish')

    model = train_eval_test_v1(model, optimizer, loss_fcn, scheduler, train_dataloader, eval_dataloader, test_dataloader,
                               summary_writer, train_test_config)
    return model

# 启动函数
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    args = parser_args()
    model = main(args)
