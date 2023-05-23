import argparse


def parser_args():
    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument("--seed", type=int, default=42)

    # 1. data configuration
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument('--test_ratio', type=float, default=0.3)

    # 2. model configuration
    parser.add_argument("--model_name", type=str, default='')

    # 3. train and test
    # 3.1 Train & Evaluation & Test Implementation Details
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--optimizer_name", type=str, default='SGD')
    parser.add_argument('--loss_fcn_name', type=str, default='MSELoss')
    parser.add_argument('--scheduler_name', type=str, default='StepLR')
    # 3.2 Devices & Distributed Device
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument('--comment', type=str, default='default comment')
    args = parser.parse_args()
    return args
