from torch.utils.data import DataLoader
from dataset import build_dataset


def build_dataloader(dataset_config):
    train_dataset, eval_dataset, test_dataset = build_dataset(dataset_config)
    batch_size: int = dataset_config['batch_size']
    shuffle: bool = dataset_config['shuffle']
    num_workers: int = dataset_config['num_workers']
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    if test_dataset is not None:
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return train_dataloader, eval_dataloader, test_dataloader
    else:
        return train_dataloader, eval_dataloader, None

