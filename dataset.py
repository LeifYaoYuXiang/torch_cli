from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


def build_dataset(dataset_config):
    return