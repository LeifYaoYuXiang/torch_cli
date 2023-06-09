import torch


def build_jit_model(model, example, save_filepath):
    scripted_model = torch.jit.script(model, example)
    scripted_model.save(save_filepath)
