from utils import load_model_configuration

IMPLEMENTED_MODEL = []


def build_model(model_config):
    model_name = model_config['model_name']
    assert model_name in IMPLEMENTED_MODEL

    # load pre-stored model parameters: 加载预存好的模型参数
    if 'model_config_filepath' in model_config.keys() and 'model_section' in model_config.keys():
        loaded_model_config = load_model_configuration(model_config['model_config_filepath'],
                                                       model_config['model_section'])
        model_config.update(loaded_model_config)

    # build model: 生成模型
    if model_name == '':
        pass
    else:
        raise NotImplementedError
