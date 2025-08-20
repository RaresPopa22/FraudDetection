def is_tree_model(config):
    tree_models = config['models']['tree']
    model_name = config['model_name']

    return model_name in tree_models

def is_sequential_model(config):
    sequential_models = config['models']['sequential']
    model_name = config['model_name']

    return model_name in sequential_models