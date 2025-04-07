import os
import json
import deepchem as dc
import importlib

def save_pretrained(model, model_dir, config):
    os.makedirs(model_dir, exist_ok=True)
    model.save_checkpoint(model_dir)

    config['model_class'] = model.__class__.__name__
    config['deepchem_version'] = dc.__version__

    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

def load_from_pretrained(model_dir):
    config_path = os.path.join(model_dir, 'config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError("Missing config.json.")

    with open(config_path, 'r') as f:
        config = json.load(f)

    model_class_name = config.pop('model_class')
    model_module = importlib.import_module('deepchem.models')
    model_class = getattr(model_module, model_class_name)

    model = model_class(**config)
    model.restore(model_dir)
    return model