from create_splits_seq import create_splits_main
import yaml
# Rest of your imports and functions remain unchanged
class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
with open('create_splits_config.yaml', 'r') as file:
    args_for_CS = Config(yaml.safe_load(file))
create_splits_main(args_for_CS)
print("hello")
