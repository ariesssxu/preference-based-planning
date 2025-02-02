import pathlib

import yaml

SURVIVAL_LOG = False

root = pathlib.Path(__file__).parent
for key, value in yaml.safe_load((root / 'rearrangement_config.yaml').read_text()).items():
    globals()[key] = value

for key, value in yaml.safe_load((root / 'preference_config.yaml').read_text()).items():
    globals()[key] = value

# for key, value in yaml.safe_load((root / 'sequence_config.yaml').read_text()).items():
#     globals()[key] = value

for key, value in yaml.safe_load((root / 'general_config.yaml').read_text()).items():
    globals()[key] = value