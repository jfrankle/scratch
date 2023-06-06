# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import sys
import warnings
import torch
import composer

from omegaconf import OmegaConf as om

def main(cfg):

    print ("going to download: ", cfg.remote_path)
    composer.utils.get_file(cfg.remote_path, cfg.mono_checkpoint_path)
    print ("finished downloading")
    state_dict = torch.load(cfg.mono_checkpoint_path)

    del state_dict['state']['model']
    if 'optimizers' in state_dict['state']:
        for optimizer in state_dict['state']['optimizers']:
            state_dict['state']['optimizers'][optimizer] = {}
    
    torch.save(state_dict, cfg.raw_state_dict)

    print ("saved just the state to: ", cfg.raw_state_dict)


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
