import torch
import warnings
import sys
from omegaconf import OmegaConf as om

from composer.trainer.dist_strategy import prepare_fsdp_module
from composer.utils import dist, get_device
from composer.core import Precision
from composer.core.state import fsdp_state_dict_type_context

from llmfoundry import COMPOSER_MODEL_REGISTRY
from llmfoundry.utils.builders import build_optimizer
from torch.distributed.fsdp import FullyShardedDataParallel

def build_composer_model(model_cfg, tokenizer):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    if model_cfg.name not in COMPOSER_MODEL_REGISTRY:
        raise ValueError(
            f'Not sure how to build model with name={model_cfg.name}')
    return COMPOSER_MODEL_REGISTRY[model_cfg.name](model_cfg, tokenizer)

def load_optimizer_checkpoint(model, cfg, state_dict, state_optimizer_name):

    full_optim_state_dict = None
    if dist.get_local_rank() == 0:
        print ("loading optimizer state dict")
        full_optim_state_dict = state_dict['state']['optimizers'][state_optimizer_name]
        print ("loaded optimizer state dict")

    print ("before dist barrier")
    dist.barrier()
    print ("after dist barrier")

    # This really just makes rank 0 scatter everything
    return FullyShardedDataParallel.scatter_full_optim_state_dict(full_optim_state_dict, model)

def main(cfg):
    device = get_device(None)

    dist.initialize_dist(device, timeout=60)

    model_cfg = cfg.model
    if dist.get_local_rank() == 0:
        model_cfg.init_device = 'cpu'

    model = build_composer_model(model_cfg, cfg.tokenizer)

    # Just the state dict itself
    raw_state_dict = torch.load(cfg.raw_state_dict)
    state_optimizer_name = list(raw_state_dict['state']['optimizers'].keys())[0]

    state_dict = None
    if dist.get_local_rank() == 0:
        print ("loaded checkpoint")
        state_dict = torch.load(cfg.mono_checkpoint_path)
        model.load_state_dict(state_dict['state']['model'])
        print ("lod")

    precision = Precision('amp_bf16')

    # Read FSDP Config as a dict
    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(fsdp_config,
                                  resolve=True) if fsdp_config else None

    # This otherwise breaks w/ torch 2.0
    fsdp_config['use_orig_params'] = False
    fsdp_config['state_dict_type'] = 'full'
    fsdp_config['sync_module_states'] = True
    fsdp_config['mixed_precision'] = 'FULL'

    prepare_fsdp_module(model, [], fsdp_config, precision=precision, device=device, auto_microbatching=False)

    print ("before building optimizer")
    optimizer = build_optimizer(cfg.optimizer, model)
    print ("after loading optimizer")
    optimizer_state_dict = load_optimizer_checkpoint(model, cfg, state_dict, state_optimizer_name)
    print ("after got optimizer state dict")
    optimizer.load_state_dict(optimizer_state_dict)
    print ("after loaded optimizer state dict")

    with fsdp_state_dict_type_context(model, state_dict_type='sharded'):
        raw_state_dict['state']['model'] = model.state_dict()

        # This works with torch 2.0...
        raw_state_dict['state']['optimizers'][state_optimizer_name] = FullyShardedDataParallel.optim_state_dict(model, optimizer)

    torch.save(raw_state_dict, f"temp/rank{dist.get_global_rank()}.pt")

    print ("Finished saving resharded checkpoint")

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)