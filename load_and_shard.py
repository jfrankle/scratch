import sys
import os
import torch
import warnings

from composer import Trainer
from composer.core import Evaluator
from omegaconf import OmegaConf as om
from omegaconf import DictConfig
from llmfoundry import COMPOSER_MODEL_REGISTRY
from composer.core import Event

from composer.utils import dist, get_device, reproducibility
from composer.callbacks import (HealthChecker, LRMonitor, MemoryMonitor,
                                OptimizerMonitor, RuntimeEstimator,
                                SpeedMonitor)

from llmfoundry.utils.builders import (build_icl_evaluators, build_tokenizer,
                                       build_logger, build_optimizer)
from llmfoundry.data.text_data import build_text_dataloader

from sharded_checkpointing_callback import ShardedCheckpointSaver

from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

def build_composer_model(model_cfg, tokenizer):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    if model_cfg.name not in COMPOSER_MODEL_REGISTRY:
        raise ValueError(
            f'Not sure how to build model with name={model_cfg.name}')
    return COMPOSER_MODEL_REGISTRY[model_cfg.name](model_cfg, tokenizer)

def build_dataloader(cfg, tokenizer, device_batch_size):
    if cfg.name == 'text':
        return build_text_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == 'text_denoising':
        return build_text_denoising_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == 'finetuning':
        return build_finetuning_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )

    else:
        raise ValueError(f'Not sure how to build dataloader with config: {cfg}')

def main(cfg):
    reproducibility.seed_all(cfg.seed)

    cfg.dist_timeout = cfg.get('dist_timeout', 600.0)

    dist.initialize_dist(get_device(None), timeout=cfg.dist_timeout)


    og_state_dict = torch.load(cfg.load_path)

    del og_state_dict['state']['model']
    optimizer_name = list(og_state_dict['state']['optimizers'].keys())[0]
    print ("optimizer name is: ", optimizer_name)
    del og_state_dict['state']['optimizers']

    for key in og_state_dict['state'].keys():
        print ("key is: ", key)
    
    tokenizer = build_tokenizer(cfg.tokenizer)

    model_cfg = cfg.model
    model = build_composer_model(model_cfg, tokenizer)

    # Read FSDP Config as a dict
    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(fsdp_config,
                                  resolve=True) if fsdp_config else None

    # Restrict model init_device to 'meta' and 'cpu',
    # using 'cuda' vs. 'cuda:id' is tricky and can lead to common user errors
    # when multiple GPUs are available.
    # Also 'meta' is only valid when using FSDP
    init_device = cfg.model.get('init_device', 'cpu')
    assert init_device in ['meta', 'cpu']
    if fsdp_config is None and init_device == 'meta':
        warnings.warn(
            "Using `cfg.model.init_device='meta'` is only valid when using FSDP! " +\
            "Reverting to `cfg.model.init_device='cpu'`.")
        cfg.model.init_device = 'cpu'

    # Build Model
    print('Initializing model...')
    model = build_composer_model(cfg.model, cfg.tokenizer)
    cfg.n_params = sum(p.numel() for p in model.parameters())
    print(f'{cfg.n_params=:.2e}')

    # Dataloaders
    evaluators = []
    if 'eval_loader' in cfg:
        eval_loader = Evaluator(label='eval',
                                dataloader=build_dataloader(,
                                    cfg,
                                    cfg.eval_loader,
                                    cfg.device_eval_batch_size),
                                metric_names=list(model.train_metrics.keys()))
        evaluators.append(eval_loader)

    if 'icl_tasks' in cfg:
        icl_evaluators, _ = build_icl_evaluators(cfg.icl_tasks, tokenizer,
                                                 cfg.max_seq_len,
                                                 cfg.device_eval_batch_size)        
        evaluators.extend(icl_evaluators)

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Loggers
    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in (cfg.get('loggers') or {}).items()
    ]

    # Callbacks
    callbacks = [
        ShardedCheckpointSaver(**cfg.callbacks.sharded_ckpt_saver)
    ]

    # Build the Trainer
    print('Building trainer...')
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        optimizers=optimizer,
        progress_bar=cfg.get('progress_bar', False),
        log_to_console=cfg.get('log_to_console', True),
        console_log_interval=cfg.get('console_log_interval', '1ba'),
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        # algorithms=algorithms,
        fsdp_config=fsdp_config,  # type: ignore
        save_overwrite=cfg.get('save_overwrite', False),
        load_path=cfg.get('load_path', None),
        dist_timeout=cfg.dist_timeout,
    )

    for callback in callbacks:
        if isinstance(callback, ShardedCheckpointSaver):
            print ("trying to save")
            callback._save_checkpoint(trainer.state, og_state_dict, optimizer_name)
            print ("saved checkpoints")

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
