import sys
import os
import torch
import warnings

from composer import Trainer
from composer.core import Evaluator
from omegaconf import OmegaConf as om
from omegaconf import DictConfig
from examples.llm.src import COMPOSER_MODEL_REGISTRY
from composer.core import Event

from composer.utils import dist, get_device, reproducibility
from composer.callbacks import (HealthChecker, LRMonitor, MemoryMonitor,
                                OptimizerMonitor, RuntimeEstimator,
                                SpeedMonitor)

from examples.common.builders import (build_algorithm, build_callback,
                                      build_icl_evaluators, build_logger,
                                      build_optimizer, build_scheduler)

from examples.common.config_utils import log_config, update_batch_size_info

from sharded_checkpointing_callback import ShardedCheckpointSaver
from examples.common.text_data import build_text_dataloader

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

def build_dataloader(cfg, device_batch_size):
    if cfg.name == 'text':
        return build_text_dataloader(cfg, device_batch_size)
    elif cfg.name == 'text_denoising':
        return build_text_denoising_dataloader(cfg, device_batch_size)
    else:
        raise ValueError(f'Not sure how to build dataloader with config: {cfg}')

def build_callback(name, kwargs):
    if name == 'lr_monitor':
        return LRMonitor()
    elif name == 'memory_monitor':
        return MemoryMonitor()
    elif name == 'speed_monitor':
        return SpeedMonitor(window_size=kwargs.get('window_size', 1),
                            gpu_flops_available=kwargs.get(
                                'gpu_flops_available', None))
    elif name == 'fdiff':
        return FDiffMetrics(**kwargs)
    elif name == 'runtime_estimator':
        return RuntimeEstimator()
    elif name == 'optimizer_monitor':
        return OptimizerMonitor(log_optimizer_metrics=kwargs.get(
            'log_optimizer_metrics', True),)
    elif name == 'health_checker':
        return HealthChecker(**kwargs)
    elif name == 'generate_callback':
        prompts = kwargs.pop('prompts')
        return Generate(prompts=list(prompts), **kwargs)
    elif name == 'global_lr_scaling':
        return GlobalLRScaling(**kwargs)
    elif name == 'layer_freezing':
        return LayerFreezing(**kwargs)
    elif name == 'mono_ckpt_saver':
        return MonolithicCheckpointSaver(**kwargs)
    elif name == 'scheduled_gc':
        return ScheduledGarbageCollector(**kwargs)
    elif name == "sharded_ckpt_saver":
        return ShardedCheckpointSaver(**kwargs)
    else:
        raise ValueError(f'Not sure how to build callback: {name}')

def main(cfg):
    reproducibility.seed_all(cfg.seed)

    og_state_dict = torch.load(cfg.load_path)

    del og_state_dict['state']['model']
    optimizer_name = list(og_state_dict['state']['optimizers'].keys())[0]
    print ("optimizer name is: ", optimizer_name)
    del og_state_dict['state']['optimizers']

    for key in og_state_dict['state'].keys():
        print ("key is: ", key)

    model_cfg = cfg.model
    model = build_composer_model(model_cfg, cfg.tokenizer)

    # Get batch size info
    cfg = update_batch_size_info(cfg)

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
    # print('Building train loader...')
    # train_loader = build_dataloader(cfg.train_loader,
                                    # cfg.device_train_batch_size)
    # print('Building eval loader...')
    evaluators = []
    if 'eval_loader' in cfg:
        eval_loader = Evaluator(label='eval',
                                dataloader=build_dataloader(
                                    cfg.eval_loader,
                                    cfg.device_eval_batch_size),
                                metric_names=list(model.train_metrics.keys()))
        evaluators.append(eval_loader)

    if 'icl_tasks' in cfg:
        icl_evaluators, _ = build_icl_evaluators(cfg, model.tokenizer)
        evaluators.extend(icl_evaluators)

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in (cfg.get('loggers') or {}).items()
    ]

    # Callbacks
    callbacks = [
        build_callback(name, callback_cfg)
        for name, callback_cfg in (cfg.get('callbacks') or {}).items()
    ]

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in (cfg.get('algorithms') or {}).items()
    ]

    # Build the Trainer
    print('Building trainer...')
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        # train_dataloader=train_loader,
        eval_dataloader=evaluators,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        # eval_interval=cfg.eval_interval,
        # eval_subset_num_batches=cfg.get('eval_subset_num_batches', -1),
        progress_bar=cfg.get('progress_bar', False),
        log_to_console=cfg.get('log_to_console', True),
        console_log_interval=cfg.get('console_log_interval', '1ba'),
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        algorithms=algorithms,
        device_train_microbatch_size=cfg.get('device_train_microbatch_size',
                                             'auto'),
        fsdp_config=fsdp_config,  # type: ignore
        save_folder=cfg.get('save_folder', None),
        save_filename=cfg.get('save_filename',
                              'ep{epoch}-ba{batch}-rank{rank}.pt'),
        save_latest_filename=cfg.get('save_latest_filename',
                                     'latest-rank{rank}.pt'),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep',
                                             -1),
        save_overwrite=cfg.get('save_overwrite', False),
        load_path=cfg.get('load_path', None),
        load_weights_only=cfg.get('load_weights_only', False),
        load_ignore_keys=cfg.get('load_ignore_keys', None),
        autoresume=cfg.get('autoresume', False),
        python_log_level=cfg.get('python_log_level', None),
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
