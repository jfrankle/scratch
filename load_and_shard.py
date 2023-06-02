import sys
import os
import warnings

from composer import Trainer
from omegaconf import OmegaConf as om
from omegaconf import DictConfig
from examples.llm.src import COMPOSER_MODEL_REGISTRY
from composer.core import Event

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

def build_tokenizer(om_tokenizer_config: DictConfig,):
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    resolved_om_tokenizer_config = om.to_container(om_tokenizer_config,
                                                   resolve=True)
    tokenizer_kwargs = resolved_om_tokenizer_config.get(  # type: ignore
        'kwargs', {})
    tokenizer_name = resolved_om_tokenizer_config['name']  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                              **tokenizer_kwargs)

    # HuggingFace does not respect the model_max_length kwarg, and overrides it with
    # min(kwargs['model_max_length'], original_config['model_max_length']), so we
    # explicitly set it here
    tokenizer.model_max_length = tokenizer_kwargs.get(
        'model_max_length',
        int(1e30),
    )

    return tokenizer

def main(cfg):
    model_cfg = cfg.model
    model = build_composer_model(model_cfg, cfg.tokenizer)
    
    sharded_checkpoint_saver = ShardedCheckpointSaver(**cfg.callbacks.sharded_ckpt_saver)

    trainer = Trainer(model=model,
            fsdp_config=cfg.fsdp_config,
            load_path=cfg.load_path,
            # callbacks=callbacks,
            # max_duration=0,
            )

    sharded_checkpoint_saver._save_checkpoint(trainer.state)

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
