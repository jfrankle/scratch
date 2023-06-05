import torch
from composer.utils import dist, get_device
from transformers import AutoTokenizer
from llmfoundry.models.hf import ComposerHFCausalLM
from omegaconf import OmegaConf as om

from transformers import AutoModel, AutoModelForCausalLM
from accelerate import init_empty_weights

from composer.trainer.dist_strategy import prepare_fsdp_module
from composer.core import Precision

# model_type = 'hf_model'
model_type = 'composer_model'
num_repeats = 1

def main():    
    device = get_device(None)

    dist.initialize_dist(device, timeout=600)
    # model_name = "mosaicml/mpt-7b"
    # model_name = "EleutherAI/pythia-410m-deduped"
    model_name = "EleutherAI/gpt-neox-20b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]])

    if model_type == 'composer_model':
        if dist.get_local_rank() == 0:
            cfg = om.create({
                'pretrained_model_name_or_path': model_name,
                'pretrained': True,
                'init_device': 'cpu'
            })
            model = ComposerHFCausalLM(cfg, tokenizer)
            output = model(dict(input_ids=input_ids, labels=input_ids))
            print ("intermediate output is: ", output.logits.mean())
        else:
            cfg = om.create({
                'pretrained_model_name_or_path': model_name,
                'pretrained': False,
                'init_device': 'meta'
            })
            model = ComposerHFCausalLM(cfg, tokenizer)
    elif model_type == 'hf_model':
        if dist.get_local_rank() == 0:
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, config={'pretrained': True})
            for _ in range(num_repeats):
                # AutoModel, not causal
                # print ("original model output is: ", model(input_ids).last_hidden_state.mean())
                print ("original model output is: ", model(input_ids).logits.mean())
        else:
            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, config={'pretrained': True})
        print ("model is: ", model)

        # for module in model.transformer.children():
            # if isinstance(module, torch.nn.ModuleList) or isinstance(module, torch.nn.Dropout):
                # continue
            # if isinstance(module, torch.nn.Module):
                # module._fsdp_wrap = True

        # This code works for GPT-NeoX
        for layer in model.gpt_neox.layers:
            layer._fsdp_wrap = True
        model.gpt_neox.embed_in._fsdp_wrap = True
        model.gpt_neox.final_layer_norm._fsdp_wrap = True

        model.gpt_neox.param_init_fn = lambda module: model._init_weights(module)

    num_p = 0
    for _, p in model.named_parameters():
        num_p += p.numel()
    
    dist.barrier()

    precision = Precision('amp_bf16')
    prepare_fsdp_module(model, [], {'sharding_strategy': 'FULL_SHARD', 'mixed_precision': 'FULL','sync_module_states': True, 'verbose': True}, precision=precision, device=device, auto_microbatching=False)

    dist.barrier()

    from composer.optim import DecoupledSGDW
    optimizer = DecoupledSGDW(model.parameters(), lr=0.1)

    print ("model is: ", model)

    input_ids.to(torch.cuda.current_device())
    print('Running forward pass')
    if model_type == 'composer_model':
        output = model(dict(input_ids=input_ids, labels=input_ids))
    else:
        output = model(input_ids=input_ids)
    for _ in range(num_repeats):
        # print(output.last_hidden_state.mean(), "\nI GOT OUTPUT!")
        print(output.logits.mean(), "\nI GOT OUTPUT!")
    
    output['loss'].backward()
    print('I went backward!')

    optimizer.step()
    print('I TOOK A STEPP!!!!')

if __name__ == "__main__":
    main()