import gc
import json
import math
import os
import random
import sys
import time
import traceback
from collections import deque
from contextlib import nullcontext
from functools import partial
from distutils.util import strtobool
from typing import List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from torch.nn import functional as F
from torch.profiler import record_function
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast


import infinity.utils.dist as dist  
from infinity.dataset.build import build_t2i_dataset
from infinity.utils.save_and_load import auto_resume, TXTSaver
from infinity.utils import arg_util, misc, wandb_utils
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

from torch.utils.data import Dataset
import argparse
from infinity.models.infinity_final import Infinity
from torchvision import transforms
from PIL import Image
from PIL.ImageOps import exif_transpose
import copy
from pathlib import Path

class PrefixConfig:
    def __init__(self, ti_seq_len: int=1):
        self.ti_seq_len = ti_seq_len

enable_timeline_sdk = False

def encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False,
                  special_token_id=None, special_token=None):
    if enable_positive_prompt:
        print(f'before positive_prompt aug: {prompt}')
        prompt = aug_with_positive_prompt(prompt)
        print(f'after positive_prompt aug: {prompt}')

    print(f'prompt={prompt}')
    tokens = text_tokenizer(text=prompt, max_length=512, padding='max_length',
                            truncation=True, return_tensors='pt')
    input_ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)
    
    text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
    orig_lens = mask.sum(dim=-1).tolist()
    
    new_kv_compact_list = []
    new_lens = []
    batch_size = input_ids.shape[0]
    for i in range(batch_size):
        valid_len = orig_lens[i]
        sample_ids = input_ids[i, :valid_len]
        sample_feats = text_features[i, :valid_len]  
        sample_new_feats = [] 
        for j in range(valid_len):
            token = sample_ids[j].item()
            if special_token_id is not None and special_token is not None and token == special_token_id:
                sample_new_feats.append(special_token)  # (num_special_tokens, D)
            else:
                sample_new_feats.append(sample_feats[j].unsqueeze(0))
        sample_new = torch.cat(sample_new_feats, dim=0)
        new_kv_compact_list.append(sample_new)
        new_lens.append(sample_new.shape[0])
    kv_compact = torch.cat(new_kv_compact_list, dim=0)
    cu_seqlens_k = F.pad(torch.tensor(new_lens, dtype=torch.int32, device=input_ids.device).cumsum(dim=0), (1, 0))
    cu_seqlens_k = cu_seqlens_k.to(torch.int32)
    Ltext = max(new_lens)
    
    text_cond_tuple = (kv_compact, new_lens, cu_seqlens_k, Ltext)
    return text_cond_tuple


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    max_length = 512
    text_inputs = tokenizer(
        text=prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        regularize_prompt,
        class_prompt,
        tokenizer,
        size=1024,
        class_data_root='',
        center_crop=False,
        encoder_hidden_states=None,
        tokenizer_max_length=None,
        prompt_template_file="prompt_upsampled_short.txt",
        prompt_upsampling=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = Path(instance_data_root)
        if len(class_data_root) > 0:
            self.class_data_root = Path(class_data_root)
        else:
            self.class_data_root = ''
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")
        if len(self.class_data_root) > 0 and not self.class_data_root.exists():
            raise ValueError(f"Instance {self.class_data_root} images root doesn't exists.")
        if len(self.class_data_root) > 0:
            self.class_images_path = list(Path(class_data_root).iterdir())
            self.num_class_images = len(self.class_images_path)

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self.regularize_prompt = regularize_prompt
        self.class_prompt = class_prompt
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        if prompt_upsampling:
            with open(prompt_template_file, "r", encoding="utf-8") as f:
                templates = f.read().strip().split("\n")
            self.prompt_templates = [temp.strip() for temp in templates if temp.strip()]
        else:
            self.prompt_templates = None

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if len(self.class_data_root) > 0:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            if self.prompt_templates is not None:
                if random.random() < 0.2:
                    chosen_prompt = self.regularize_prompt
                else:
                    chosen_prompt = random.choice(self.prompt_templates).format(self.class_prompt)
            else:
                chosen_prompt = self.regularize_prompt
            reg_inputs = tokenize_prompt(
                self.tokenizer, chosen_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask
            example["reg_prompt_ids"] = reg_inputs.input_ids
            example["reg_attention_mask"] = reg_inputs.attention_mask

        return example


def collate_fn(examples, with_reg=True):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    reg_input_ids = [example["reg_prompt_ids"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]
        reg_attention_mask = [example["reg_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    reg_input_ids = torch.cat(reg_input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "reg_input_ids": reg_input_ids,
        "pixel_values": pixel_values,
    }

    if "class_images" in examples[0].keys():
        cls_pixel_values = [example["class_images"] for example in examples]
        class_pixel_values = torch.stack(class_pixel_values)
        class_pixel_values = class_pixel_values.to(memory_format=torch.contiguous_format).float()
        batch["class_pixel_values"] = class_pixel_values

    if has_attention_mask:
        attention_mask = torch.cat(attention_mask, dim=0)
        reg_attention_mask = torch.cat(reg_attention_mask, dim=0)
        batch["attention_mask"] = attention_mask
        batch["reg_attention_mask"] = reg_attention_mask

    return batch

def load_infinity_for_training(
    vae, 
    args,
    device='cuda',
    requires_grad=True,
):
    if args.model_type == 'infinity_2b':
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8) # 2b model
    elif args.model_type == 'infinity_layer12':
        kwargs_model = dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer16':
        kwargs_model = dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer24':
        kwargs_model = dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer32':
        kwargs_model = dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer40':
        kwargs_model = dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer48':
        kwargs_model = dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)

    prefix_config = PrefixConfig(args.text_prefix_length)

    infinity_model = Infinity(
        vae_local=vae, 
        text_channels=args.text_channels, 
        text_maxlen=512,
        prefix_config=prefix_config,
        shared_aln=True, 
        raw_scale_schedule=None,   
        checkpointing='full-block',
        customized_flash_attn=False,
        fused_norm=True,
        pad_to_multiplier=128,
        use_flex_attn=args.use_flex_attn,
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block,
        use_bit_label=args.use_bit_label,
        rope2d_each_sa_layer=args.rope2d_each_sa_layer,
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        pn=args.pn,
        apply_spatial_patchify=args.apply_spatial_patchify,
        inference_mode=False,  
        train_h_div_w_list=[1.0],  
        **kwargs_model,
    ).to(device=device)

    if args.bf16:
        for block in infinity_model.unregistered_blocks:
            block.bfloat16()

    infinity_model.train()  
    infinity_model.requires_grad_(False)  
    if requires_grad:
        infinity_model.special_token.requires_grad_(True)
        for i in range(len(infinity_model.block_chunks)):
            for j in range(len(infinity_model.block_chunks[i].module)):
                if args.enable_sa_layer or args.enable_all_layer:
                    infinity_model.block_chunks[i].module[j].sa.requires_grad_(True)
                if args.enable_ca_layer or args.enable_all_layer:
                    infinity_model.block_chunks[i].module[j].ca.requires_grad_(True)
                if args.enable_ffn_layer or args.enable_all_layer:
                    infinity_model.block_chunks[i].module[j].ffn.requires_grad_(True)
    infinity_model.cuda()
    
    print(f'[Load Infinity weights from {args.model_path}]')
    state_dict = torch.load(args.model_path, map_location=device)
    missing, unexpected = infinity_model.load_state_dict(state_dict, strict=False)
    print(f'load_state_dict done; missing={missing}, unexpected={unexpected}')
    
    return infinity_model

def load_visual_tokenizer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load vae
    if args.vae_type in [16,18,20,24,32,64]:
        from infinity.models.bsq_vae.vae import vae_model
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type
        codebook_size = 2**codebook_dim
        if args.apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult=[1, 2, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult=[1, 2, 4, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4, 4]
        vae = vae_model(args.vae_path, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                        encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(device)
    return vae

def build_everything_from_args(args: arg_util.Args, saver):
    # set seed
    args.set_initial_seed(benchmark=True)
    if args.seed is not None and not args.rand: 
        misc.check_randomness(args)


    # build models
    text_tokenizer, text_encoder, vae_local, gpt_uncompiled, gpt_ori, gpt_single, gpt_ema_single, gpt_optim, text_optim, special_token_id, text_embedding_weight, other_weight = build_model_optimizer(args)

    # build dataset
    iters_train, ld_train, ld_val = build_dataloaders(args, text_tokenizer)   
    # build data
    
    # InfinityTrainer import
    from booth_trainer_final import InfinityTrainer

    # build trainer
    trainer = InfinityTrainer(
        is_visualizer=True,  
        device=args.device,
        raw_scale_schedule=args.scale_schedule,
        resos=args.resos,
        vae_local=vae_local,
        gpt_wo_ddp=None,    
        gpt=gpt_single,     
        ema_ratio=args.tema,
        max_it=iters_train * args.ep,
        gpt_opt=gpt_optim,
        txt_opt=text_optim,
        label_smooth=args.ls,
        z_loss_ratio=args.lz,
        eq_loss=args.eq,
        xen=args.xen,
        dbg_unused=args.dbg,
        zero=False,  
        vae_type=args.vae_type,
        gpt_wo_ddp_ema=gpt_ema_single,  
        gpt_ema=gpt_ema_single, 
        gpt_ori=gpt_ori,
        use_fsdp_model_ema=False,       
        other_args=args,
    )
    trainer.special_token_id = special_token_id
    
    # auto resume
    auto_resume_info, start_ep, start_it, acc_str, eval_milestone, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    # args.dump_log()

    if start_ep == args.ep:
        args.dump_log()
        print(f'[vgpt] AR finished ({acc_str}), skipping ...\n\n')
        return None

    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True)
    
    start_it = 0
    print(f"{start_it=}, {iters_train=}")

    return (
        text_tokenizer, text_encoder, vae_local, text_embedding_weight, other_weight, trainer,
        start_ep, start_it, acc_str, eval_milestone, iters_train, ld_train, ld_val
    )


def build_model_optimizer(args):
    from infinity.models.infinity_final import Infinity, MultipleLayers
    from infinity.models.init_param import init_weights
    from infinity.utils.amp_opt import AmpOptimizer, TextEmbeddingAmpOptimizer
    from infinity.utils.lr_control import filter_params
    from infinity.utils.load import build_vae_gpt
    
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    
    # del vae_ckpt
    vae_local = load_visual_tokenizer(args)
    base_model = load_infinity_for_training(vae_local, args)

    teacher_model = copy.deepcopy(base_model)
    if args.enable_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            target_modules=[
                "mat_qkv",
                "mat_kv",
                "proj",
            ],
        )
        train_model = get_peft_model(base_model, lora_config)
    else:
        train_model = base_model
    gpt_wo_ddp = train_model
    gpt_uncompiled = train_model
    gpt_single = args.compile_model(gpt_wo_ddp, args.tfast)
    gpt_ori = teacher_model.to(args.device)
    gpt_ori.eval()
    gpt_ori.requires_grad_(False)

    gpt_wo_ddp_ema = None

    # rwe
    if args.rwe:
        gpt_wo_ddp.word_embed.weight.requires_grad = False
        torch.nn.init.trunc_normal_(gpt_wo_ddp.word_embed.weight.data, std=1.5 * math.sqrt(1 / gpt_wo_ddp.C / 3))
        if hasattr(gpt_wo_ddp.word_embed, 'bias'):
            gpt_wo_ddp.word_embed.bias.requires_grad = False
            gpt_wo_ddp.word_embed.bias.data.zero_()
    
    ndim_dict = {name: para.ndim for name, para in gpt_wo_ddp.named_parameters() if para.requires_grad}
    print(f'[PT] GPT model = {gpt_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}'
    print(f'[PT][#para] VAE={count_p(vae_local)}, VAE.quant={count_p(vae_local.quantize)}')
    print(f'[PT][#para] GPT={count_p(gpt_wo_ddp)}\n\n')
    

    if args.resume_ckpts is not None and len(args.resume_ckpts) > 0:
        model_folder = os.path.join(args.resume_ckpts)
        files = sorted([f for f in os.listdir(model_folder) 
                    if os.path.isfile(os.path.join(model_folder, f)) and f.endswith('.pth')])
        if files:
            model_path = os.path.join(model_folder, files[-1])
        else:
            model_path = model_folder 
        state_dict = torch.load(model_path, map_location=args.device)
        missing, unexpected = gpt_single.load_state_dict(state_dict, strict=False)

    gpt_ema_single = gpt_wo_ddp_ema

    # =============== build optimizer ===============
    nowd_keys = set()
    if args.nowd >= 1:
        nowd_keys |= {
            'cls_token', 'start_token', 'task_token', 'cfg_uncond',
            'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
            'gamma', 'beta',
            'ada_gss', 'moe_bias',
            'scale_mul',
            'text_proj_for_sos.ca.mat_q',
        }
    if args.nowd >= 2:
        nowd_keys |= {'class_emb', 'embedding'}
    names, paras, para_groups = filter_params(gpt_single, ndim_dict, nowd_keys=nowd_keys)
    del ndim_dict

    if '_' in args.ada:
        beta0, beta1 = map(float, args.ada.split('_'))
    else:
        beta0, beta1 = float(args.ada), -1
    
    opt_clz = {
        'sgd':   partial(torch.optim.SGD, momentum=beta0, nesterov=True),
        'adam':  partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.afuse),
    }[args.opt]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    if args.oeps:
        opt_kw['eps'] = args.oeps

    print(f'[vgpt] optim={opt_clz}, opt_kw={opt_kw}\n')
    from infinity.utils.amp_opt import AmpOptimizer

    gpt_optim = None
    if args.online_t5: # jw: True
        print(f'Loading T5 from {args.t5_path}...')
        text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(args.t5_path, revision=None, legacy=True)
        text_tokenizer.model_max_length = args.tlen
        text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(args.t5_path, torch_dtype=torch.float16)
        text_encoder.to(args.device)

        special_token = "<placeholder>"
        text_tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})
        text_encoder.resize_token_embeddings(len(text_tokenizer))
        special_token_id = text_tokenizer.convert_tokens_to_ids(special_token)
        text_embedding_weight = text_encoder.get_input_embeddings().weight

        text_embedding_weight.requires_grad = False
        text_encoder.gradient_checkpointing_enable()


        other_weight = []
        gpt_single.special_token.requires_grad = True
        other_weight += paras

        adamw = torch.optim.AdamW(
            [gpt_single.special_token] +
            other_weight,
            **opt_kw
        )
        txt_amp_opt = TextEmbeddingAmpOptimizer(
            model_name_3letters="txt",
            mixed_precision=2,      
            optimizer=adamw,
            embedding_parameters=[gpt_single.special_token] + other_weight,
            r_accu=args.r_accu,
            grad_clip=1.0,         
            enable_grad_clip=False,
            zero=0,
        )
    else:
        text_tokenizer = None
        text_encoder = None
    del names, paras, para_groups
    
    return text_tokenizer, text_encoder, vae_local, gpt_uncompiled, gpt_ori, gpt_single, gpt_ema_single, gpt_optim, txt_amp_opt, special_token_id, gpt_single.special_token, other_weight


def build_dataloaders(args, text_tokenizer):
    dataset_train = DreamBoothDataset(
        instance_data_root=args.data_path,
        instance_prompt=args.instance_prompt,
        regularize_prompt=args.regularize_prompt,
        class_prompt=args.class_prompt,
        class_data_root=args.reg_data_path,
        tokenizer=text_tokenizer,
        size=1024,
        center_crop=args.center_crop,
    )
    if len(dataset_train) < args.finetuning_batch_size:
        print(f"Dataset size is smaller than the batch size. Finetuning batch size is set to the dataset size.")
        args.finetuning_batch_size = len(dataset_train)

    ld_train = DataLoader(
        dataset_train,
        batch_size=args.finetuning_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        collate_fn=lambda examples: collate_fn(examples, args.with_reg),
    )
    ld_val = None
    iters_train = len(ld_train)

    return iters_train, ld_train, ld_val


def main_train(args: arg_util.Args):
    saver = TXTSaver(True, eval_milestone=None)

    ret = build_everything_from_args(args, saver)
    if ret is None:
        return
    
    (
        text_tokenizer, text_encoder, vae_local, text_embedding_weight, other_weight, trainer,
        start_ep, start_it, acc_str, eval_milestone,
        iters_train, ld_train, ld_val
    ) = ret

    gc.collect()
    torch.cuda.empty_cache()
    
    from booth_trainer_final import InfinityTrainer

    if args.wandb: 
        run_name = args.local_out_path.split(os.sep)[-2]
        wandb_utils.wandb.init(project=f"{args.project_name}_{args.exp_name}", name=run_name, config={})

    world_size = 1  
    start_time = time.time()
    min_L_mean, min_L_tail, max_acc_mean, max_acc_tail = 999., 999., -1., -1.
    last_val_loss_mean, best_val_loss_mean = 999., 999.
    last_val_loss_tail, best_val_loss_tail = 999., 999.
    last_val_acc_mean, best_val_acc_mean = 0., 0.
    last_val_acc_tail, best_val_acc_tail = 0., 0.

    # epoch loop
    cur_iter = 0
    for ep in range(start_ep, args.ep):
        # train one epoch
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep=ep,
            is_first_ep=(ep == start_ep),
            start_it=start_it if ep == start_ep else 0,
            me=None,
            saver=saver,
            args=args,
            ld_or_itrt=iter(ld_train),
            iters_train=iters_train,
            text_tokenizer=text_tokenizer,
            text_encoder=text_encoder,
            vae=vae_local,
            text_embedding_weight=text_embedding_weight,
            other_weight=other_weight,
            trainer=trainer,
            logging_params_milestone=[],
            enable_timeline_sdk=enable_timeline_sdk,
        )

        L_mean, L_tail = stats['Lm'], stats['Lt']
        acc_mean, acc_tail = stats['Accm'], stats['Acct']
        grad_norm = stats['tnm']

        min_L_mean = min(min_L_mean, L_mean)
        max_acc_mean = max(max_acc_mean, acc_mean)
        if L_tail != -1:
            min_L_tail = min(min_L_tail, L_tail)
        if acc_tail > -1:
            max_acc_tail = max(max_acc_tail, acc_tail)

        is_val_and_also_saving = ((ep + 1) % max(1, args.ep // 25) == 0) or ((ep + 1) == args.ep)
        if is_val_and_also_saving:
            if ld_val is None or isinstance(ld_val, int):
                last_val_loss_mean = 0.666
                last_val_loss_tail = 0.555
                last_val_acc_mean = 5.55
                last_val_acc_tail = 6.66
            else:
                last_val_loss_mean, last_val_loss_tail, last_val_acc_mean, last_val_acc_tail, tot, cost = trainer.eval_ep(ep, args, ld_val)

            best_val_loss_mean = min(best_val_loss_mean, last_val_loss_mean)
            best_val_loss_tail = min(best_val_loss_tail, last_val_loss_tail)
            best_val_acc_mean = max(best_val_acc_mean, last_val_acc_mean)
            best_val_acc_tail = max(best_val_acc_tail, last_val_acc_tail)
        
        print(f'  [*] [ep{ep}]  Lmean: {min_L_mean:.3f} (cur {L_mean:.3f}), Ltail {min_L_tail:.3f} (cur {L_tail:.3f}),'
              f'  Acc m-t: {max_acc_mean:.2f} {max_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}',
              flush=True)
        cur_iter += iters_train
        if cur_iter > args.train_iter:
            break
        
    total_time = f'{(time.time() - start_time) / 3600:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total Time: {total_time},   '
          f' Lm: {min_L_mean:.3f} ({L_mean}),   Lt: {min_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)
    return

@torch._dynamo.disable
def gen_one_img(
    infinity_test, 
    vae, 
    text_tokenizer,
    text_encoder,
    prompt, 
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    special_token_id=None,
    special_token=None,
    text_cond_tuple=None,
):
    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    if text_cond_tuple is None:
        text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt, special_token_id=special_token_id, special_token=special_token)
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
    else:
        negative_label_B_or_BLT = None
    print(f'cfg: {cfg_list}, tau: {tau_list}')
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=False):
        stt = time.time()
        _, _, img_list = infinity_test.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
            B=1, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
            cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
            cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type, softmax_merge_topk=softmax_merge_topk,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
            sampling_per_bits=sampling_per_bits,
        )
    print(f"cost: {time.time() - sstt}, infinity cost={time.time() - stt}")
    img = img_list[0]
    return img

g_speed_ls = deque(maxlen=128)
def train_one_ep(
    ep: int,
    is_first_ep: bool,
    start_it: int,
    me,
    saver: TXTSaver,
    args: arg_util.Args,
    ld_or_itrt,
    iters_train: int, 
    text_tokenizer: T5TokenizerFast,
    text_encoder: T5EncoderModel,
    vae,
    text_embedding_weight,
    other_weight,
    trainer,
    logging_params_milestone,
    enable_timeline_sdk: bool,
):
    from booth_trainer_final import InfinityTrainer
    from infinity.utils.lr_control import lr_wd_annealing
    trainer: InfinityTrainer
    
    step_cnt = 0
    header = f'[Ep]: [{ep:4d}/{args.train_iter // iters_train}]'

    g_it = ep * iters_train
    max_it = args.ep * iters_train
    me = misc.MetricLogger() if me is None else me
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{value:.2g}')) for x in ['tlr']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['tnm']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]

    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.3f} ({global_avg:.3f})')) for x in ['reg_Lm', 'reg_Lt']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['reg_Accm', 'reg_Acct']]


    last_t_perf = time.time()
    speed_ls = g_speed_ls
    FREQ = min(args.prof_freq, iters_train//2) if iters_train > 2 else 1

    # eval list making
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [ (1, h, w) for (_, h, w) in scale_schedule]

    with open(args.val_prompt_path, "r") as fin:
        val_prompts = fin.read()
        val_prompts = val_prompts.split("\n")
    
    g_it = ep * iters_train

    for it, data in me.log_every(start_it, iters_train, ld_or_itrt, args.log_freq, args.log_every_iter, header):
        g_it = ep * iters_train + it
        if g_it > args.train_iter:
            break

        if (it + 1) % FREQ == 0:
            speed_ls.append((time.time() - last_t_perf) / FREQ)
            last_t_perf = time.time()

        with torch.no_grad():
            if (g_it + 1) % args.save_txt_iters_freq == 0 or (g_it + 1) % args.save_model_iters_freq == 0:
                generated_images = []
                prompts = []
                # todo
                for val_idx, val_prompt in enumerate(val_prompts):
                    val_prompt_ = val_prompt.format(args.instance_prompt)
                    generated_image = gen_one_img(
                        trainer.gpt,
                        vae,
                        text_tokenizer,
                        text_encoder,
                        val_prompt_,
                        g_seed=args.seed,
                        gt_leak=0,
                        gt_ls_Bl=None,
                        cfg_list=args.cfg,
                        tau_list=args.tau,
                        scale_schedule=scale_schedule,
                        vae_type=args.vae_type,
                        cfg_insertion_layer=[args.cfg_insertion_layer],
                        sampling_per_bits=args.sampling_per_bits,
                        enable_positive_prompt=args.enable_positive_prompt,
                        special_token_id=trainer.special_token_id,
                        special_token=trainer.gpt.special_token,
                    )
                    generated_images.append(generated_image)
                    prompts.append(val_prompt_)
                for val_idx, val_prompt in enumerate(val_prompts):
                    val_prompt_ = val_prompt.format(args.regularize_prompt)
                    generated_image = gen_one_img(
                        trainer.gpt,
                        vae,
                        text_tokenizer,
                        text_encoder,
                        val_prompt_,
                        g_seed=args.seed,
                        gt_leak=0,
                        gt_ls_Bl=None,
                        cfg_list=args.cfg,
                        tau_list=args.tau,
                        scale_schedule=scale_schedule,
                        vae_type=args.vae_type,
                        cfg_insertion_layer=[args.cfg_insertion_layer],
                        sampling_per_bits=args.sampling_per_bits,
                        enable_positive_prompt=args.enable_positive_prompt,
                    )
                    generated_images.append(generated_image)
                    prompts.append(val_prompt)

                saver.sav(
                    args=args,
                    g_it=(g_it + 1),
                    next_ep=ep,
                    next_it=it + 1,
                    trainer=trainer,
                    generated_images = generated_images,
                    prompts = prompts,
                    acc_str='[todo]',
                    eval_milestone=None,
                    also_save_to=None,
                    best_save_to=None,
                    only_img_save=True,
                )

            if (g_it + 1) % args.save_model_iters_freq == 0:
                saver.sav(
                    args=args,
                    g_it=(g_it + 1),
                    next_ep=ep,
                    next_it=it + 1,
                    trainer=trainer,
                    generated_images = generated_images,
                    prompts = prompts,
                    acc_str='[todo]',
                    eval_milestone=None,
                    also_save_to=None,
                    best_save_to=None,
                )

        inp = data["pixel_values"] # (1, 3, 1024, 1024)
        bs = inp.shape[0]
        input_ids = data["input_ids"] # (1, 1024)
        attention_mask = data["attention_mask"] # (1, 1024)

        input_ids = input_ids.cuda(non_blocking=True)
        mask = attention_mask.cuda(non_blocking=True)
        text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()

        lens = mask.sum(dim=-1).tolist()
        cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
        Ltext = max(lens)

        kv_compact = []
        for len_i, feat_i in zip(lens, text_features.unbind(0)):
            kv_compact.append(feat_i[:len_i])
        kv_compact = torch.cat(kv_compact, dim=0)

        reg_input_ids = data["reg_input_ids"] # (1, 1024)
        reg_attention_mask = data["reg_attention_mask"] # (1, 1024)

        reg_input_ids = reg_input_ids.cuda(non_blocking=True)
        reg_mask = reg_attention_mask.cuda(non_blocking=True)
        reg_text_features = text_encoder(input_ids=reg_input_ids, attention_mask=reg_mask)['last_hidden_state'].float()
        reg_cu_seqlens_k = F.pad(reg_mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
        reg_lens = reg_mask.sum(dim=-1).tolist()
        reg_Ltext = max(reg_lens)

        reg_kv_compact = []
        for len_i, feat_i in zip(reg_lens, reg_text_features.unbind(0)):
            reg_kv_compact.append(feat_i[:len_i])
        reg_kv_compact = torch.cat(reg_kv_compact, dim=0)
        reg_text_cond_tuple = (reg_kv_compact, reg_lens, reg_cu_seqlens_k, reg_Ltext)

        if trainer.special_token_id is not None:
            new_kv_compact_list = []  
            new_lens = []             
            batch_size = input_ids.shape[0]
            for b in range(batch_size):
                valid_len = lens[b]
                sample_ids = input_ids[b, :valid_len]
                sample_feats = text_features[b, :valid_len]  # shape: (valid_len, D)
                sample_new_feats = [] 
                for i, token in enumerate(sample_ids):
                    if token.item() == trainer.special_token_id:
                        sample_new_feats.append(trainer.gpt.special_token)  # shape: (4, D)
                    else:
                        sample_new_feats.append(sample_feats[i].unsqueeze(0))
                sample_new = torch.cat(sample_new_feats, dim=0)
                new_kv_compact_list.append(sample_new)
                new_lens.append(sample_new.shape[0])
            kv_compact = torch.cat(new_kv_compact_list, dim=0)
            lens = new_lens
            cu_seqlens_k = torch.tensor([0] + list(np.cumsum(new_lens)), dtype=torch.int32, device=input_ids.device)


        text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)

        inp = inp.to(args.device, non_blocking=True)

        if len(args.reg_data_path) > 0:
            cls_inp = data["class_pixel_values"] # (1, 3, 1024, 1024)
            cls_inp = cls_inp.to(args.device, non_blocking=True)
        else:
            cls_inp = None


        # lr schedule
        wp_it = args.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(
            args.sche,
            trainer.txt_opt.optimizer,
            args.tlr,
            args.twd,
            args.twde,
            g_it,
            wp_it,
            max_it,
            wp0=args.wp0,
            wpe=args.wpe
        )

        progress = g_it / (max_it - 1)
        clip_decay_ratio = (0.3 ** (20 * progress) + 0.2) if args.cdec else 1
        stepping = ((g_it + 1) % args.ac == 0)
        step_cnt += int(stepping)

        grad_norm_t, scale_log2_t = trainer.train_step(
            ep=ep,
            it=it,
            g_it=g_it,
            stepping=stepping,
            clip_decay_ratio=clip_decay_ratio,
            metric_lg=me,
            logging_params=(stepping and step_cnt == 1 and (ep < 4 or ep in logging_params_milestone)),
            inp_B3HW=inp,
            cls_inp_B3HW=cls_inp,
            text_cond_tuple=text_cond_tuple,
            reg_text_cond_tuple=reg_text_cond_tuple,
            args=args,
        )
        me.update(tlr=max_tlr)
    

    if (ep == args.ep - 1 or g_it >= args.train_iter):

        # last save
        generated_images = []
        prompts = []
        # todo
        for val_idx, val_prompt in enumerate(val_prompts):
            val_prompt_ = val_prompt.format(args.instance_prompt)
            generated_image = gen_one_img(
                trainer.gpt,
                vae,
                text_tokenizer,
                text_encoder,
                val_prompt_,
                g_seed=args.seed,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=args.cfg,
                tau_list=args.tau,
                scale_schedule=scale_schedule,
                vae_type=args.vae_type,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
                special_token_id=trainer.special_token_id,
                special_token=trainer.gpt.special_token,
            )
            generated_images.append(generated_image)
            prompts.append(val_prompt_)
        for val_idx, val_prompt in enumerate(val_prompts):
            val_prompt_ = val_prompt.format(args.regularize_prompt)
            generated_image = gen_one_img(
                trainer.gpt,
                vae,
                text_tokenizer,
                text_encoder,
                val_prompt_,
                g_seed=args.seed,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=args.cfg,
                tau_list=args.tau,
                scale_schedule=scale_schedule,
                vae_type=args.vae_type,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
            )
            generated_images.append(generated_image)
            prompts.append(val_prompt)
        saver.sav(
            args=args,
            g_it=(g_it + 1),
            next_ep=ep,
            next_it=it + 1,
            trainer=trainer,
            generated_images = generated_images,
            prompts = prompts,
            acc_str='[todo]',
            eval_milestone=None,
            also_save_to=None,
            best_save_to=None
        )

    return (
        {k: meter.global_avg for k, meter in me.meters.items()},
        me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)
    )


def main():
    args: arg_util.Args = arg_util.init_dist_and_get_args()  
    main_train(args)

    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    args.cur_phase = 'OK'

    time.sleep(2)


if __name__ == '__main__':
    try:
        main()
    except Exception as _e:
        try:
            print(f'[err]:\n{_e}')
            traceback.print_exc()
        except:
            pass
        raise _e
