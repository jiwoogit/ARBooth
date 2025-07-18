import gc
import os
import subprocess
import time
import re
import numpy as np
from typing import List, Optional, Tuple

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import glob
import shutil
from infinity.utils import arg_util
import infinity.utils.dist as dist
import cv2


def glob_with_epoch_iter(pattern, recursive=False): 
    def extract_ep_iter(filename):
        match = re.search(r'ep(\d+)-iter(\d+)', filename)
        if match:
            ep = int(match.group(1))
            iter_idx = int(match.group(2))
            return ep, iter_idx
        return 0, 0
    return sorted(glob.glob(pattern, recursive=recursive), key=lambda x: extract_ep_iter(os.path.basename(x)), reverse=True)


def glob_with_global_step(pattern, recursive=False): 
    def extract_ep_iter(filename):
        match = re.search(r'global_step_(\d+)', filename)
        if match:
            iter_idx = int(match.group(1))
            return iter_idx
        return 0
    return sorted(glob.glob(pattern, recursive=recursive), key=lambda x: extract_ep_iter(os.path.basename(x)), reverse=True)
        
class TXTSaver(object):
    def __init__(self, is_master: bool, eval_milestone: List[Tuple[float, float]]):
        self.is_master = is_master
        self.time_stamp = torch.tensor([time.time() - 1e5, time.time()], device=dist.get_device())
        self.sp_also: subprocess.Popen = None
        self.sp_best: subprocess.Popen = None
        self.sp_backup: subprocess.Popen = None
        self.acc_str, self.eval_milestone = '[no acc str]', eval_milestone

    def add_prompt_below_cv2(self, img_np: np.ndarray, prompt: str, text_area_height=50) -> np.ndarray:
        """
        OpenCV용 NumPy 이미지 아래에 프롬프트 텍스트를 추가한 새 이미지를 반환합니다.
        
        Parameters:
            img_np (np.ndarray): (H, W, 3) 모양의 BGR 이미지 (uint8)
            prompt (str): 이미지 아래에 표시할 텍스트
            text_area_height (int): 이미지 아래쪽에 추가로 확보할 영역(세로 크기)
        
        Returns:
            np.ndarray: 텍스트가 붙은 최종 이미지 (BGR, uint8)
        """
        # 텍스트를 표시할 검은색 영역 만들기
        text_area = np.zeros((text_area_height, img_np.shape[1], 3), dtype=np.uint8)

        # OpenCV 폰트 설정
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)  # 흰색
        thickness = 2

        # 문자열의 시작 위치(10, text_area_height//2 정도에 배치)
        position = (10, text_area_height // 2)

        # 텍스트 그리기
        cv2.putText(text_area, prompt, position, font, font_scale, color, thickness, cv2.LINE_AA)
        
        # 원본 이미지 아래에 텍스트 영역을 붙여서 최종 이미지 생성
        return np.concatenate([img_np, text_area], axis=0)
    
    def clone_detach(self, weight):
        if isinstance(weight, list):
            return [w.clone().detach() for w in weight]
        else:
            return weight.clone().detach() 
    
    def sav(
        self, args: arg_util.Args, g_it: int, next_ep: int, next_it: int, trainer, generated_images, prompts,
        acc_str: Optional[str] = None, eval_milestone: Optional[List[Tuple[float, float]]] = None,
        also_save_to: str = None, best_save_to: str = None, only_img_save=False,
    ):
        self.time_stamp[1] = time.time()
        last_save_time, cur_time = self.time_stamp.cpu().tolist()
        
        auto_save = cur_time - last_save_time > 20 * 60
        need_save = also_save_to is not None or best_save_to is not None or next_ep == args.ep or auto_save
        if not need_save:
            return
        
        if acc_str is not None: self.acc_str = acc_str
        if eval_milestone is not None: self.eval_milestone = eval_milestone
        
        fname = f'txt_emb-giter{g_it//1000:03d}K-ep{next_ep}-iter{next_it}-last.pth' if args.gpt_training else f'ckpt-last.pth'
        iname = f'txt_emb-giter{g_it//1000:03d}K-ep{next_ep}-iter{next_it}-last.jpg' if args.gpt_training else f'ckpt-last.png'
        imgs_with_text = []
        num_imgs = len(generated_images)
        half = num_imgs // 2
        top_row = torch.cat(generated_images[:half], dim=1)
        bottom_row = torch.cat(generated_images[half:], dim=1)
        final_img = torch.cat([top_row, bottom_row], dim=0)
        final_img_np = final_img.detach().cpu().numpy()
        # dtype이 float형(0~1 범위)인 경우 0~255 범위로 변환
        if final_img_np.dtype != np.uint8:
            final_img_np = np.clip(final_img_np, 0, 1) * 255
            final_img_np = final_img_np.astype(np.uint8)
        cv2.imwrite(os.path.join(args.local_out_path, iname), final_img_np)

        local_out_ckpt = os.path.join(args.local_out_path, fname)
        
        if self.is_master and (not only_img_save):
            stt = time.time()
            torch.save({
                'args':         args.state_dict(),
                'gpt_training': args.gpt_training,
                'arch':         args.model if args.gpt_training else args.vv,
                'epoch':        next_ep,
                'iter':         next_it,
                'model':        trainer.gpt.state_dict(),
                'acc_str':      self.acc_str,
            }, local_out_ckpt)
            
            print(f'[BoothSaver][rank00] start: {also_save_to=} {best_save_to=} {(next_ep == args.ep)=} {auto_save=}  |  see {local_out_ckpt}', flush=True)
            print(f'[BoothSaver][rank00] dbg: {args.bed=}', flush=True)                
            cost = time.time() - stt
            print(f'[BoothSaver][rank00] cost: {cost:.2f}s', flush=True)

class CKPTSaver(object):
    def __init__(self, is_master: bool, eval_milestone: List[Tuple[float, float]]):
        self.is_master = is_master
        self.time_stamp = torch.tensor([time.time() - 1e5, time.time()], device=dist.get_device())
        self.sp_also: subprocess.Popen = None
        self.sp_best: subprocess.Popen = None
        self.sp_backup: subprocess.Popen = None
        self.acc_str, self.eval_milestone = '[no acc str]', eval_milestone
    
    def sav(
        self, args: arg_util.Args, g_it: int, next_ep: int, next_it: int, trainer,
        acc_str: Optional[str] = None, eval_milestone: Optional[List[Tuple[float, float]]] = None,
        also_save_to: str = None, best_save_to: str = None,
    ):
        self.time_stamp[1] = time.time()
        dist.broadcast(self.time_stamp, src_rank=0)
        last_save_time, cur_time = self.time_stamp.cpu().tolist()
        
        auto_save = cur_time - last_save_time > 20 * 60
        need_save = also_save_to is not None or best_save_to is not None or next_ep == args.ep or auto_save
        if not need_save:
            return
        
        if acc_str is not None: self.acc_str = acc_str
        if eval_milestone is not None: self.eval_milestone = eval_milestone
        
        fname = f'ar-ckpt-giter{g_it//1000:03d}K-ep{next_ep}-iter{next_it}-last.pth' if args.gpt_training else f'ckpt-last.pth'
        local_out_ckpt = os.path.join(args.local_out_path, fname)
        
        stt = time.time()
        torch.save({
            'args':         args.state_dict(),
            'gpt_training': args.gpt_training,
            'arch':         args.model if args.gpt_training else args.vv,
            'epoch':        next_ep,
            'iter':         next_it,
            'trainer':      trainer_state,
            'acc_str':      self.acc_str,
            'milestones':   self.eval_milestone,
        }, local_out_ckpt)
        
        print(f'[TXTSaver][rank00] start: {also_save_to=} {best_save_to=} {(next_ep == args.ep)=} {auto_save=}  |  see {local_out_ckpt}', flush=True)
        print(f'[TXTSaver][rank00] dbg: {args.bed=}', flush=True)                
        cost = time.time() - stt
        print(f'[TXTSaver][rank00] cost: {cost:.2f}s', flush=True)
        
        del trainer_state
        time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
        dist.barrier()
        

def auto_resume(args: arg_util.Args, pattern='ckpt*.pth') -> Tuple[List[str], int, int, str, List[Tuple[float, float]], dict, dict]:
    info = []
    resume = ''
    if args.auto_resume:
        for dd in (args.local_out_path, args.bed):
            all_ckpt = glob_with_epoch_iter(os.path.join(dd, pattern))
            if len(all_ckpt): break
        if len(all_ckpt) == 0:
            info.append(f'[auto_resume] no ckpt found @ {pattern}')
            info.append(f'[auto_resume quit]')
        else:
            resume = all_ckpt[0]
            info.append(f'[auto_resume] auto load from @ {resume} ...')
    else:
        info.append(f'[auto_resume] disabled')
        info.append(f'[auto_resume quit]')
    
    if len(resume) == 0:
        return info, 0, 0, '[no acc str]', [], {}, {}

    print(f'auto resume from {resume}')

    try:
        ckpt = torch.load(resume, map_location='cpu')
    except Exception as e:
        info.append(f'[auto_resume] failed, {e} @ {resume}')
        if len(all_ckpt) < 2:
            return info, 0, 0, '[no acc str]', [], {}, {}
        try: # another chance to load from bytenas
            ckpt = torch.load(all_ckpt[1], map_location='cpu')
        except Exception as e:
            info.append(f'[auto_resume] failed, {e} @ {all_ckpt[1]}')
            return info, 0, 0, '[no acc str]', [], {}, {}
    
    dist.barrier()
    ep, it = ckpt['epoch'], ckpt['iter']
    eval_milestone = ckpt.get('milestones', [])
    info.append(f'[auto_resume success] resume from ep{ep}, it{it},    eval_milestone: {eval_milestone}')
    return info, ep, it, ckpt.get('acc_str', '[no acc str]'), eval_milestone, ckpt['trainer'], ckpt['args']
