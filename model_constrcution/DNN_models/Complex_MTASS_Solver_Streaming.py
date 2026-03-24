

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from thop import profile
from thop import clever_format
import wave
import struct
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
import scipy.signal as signal
import os
import gc
import datetime
import time
import random
import glob
from tqdm import tqdm
import pandas as pd
import itertools
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("..")
from utils.utils_library_gpu import *
from DNN_models.Complex_MTASS_Streaming import *





# -----------------------------------------------------------------------------------------------------------------------------
# * Class:
#     Complex_MTASS_model_Streaming---Implements a Complex-domain MTASS model for speech, noise and music separation 
#                                       with Streaming (Causal) Self-Attention support
# 
# * Note:
#   The Complex MTASS model takes the mixture Mag feratures as the inputs 
#   and outputs the complex ratio masks (cRMs).
#   In this model, 8 sub-bands are divided and performed the multi-scale analysis.
#   **Streaming version supports frame-by-frame inference for real-time applications.**
#   
#
# * Copyright and Authors:
#    Writen by Mr. Wind at Harbin Institute of Technology, Shenzhen.
#    Contact Email: zhanglu_wind@163.com
#
# * Streaming Modifications:
#    - Added streaming inference support with state management
#    - Added causal mask to self-attention for training
#    - Added frame-by-frame processing utilities
# -----------------------------------------------------------------------------------------------------------------------------

class Complex_MTASS_model_Streaming:

    #-------------------------------------------------------------------------------------------------------------------
    # * Functions:
    #     model_description()-- print the description of model and the information of training data 
    #        * Arguments:
    #            * train_datain_path1 -- train data path of input
    #            * train_datain_list1 -- train data list of input
    #            * dev_datain_path1 -- dev data path of input
    #            * dev_datain_list1 -- dev data list of output
    #            * mini_batch_size -- the size of each mini_batch
    #        * Returns:
    #            * m_x1_train -- input feature size, shape [0]
    #            * n_x1_train -- input feature size, shape [1]
    #            * num_minibatches_train -- the toal numbers of training data
    #            * num_minibatches_dev -- the total numbers of dev data
    #
    #---------------------------------------------------------------------------------------------------------------------

    def model_description(train_datain_path1,train_datain_list1,dev_datain_path1,dev_datain_list1,mini_batch_size):
        ### START CODE HERE ###
        print('The Complex MTASS learning structure (Mag_to_Com, Residual Compensation, F-MSE+T-SNR) is : 257+ComplexMSTCN(15)+3*GTCN(5,8)+(514,514,514)')
        print('The Complex MTASS model is trained to separate three targets!') 
        print('*** Streaming Version with Causal Self-Attention ***')
        print('The sizes of each train/dev file are as follows:')
        num_minibatches_train = 0
        num_minibatches_dev = 0        
        for train_datain in train_datain_list1:
            path = train_datain_path1 + os.sep + train_datain
            data = np.load(path)
            # print('n_x_train', n_x_train)
            num_minibatches_train += math.floor(data.shape[0] / mini_batch_size)
            split_sentence_len = data.shape[2] 

        for dev_datain in dev_datain_list1:
            path = dev_datain_path1 + os.sep + dev_datain
            data = np.load(path)
            # print('n_x_dev', n_x_dev)
            num_minibatches_dev += math.floor(data.shape[0] / mini_batch_size)
                
        print('The mini_batch size is:', mini_batch_size)
        print('num_minibatches_train:', num_minibatches_train)
        print('num_minibatches_dev:', num_minibatches_dev)
    
        del data
        gc.collect()
        return num_minibatches_train, num_minibatches_dev, split_sentence_len 

    def masked_sisdr_loss(estimate, target, eps=1e-8):
        target_energy = torch.sum(target ** 2, dim=-1)
        mask = target_energy > eps
        batch_size = estimate.shape[0]
        loss_vector = torch.zeros(batch_size, device=estimate.device)
        if mask.sum() > 0:
            valid_est = estimate[mask]
            valid_tgt = target[mask]
            valid_sisdr = Complex_MTASS_model_Streaming.sisdr_cost(valid_est, valid_tgt)
            loss_vector[mask] = -valid_sisdr
        mask_float = mask.float()  
        return loss_vector, mask_float
  
    def compute_out_cost(Z1, Z2, Z3, Y_targets, R_targets):
        ### START CODE HERE ###
        win_len = 512
        win_inc = 256 # frame shift
        fft_len = 512
        Z1_time = Complex_MTASS_model_Streaming.Inverse_STFT(Z1, win_len, win_inc, fft_len)
        Z2_time = Complex_MTASS_model_Streaming.Inverse_STFT(Z2, win_len, win_inc, fft_len)
        Z3_time = Complex_MTASS_model_Streaming.Inverse_STFT(Z3, win_len, win_inc, fft_len)
        Y1, Y2, Y3 = Y_targets[0], Y_targets[1], Y_targets[2]
        R1, R2, R3 = R_targets[0], R_targets[1], R_targets[2]
        
        mse_cost = torch.nn.MSELoss()
        cost_freq = mse_cost(Z1, Y1) + mse_cost(Z2, Y2) + mse_cost(Z3, Y3)

        loss_s, mask_s = Complex_MTASS_model_Streaming.masked_sisdr_loss(Z1_time, R1)
        loss_m, mask_m = Complex_MTASS_model_Streaming.masked_sisdr_loss(Z2_time, R2)
        loss_o, mask_o = Complex_MTASS_model_Streaming.masked_sisdr_loss(Z3_time, R3)
        sum_loss = loss_s + loss_m + loss_o
        num_tasks = mask_s + mask_m + mask_o
        num_tasks = torch.clamp(num_tasks, min=1.0)
        per_sample_loss = sum_loss / num_tasks
        cost_time_sisdr = torch.mean(per_sample_loss)
        total_cost = cost_freq + cost_time_sisdr

        return total_cost, cost_freq, cost_time_sisdr

    def sisdr_cost(estimated, target, eps=1e-8):
        dot = torch.sum(estimated * target, dim=-1, keepdim=True)
        s_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps

        scale = dot / s_energy
        target_scaled = scale * target
        e_noise = estimated - target_scaled
        target_pow = torch.sum(target_scaled ** 2, dim=-1) + eps
        noise_pow = torch.sum(e_noise ** 2, dim=-1) + eps
        
        sisdr = 10 * torch.log10(target_pow / noise_pow)
        return sisdr.squeeze(-1)

    def SNR_cost(Z1,Y1,eps=1e-8):
        # Z1.shape=[-1,sen_len]
        # Y1.shape=[-1,sen_len]

        snr = torch.sum(Y1**2, dim=1, keepdim=True) / (torch.sum((Z1 - Y1)**2, dim=1, keepdim=True)+eps)
        loss = -10*torch.log10(snr + eps).mean()
        
        return loss


    def Inverse_STFT(inputs, win_len, win_hop, fft_len):
        # inputs.shape = [B,fea_size,sen_len] (Complex STFT)
        cutoff = fft_len // 2 + 1
        real_part = inputs[:, :cutoff, :]
        imag_part = inputs[:, cutoff:, :]

        complex_spec = torch.complex(real_part, imag_part)
        #istft_window = torch.ones(win_len, device=inputs.device)
        istft_window = torch.hamming_window(win_len, device=inputs.device)

        reconstruction = torch.istft(
            complex_spec,
            n_fft=fft_len,
            hop_length=win_hop,
            win_length=win_len,
            window=istft_window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=False 
        )
        
        return reconstruction
    
    #-------------------------------------------------------------------------------------------------------------------
    # Streaming Inference Utilities
    #-------------------------------------------------------------------------------------------------------------------
    
    def streaming_separation(model, input_frames, chunk_size=1):
        """
        Perform streaming separation on input frames.
        
        Args:
            model: StreamingComplexMTASS model instance
            input_frames: Input STFT frames of shape [batch, 514, num_frames]
            chunk_size: Number of frames to process at once (default: 1 for frame-by-frame)
        
        Returns:
            y1_out, y2_out, y3_out: Separated speech, music, others [batch, 514, num_frames]
        """
        model.eval()
        model.reset_state()
        
        batch_size, feature_dim, total_frames = input_frames.shape
        
        y1_list, y2_list, y3_list = [], [], []
        
        with torch.no_grad():
            for start_idx in range(0, total_frames, chunk_size):
                end_idx = min(start_idx + chunk_size, total_frames)
                chunk = input_frames[:, :, start_idx:end_idx]
                
                y1_chunk, y2_chunk, y3_chunk = model.streaming_forward(chunk)
                
                y1_list.append(y1_chunk)
                y2_list.append(y2_chunk)
                y3_list.append(y3_chunk)
        
        y1_out = torch.cat(y1_list, dim=-1)
        y2_out = torch.cat(y2_list, dim=-1)
        y3_out = torch.cat(y3_list, dim=-1)
        
        return y1_out, y2_out, y3_out
    
    def streaming_separation_with_overlap_add(model, mixture_signal, win_len=512, win_inc=256, 
                                               fft_len=512, chunk_size=1, device='cpu'):
        """
        Complete streaming separation pipeline from time-domain signal to separated signals.
        
        Args:
            model: StreamingComplexMTASS model instance
            mixture_signal: Input time-domain signal [batch, num_samples]
            win_len: Window length for STFT
            win_inc: Frame shift for STFT
            fft_len: FFT length
            chunk_size: Number of frames to process at once
            device: Device to run on
        
        Returns:
            speech, music, others: Separated time-domain signals [batch, num_samples]
        """
        model.eval()
        model.to(device)
        model.reset_state()
        
        batch_size, num_samples = mixture_signal.shape
        
        # Create window
        window = torch.hamming_window(win_len, device=device)
        
        # Compute STFT
        mixture_signal = mixture_signal.to(device)
        
        # Pad signal for STFT
        pad_length = win_len
        mixture_padded = F.pad(mixture_signal, (pad_length, pad_length), mode='constant', value=0.0)
        
        # Compute STFT
        complex_spec = torch.stft(
            mixture_padded,
            n_fft=fft_len,
            hop_length=win_inc,
            win_length=win_len,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True
        )
        
        # Convert to RI format
        real_part = complex_spec.real
        imag_part = complex_spec.imag
        ri_input = torch.cat([real_part, imag_part], dim=1)  # [batch, 514, num_frames]
        
        # Streaming separation
        y1_ri, y2_ri, y3_ri = Complex_MTASS_model_Streaming.streaming_separation(
            model, ri_input, chunk_size=chunk_size
        )
        
        # Convert back to complex and perform ISTFT
        def ri_to_time(ri_spec):
            cutoff = fft_len // 2 + 1
            real = ri_spec[:, :cutoff, :]
            imag = ri_spec[:, cutoff:, :]
            complex_out = torch.complex(real, imag)
            
            time_sig = torch.istft(
                complex_out,
                n_fft=fft_len,
                hop_length=win_inc,
                win_length=win_len,
                window=window,
                center=False,
                normalized=False,
                onesided=True,
                return_complex=False,
                length=num_samples + 2 * pad_length
            )
            return time_sig[:, pad_length:pad_length+num_samples]
        
        speech = ri_to_time(y1_ri)
        music = ri_to_time(y2_ri)
        others = ri_to_time(y3_ri)
        
        return speech, music, others
