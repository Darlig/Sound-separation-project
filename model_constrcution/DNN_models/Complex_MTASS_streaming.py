

import math
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from thop import profile
from thop import clever_format
import wave
import struct
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
import scipy.signal as signal
from scipy.signal import get_window
import os
import gc
import datetime
import random
from tqdm import tqdm

import sys
sys.path.append("..")
from utils.utils_library_gpu import *


class Complex_MTASS_Streaming(nn.Module):
    def __init__(self, is_causal=True):
        super(Complex_MTASS_Streaming, self).__init__()
        
        self.is_causal = is_causal
        self.stage1_alpha = False
        self.stage2_beta = False
        
        self.conv1d_1 = Conv_layer_1d(257, 1028)
        self.conv1d_2 = Conv_layer_1d(1028, 1028)
        self.conv1d_3 = nn.Conv1d(1028, 514, kernel_size=1)
        self.conv1d_4 = nn.Conv1d(1028, 514, kernel_size=1)
        self.conv1d_5 = nn.Conv1d(1028, 514, kernel_size=1)
        self.ms_resblock_1 = MS_ResBlock(3, 1, is_causal=self.is_causal)
        self.ms_resblock_2 = MS_ResBlock(3, 3, is_causal=self.is_causal)
        self.ms_resblock_3 = MS_ResBlock(3, 5, is_causal=self.is_causal)
        self.ms_resblock_4 = MS_ResBlock(3, 7, is_causal=self.is_causal)
        self.ms_resblock_5 = MS_ResBlock(3, 11, is_causal=self.is_causal)
        self.ms_resblock_6 = MS_ResBlock(3, 1, is_causal=self.is_causal)
        self.ms_resblock_7 = MS_ResBlock(3, 3, is_causal=self.is_causal)
        self.ms_resblock_8 = MS_ResBlock(3, 5, is_causal=self.is_causal)
        self.ms_resblock_9 = MS_ResBlock(3, 7, is_causal=self.is_causal)
        self.ms_resblock_10 = MS_ResBlock(3, 11, is_causal=self.is_causal)
        self.ms_resblock_11 = MS_ResBlock(3, 1, is_causal=self.is_causal)
        self.ms_resblock_12 = MS_ResBlock(3, 3, is_causal=self.is_causal)
        self.ms_resblock_13 = MS_ResBlock(3, 5, is_causal=self.is_causal)
        self.ms_resblock_14 = MS_ResBlock(3, 7, is_causal=self.is_causal)
        self.ms_resblock_15 = MS_ResBlock(3, 11, is_causal=self.is_causal)
        
        self.speech_res_block = GTCN(5, 8, is_causal=self.is_causal)
        self.music_res_block = GTCN(5, 8, is_causal=self.is_causal)
        self.others_res_block = GTCN(5, 8, is_causal=self.is_causal)

    def forward(self, X1):
        if self.stage1_alpha is True:
            with torch.no_grad():
                x_real = torch.unsqueeze(X1[:, :257, :], 1)
                x_imag = torch.unsqueeze(X1[:, 257:, :], 1)
                x_ri = torch.cat((x_real, x_imag), 1)
                x_mag = torch.norm(x_ri, dim=1)
                
                x = self.conv1d_1(x_mag)
                x = self.ms_resblock_1(x, x_mag)
                x = self.ms_resblock_2(x, x_mag)
                x = self.ms_resblock_3(x, x_mag)
                x = self.ms_resblock_4(x, x_mag)
                x = self.ms_resblock_5(x, x_mag)
                x = self.ms_resblock_6(x, x_mag)
                x = self.ms_resblock_7(x, x_mag)
                x = self.ms_resblock_8(x, x_mag)
                x = self.ms_resblock_9(x, x_mag)
                x = self.ms_resblock_10(x, x_mag)
                x = self.ms_resblock_11(x, x_mag)
                x = self.ms_resblock_12(x, x_mag)
                x = self.ms_resblock_13(x, x_mag)
                x = self.ms_resblock_14(x, x_mag)
                x = self.ms_resblock_15(x, x_mag)
                x = self.conv1d_2(x)
                
                y1_mask = self.conv1d_3(x)
                y2_mask = self.conv1d_4(x)
                y3_mask = self.conv1d_5(x)
                y1_RI = y1_mask * X1
                y2_RI = y2_mask * X1
                y3_RI = y3_mask * X1
        else:
            x_real = torch.unsqueeze(X1[:, :257, :], 1)
            x_imag = torch.unsqueeze(X1[:, 257:, :], 1)
            x_ri = torch.cat((x_real, x_imag), 1)
            x_mag = torch.norm(x_ri, dim=1)
            
            x = self.conv1d_1(x_mag)
            x = self.ms_resblock_1(x, x_mag)
            x = self.ms_resblock_2(x, x_mag)
            x = self.ms_resblock_3(x, x_mag)
            x = self.ms_resblock_4(x, x_mag)
            x = self.ms_resblock_5(x, x_mag)
            x = self.ms_resblock_6(x, x_mag)
            x = self.ms_resblock_7(x, x_mag)
            x = self.ms_resblock_8(x, x_mag)
            x = self.ms_resblock_9(x, x_mag)
            x = self.ms_resblock_10(x, x_mag)
            x = self.ms_resblock_11(x, x_mag)
            x = self.ms_resblock_12(x, x_mag)
            x = self.ms_resblock_13(x, x_mag)
            x = self.ms_resblock_14(x, x_mag)
            x = self.ms_resblock_15(x, x_mag)
            x = self.conv1d_2(x)
            
            y1_mask = self.conv1d_3(x)
            y2_mask = self.conv1d_4(x)
            y3_mask = self.conv1d_5(x)
            y1_RI = y1_mask * X1
            y2_RI = y2_mask * X1
            y3_RI = y3_mask * X1
        
        if self.stage2_beta is True:
            with torch.no_grad():
                y1_Res_in = X1 - y1_RI
                y2_Res_in = X1 - y2_RI
                y3_Res_in = X1 - y3_RI
                
                y1_Res_out = self.speech_res_block(y1_Res_in)
                y2_Res_out = self.music_res_block(y2_Res_in)
                y3_Res_out = self.others_res_block(y3_Res_in)
                
                y1_RI_out = y1_RI + y1_Res_out
                y2_RI_out = y2_RI + y2_Res_out
                y3_RI_out = y3_RI + y3_Res_out
        else:
            y1_Res_in = X1 - y1_RI
            y2_Res_in = X1 - y2_RI
            y3_Res_in = X1 - y3_RI
            
            y1_Res = self.speech_res_block(y1_Res_in)
            y2_Res = self.music_res_block(y2_Res_in)
            y3_Res = self.others_res_block(y3_Res_in)
            
            y1_RI_out = y1_RI + y1_Res
            y2_RI_out = y2_RI + y2_Res
            y3_RI_out = y3_RI + y3_Res

        return y1_RI_out, y2_RI_out, y3_RI_out


class MS_ResBlock(nn.Module):
    def __init__(self, k, dilation, is_causal=True):
        super(MS_ResBlock, self).__init__()
        self.k = k
        self.dilation = dilation
        self.is_causal = is_causal
        self.conv1d_1 = Conv_layer_1d(1028, 257)
        self.conv1d_2 = Conv_layer_1d(514, 1028)
        self.ms_conv1d = MS_dilated_layer_514d(self.k, self.dilation, is_causal=self.is_causal)

    def forward(self, prev_x, forw_x):
        x = self.conv1d_1(prev_x)
        x = torch.cat((forw_x, x), 1)
        x = self.ms_conv1d(x)
        x = self.conv1d_2(x)
        x = x + prev_x
        return x


class Conv_layer_1d(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Conv_layer_1d, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(output_dim),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.conv1d(x)


class Conv1d_sub(nn.Module):
    def __init__(self, input_dim, output_dim, k, dila, is_causal=True):
        super(Conv1d_sub, self).__init__()
        self.input_dim, self.output_dim, self.k, self.dila = input_dim, output_dim, k, dila
        self.is_causal = is_causal
        
        if self.is_causal:
            pad = nn.ConstantPad1d((2 * self.dila, 0), value=0.)
        else:
            pad = nn.ConstantPad1d((2 * self.dila // 2, 2 * self.dila // 2), value=0.)
        
        self.unit = nn.Sequential(
            pad,
            nn.Conv1d(self.input_dim, self.output_dim, self.k, dilation=self.dila),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(self.output_dim),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.unit(x)
        return x


class MS_dilated_layer_514d(nn.Module):
    def __init__(self, k, dilas, is_causal=True):
        super(MS_dilated_layer_514d, self).__init__()
        self.k, self.dilas = k, dilas
        self.is_causal = is_causal
        self.sub_groups = 8
        left_unit_list, right_unit_list = [], []
        
        for i in range(self.sub_groups):
            if i == 0:
                left_unit_list.append(Conv1d_sub(64, 64, self.k, self.dilas, is_causal=self.is_causal))
            elif (i == 1) or (i == 2) or (i == 5) or (i == 6):
                left_unit_list.append(Conv1d_sub(128, 64, self.k, self.dilas, is_causal=self.is_causal))
            elif (i == 3) or (i == 7):
                left_unit_list.append(Conv1d_sub(129, 65, self.k, self.dilas, is_causal=self.is_causal))
            else:
                left_unit_list.append(Conv1d_sub(129, 64, self.k, self.dilas, is_causal=self.is_causal))
        
        for i in range(self.sub_groups):
            if i == 7:
                right_unit_list.append(Conv1d_sub(65, 65, self.k, self.dilas, is_causal=self.is_causal))
            elif (i == 6) or (i == 2):
                right_unit_list.append(Conv1d_sub(129, 64, self.k, self.dilas, is_causal=self.is_causal))
            elif (i == 5) or (i == 4) or (i == 1) or (i == 0):
                right_unit_list.append(Conv1d_sub(128, 64, self.k, self.dilas, is_causal=self.is_causal))
            else:
                right_unit_list.append(Conv1d_sub(129, 65, self.k, self.dilas, is_causal=self.is_causal))
        
        self.left_unit_list, self.right_unit_list = nn.ModuleList(left_unit_list), nn.ModuleList(right_unit_list)

    def forward(self, inpt):
        num_subs = 8
        s0 = inpt[:, :64, :]
        s1 = inpt[:, 64:128, :]
        s2 = inpt[:, 128:192, :]
        s3 = inpt[:, 192:257, :]
        s4 = inpt[:, 257:321, :]
        s5 = inpt[:, 321:385, :]
        s6 = inpt[:, 385:449, :]
        s7 = inpt[:, 449:514, :]
        
        Left_subconv_out0 = self.left_unit_list[0](s0)
        s = torch.cat((Left_subconv_out0, s1), 1)
        Left_subconv_out1 = self.left_unit_list[1](s)
        s = torch.cat((Left_subconv_out1, s2), 1)
        Left_subconv_out2 = self.left_unit_list[2](s)
        s = torch.cat((Left_subconv_out2, s3), 1)
        Left_subconv_out3 = self.left_unit_list[3](s)
        s = torch.cat((Left_subconv_out3, s4), 1)
        Left_subconv_out4 = self.left_unit_list[4](s)
        s = torch.cat((Left_subconv_out4, s5), 1)
        Left_subconv_out5 = self.left_unit_list[5](s)
        s = torch.cat((Left_subconv_out5, s6), 1)
        Left_subconv_out6 = self.left_unit_list[6](s)
        s = torch.cat((Left_subconv_out6, s7), 1)
        Left_subconv_out7 = self.left_unit_list[7](s)
        
        Right_subconv_out7 = self.right_unit_list[7](s7)
        s = torch.cat((s6, Right_subconv_out7), 1)
        Right_subconv_out6 = self.right_unit_list[6](s)
        s = torch.cat((s5, Right_subconv_out6), 1)
        Right_subconv_out5 = self.right_unit_list[5](s)
        s = torch.cat((s4, Right_subconv_out5), 1)
        Right_subconv_out4 = self.right_unit_list[4](s)
        s = torch.cat((s3, Right_subconv_out4), 1)
        Right_subconv_out3 = self.right_unit_list[3](s)
        s = torch.cat((s2, Right_subconv_out3), 1)
        Right_subconv_out2 = self.right_unit_list[2](s)
        s = torch.cat((s1, Right_subconv_out2), 1)
        Right_subconv_out1 = self.right_unit_list[1](s)
        s = torch.cat((s0, Right_subconv_out1), 1)
        Right_subconv_out0 = self.right_unit_list[0](s)
        
        subconv_out0 = Left_subconv_out0 + Right_subconv_out0
        subconv_out1 = Left_subconv_out1 + Right_subconv_out1
        subconv_out2 = Left_subconv_out2 + Right_subconv_out2
        subconv_out3 = Left_subconv_out3 + Right_subconv_out3
        subconv_out4 = Left_subconv_out4 + Right_subconv_out4
        subconv_out5 = Left_subconv_out5 + Right_subconv_out5
        subconv_out6 = Left_subconv_out6 + Right_subconv_out6
        subconv_out7 = Left_subconv_out7 + Right_subconv_out7
        
        subconv_out = torch.cat((subconv_out0, subconv_out1, subconv_out2, subconv_out3,
                                  subconv_out4, subconv_out5, subconv_out6, subconv_out7), 1)
        
        return subconv_out


class GTCN(nn.Module):
    def __init__(self, repeats, num_blocks, is_causal=True):
        super(GTCN, self).__init__()
        self.is_causal = is_causal
        self.conv1d_in = nn.Conv1d(514, 256, kernel_size=1)
        self.tcm_list = nn.ModuleList([TCM_list(num_blocks, is_causal=self.is_causal) for _ in range(repeats)])
        self.conv1d_out = nn.Conv1d(256, 514, kernel_size=1)

    def forward(self, inpt):
        x = self.conv1d_in(inpt)
        for i in range(len(self.tcm_list)):
            x = self.tcm_list[i](x)
        x = self.conv1d_out(x)
        return x


class TCM_list(nn.Module):
    def __init__(self, X, is_causal=True):
        super(TCM_list, self).__init__()
        self.X = X
        self.tcm_list = nn.ModuleList([GLU(2 ** i, is_causal=is_causal) for i in range(self.X)])

    def forward(self, x):
        for i in range(self.X):
            x = self.tcm_list[i](x)
        return x


def self_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-2)
    query = query.transpose(-2, -1)
    scores = torch.matmul(query, key) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    value = value.transpose(-2, -1)
    attn_out = torch.matmul(p_attn, value)
    attn_out = attn_out.transpose(-2, -1)
    
    return attn_out


class GLU(nn.Module):
    def __init__(self, dilation, is_causal=True):
        super(GLU, self).__init__()
        self.apply_self_attn = False
        self.is_causal = is_causal
        self.dropout = nn.Dropout(0.1)

        if self.is_causal:
            pad = nn.ConstantPad1d((2 * dilation, 0), value=0.)
        else:
            pad = nn.ConstantPad1d((2 * dilation // 2, 2 * dilation // 2), value=0.)
        
        if self.apply_self_attn == False:
            self.in_conv = nn.Conv1d(256, 64, kernel_size=1, bias=False)
            self.left_conv = nn.Sequential(
                pad,
                nn.Conv1d(64, 64, kernel_size=3, dilation=dilation, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(64),
                nn.Dropout(0.2)
            )
            self.right_conv = nn.Sequential(
                pad,
                nn.Conv1d(64, 64, kernel_size=3, dilation=dilation, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(64),
                nn.Dropout(0.2),
                nn.Sigmoid()
            )
            self.out_conv = nn.Conv1d(64, 256, kernel_size=1, bias=False)
        
        if self.apply_self_attn == True:
            self.in_conv = nn.Sequential(
                nn.Conv1d(256, 64, kernel_size=1, bias=False),
                nn.ReLU(64),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2)
            )
            self.query_conv = nn.Conv1d(64, 64, kernel_size=1, bias=False)
            self.key_conv = nn.Conv1d(64, 64, kernel_size=1, bias=False)
            self.dilated_conv = nn.Sequential(
                pad,
                nn.ReLU(64),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Conv1d(64, 64, kernel_size=3, dilation=dilation, bias=False)
            )
            self.out_conv = nn.Conv1d(64, 256, kernel_size=1, bias=False)

    def forward(self, x):
        if self.apply_self_attn == False:
            resi = x
            x = self.in_conv(x)
            x = self.left_conv(x) * self.right_conv(x)
            x = self.out_conv(x)
            x = x + resi
        else:
            resi = x
            x = self.in_conv(x)
            query = self.query_conv(x)
            key = self.key_conv(x)
            value = x
            x = self_attention(query, key, value, mask=None, dropout=self.dropout)
            x = self.dilated_conv(x)
            x = self.out_conv(x)
            x = x + resi
        
        return x


def test_GTCN():
    x = torch.rand(1, 514, 1000)
    nnet = GTCN(5, 8, is_causal=True)
    x1 = nnet(x)
    print(x1.shape)


def test_Complex_MTASS_Streaming():
    x = torch.rand(1, 514, 1000)
    nnet = Complex_MTASS_Streaming(is_causal=True)
    x1, x2, x3 = nnet(x)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    
    Macs, params = profile(nnet, inputs=(x,))
    Macs, params = clever_format([Macs, params], "%.3f")
    print('Model Summary:')
    print('Trainable params of the model is:', params)
    print('MAC of the model is:', Macs)


if __name__ == "__main__":
    test_Complex_MTASS_Streaming()
