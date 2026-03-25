
import os
import argparse
import torch
import numpy as np
import scipy.io.wavfile as wav
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DNN_models.Complex_MTASS_model_Streaming import ComplexMTASSStreamingLightning
from DNN_models.Complex_MTASS_Streaming import StreamingComplexMTASS
from DNN_models.Complex_MTASS_Solver_Streaming import Complex_MTASS_model_Streaming

win_len = 512
win_inc = 256
fft_len = 512
fs = 16000


def masked_metric(estimate, target, eps=1e-8):
    target_energy = torch.sum(target ** 2, dim=-1)
    mask = target_energy > eps
    batch_size = estimate.shape[0]
    sdr_vector = torch.zeros(batch_size, device=estimate.device)
    sisdr_vector = torch.zeros(batch_size, device=estimate.device)
    if mask.sum() > 0:
        valid_est = estimate[mask]
        valid_tgt = target[mask]

        valid_sdr = sdr_cost(valid_est, valid_tgt)
        sdr_vector[mask] = valid_sdr

        valid_sisdr = sisdr_cost(valid_est, valid_tgt)
        sisdr_vector[mask] = valid_sisdr
    mask_float = mask.float()
    return sdr_vector, sisdr_vector, mask_float


def compute_out_cost(mix, Z1, Z2, Z3, R1, R2, R3):
    Z1_time = Inverse_STFT(Z1)
    Z2_time = Inverse_STFT(Z2)
    Z3_time = Inverse_STFT(Z3)

    sdr_s, sisdr_s, mask_s = masked_metric(Z1_time, R1)
    sdr_m, sisdr_m, mask_m = masked_metric(Z2_time, R2)
    sdr_n, sisdr_n, mask_n = masked_metric(Z3_time, R3)
    total_mask = torch.tensor([mask_s.item(), mask_m.item(), mask_n.item()])

    sum_sdr = sdr_s + sdr_m + sdr_n 
    sum_sisdr = sisdr_s + sisdr_m + sisdr_n

    num_tasks = mask_s + mask_m + mask_n
    num_tasks = torch.clamp(num_tasks, min=1.0)
    
    per_sample_sdr = sum_sdr / num_tasks
    per_sample_sisdr = sum_sisdr / num_tasks

    total_sdr = torch.mean(per_sample_sdr)
    total_sisdr = torch.mean(per_sample_sisdr)

    speech_sisdr = torch.mean(sisdr_s)
    music_sisdr = torch.mean(sisdr_m)
    others_sisdr = torch.mean(sisdr_n)
    return total_sdr, total_sisdr, Z1_time, Z2_time, Z3_time, speech_sisdr, music_sisdr, others_sisdr, total_mask


def sdr_standard(estimated, target, eps=1e-8):
    signal_pow = torch.sum(target ** 2, dim=-1) + eps
    noise = estimated - target
    noise_pow = torch.sum(noise ** 2, dim=-1) + eps
    sdr = 10 * torch.log10(signal_pow / noise_pow)
    return sdr


def sdr_cost(estimated, target, eps=1e-8):
    dot = torch.sum(estimated * target, dim=-1, keepdim=True)
    sign = torch.sign(dot) 
    estimated = estimated * sign

    est_norm = torch.norm(estimated, dim=-1, keepdim=True) + eps
    tgt_norm = torch.norm(target, dim=-1, keepdim=True) + eps
    estimated = estimated * (tgt_norm / est_norm)

    signal_pow = torch.sum(target ** 2, dim=-1) + eps
    noise = estimated - target
    noise_pow = torch.sum(noise ** 2, dim=-1) + eps
    sdr = 10 * torch.log10(signal_pow / noise_pow)
    return sdr


def sisdr_cost(estimated, target, eps=1e-8):
    dot = torch.sum(estimated * target, dim=-1, keepdim=True) + eps
    s_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    scale = dot / s_energy
    target_scaled = scale * target
    e_noise = estimated - target_scaled
    target_pow = torch.sum(target_scaled ** 2, dim=-1) + eps
    noise_pow = torch.sum(e_noise ** 2, dim=-1) + eps
    sisdr = 10 * torch.log10(target_pow / noise_pow)
    return sisdr.squeeze(-1)


def STFT(inputs):
    inputs = inputs.transpose(1, 2)
    inputs = inputs.reshape(-1, inputs.shape[2])
    window = torch.hamming_window(win_len, device=inputs.device)
    spec = torch.stft(
        inputs,
        n_fft=fft_len,
        hop_length=win_inc,
        win_length=win_len,
        window=window,
        center=False,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True
    )
    real = spec.real
    imag = spec.imag
    X = torch.cat((real, imag), 1)
    return X


def Inverse_STFT(inputs):
    cutoff = fft_len // 2 + 1
    real_part = inputs[:, :cutoff, :]
    imag_part = inputs[:, cutoff:, :]
    complex_spec = torch.complex(real_part, imag_part)
    istft_window = torch.hamming_window(win_len, device=inputs.device)
    reconstruction = torch.istft(
        complex_spec,
        n_fft=fft_len,
        hop_length=win_inc,
        win_length=win_len,
        window=istft_window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=False
    )
    return reconstruction


def wav_write(data, path, filename, fs):
    full_path = os.path.join(path, filename)
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    wav.write(full_path, fs, data)


def create_sisdr_string(total_mask, sisdr_values, labels, format_spec=".2f"):
    active_labels = [label for label, mask in zip(labels, total_mask.bool()) if mask]
    parts = []
    for label in active_labels:
        value = sisdr_values[label]
        parts.append(f"{label}{value:{format_spec}}")
    parts.append(f"{sisdr_values['total']:{format_spec}}")
    return "-".join(parts)


class WavDataset(Dataset):
    def __init__(self, wav_dir):
        self.wav_dir = wav_dir
        self.sample_dirs = []
        for item in sorted(os.listdir(wav_dir)):
            item_path = os.path.join(wav_dir, item)
            if os.path.isdir(item_path):
                mixture_path = os.path.join(item_path, 'mixture.wav')
                if os.path.exists(mixture_path):
                    self.sample_dirs.append(item_path)
    
    def __len__(self):
        return len(self.sample_dirs)
    
    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        
        mixture_path = os.path.join(sample_dir, 'mixture.wav')
        speech_gt_path = os.path.join(sample_dir, 'speech_gt.wav')
        music_gt_path = os.path.join(sample_dir, 'music_gt.wav')
        others_gt_path = os.path.join(sample_dir, 'others_gt.wav')
        
        fs_read, mixture_wav = wav.read(mixture_path)
        fs_read, speech_gt_wav = wav.read(speech_gt_path)
        fs_read, music_gt_wav = wav.read(music_gt_path)
        fs_read, others_gt_wav = wav.read(others_gt_path)
        
        if mixture_wav.dtype != np.float32:
            mixture_wav = mixture_wav.astype(np.float32) / 32767.0
            speech_gt_wav = speech_gt_wav.astype(np.float32) / 32767.0
            music_gt_wav = music_gt_wav.astype(np.float32) / 32767.0
            others_gt_wav = others_gt_wav.astype(np.float32) / 32767.0
        
        if len(mixture_wav.shape) > 1:
            mixture_wav = mixture_wav[:, 0]
            speech_gt_wav = speech_gt_wav[:, 0]
            music_gt_wav = music_gt_wav[:, 0]
            others_gt_wav = others_gt_wav[:, 0]
        
        mixture_tensor = torch.from_numpy(mixture_wav).float().unsqueeze(0).unsqueeze(0)
        X1 = STFT(mixture_tensor)
        
        speech_gt_tensor = torch.from_numpy(speech_gt_wav).float()
        music_gt_tensor = torch.from_numpy(music_gt_wav).float()
        others_gt_tensor = torch.from_numpy(others_gt_wav).float()
        
        return X1.squeeze(0), speech_gt_tensor, music_gt_tensor, others_gt_tensor, idx


def test_non_streaming(args, model, test_loader, device):
    print("="*60)
    print("Testing with NON-STREAMING inference (reference)...")
    print("="*60)
    
    labels = ["speech", "music", "others"]
    total_sdr_list = []
    total_sisdr_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Non-streaming"):
            X1, speech_gt, music_gt, others_gt, idx = batch
            X1 = X1.to(device)
            
            R_gt_speech = speech_gt.to(device)
            R_gt_music  = music_gt.to(device)
            R_gt_others  = others_gt.to(device)

            Z1, Z2, Z3 = model(X1)
            mixture = Inverse_STFT(X1)
            total_sdr, total_sisdr, Z1_time, Z2_time, Z3_time, speech_sisdr, music_sisdr, others_sisdr, total_mask = compute_out_cost(mixture, Z1, Z2, Z3, R_gt_speech, R_gt_music, R_gt_others)
            
            total_sdr_list.append(total_sdr)
            total_sisdr_list.append(total_sisdr)

    avg_sdr = torch.mean(torch.tensor(total_sdr_list))
    avg_sisdr = torch.mean(torch.tensor(total_sisdr_list))

    print(f"Non-streaming Total SDR: {avg_sdr:.4f}")
    print(f"Non-streaming Total SI-SDR: {avg_sisdr:.4f}")
    
    return avg_sdr, avg_sisdr


def test_streaming(args, model, test_loader, device, chunk_size=1):
    print("="*60)
    print(f"Testing with STREAMING inference (chunk_size={chunk_size})...")
    print("="*60)
    
    labels = ["speech", "music", "others"]
    total_sdr_list = []
    total_sisdr_list = []
    
    inference_times = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Streaming (chunk={chunk_size})"):
            X1, speech_gt, music_gt, others_gt, idx = batch
            X1 = X1.to(device)
            
            R_gt_speech = speech_gt.to(device)
            R_gt_music  = music_gt.to(device)
            R_gt_others  = others_gt.to(device)

            model.reset_streaming_state()
            
            import time
            start_time = time.time()
            Z1, Z2, Z3 = Complex_MTASS_model_Streaming.streaming_separation(
                model.model, X1, chunk_size=chunk_size
            )
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            mixture = Inverse_STFT(X1)
            total_sdr, total_sisdr, Z1_time, Z2_time, Z3_time, speech_sisdr, music_sisdr, others_sisdr, total_mask = compute_out_cost(mixture, Z1, Z2, Z3, R_gt_speech, R_gt_music, R_gt_others)
            
            sisdr_values = {
                'total': total_sisdr,
                'speech': speech_sisdr,
                'music': music_sisdr,
                'others': others_sisdr
            }
            total_sdr_list.append(total_sdr)
            total_sisdr_list.append(total_sisdr)

            if idx.item() < args.save_n_samples:
                result_label = create_sisdr_string(total_mask, sisdr_values, labels, ".2f")
                save_dir = os.path.join(args.output_dir, f"sample{idx.item()}_{result_label}_streaming")
                os.makedirs(save_dir, exist_ok=True)
                wav_write(mixture.squeeze(), save_dir, "mixture.wav", fs)
                wav_write(R_gt_speech.squeeze(), save_dir, "speech_gt.wav", fs)
                wav_write(R_gt_music.squeeze(), save_dir, "music_gt.wav", fs)
                wav_write(R_gt_others.squeeze(), save_dir, "others_gt.wav", fs)
                wav_write(Z1_time.squeeze(), save_dir, "speech_es.wav", fs)
                wav_write(Z2_time.squeeze(),  save_dir,  "music_es.wav", fs)
                wav_write(Z3_time.squeeze(),  save_dir,  "others_es.wav", fs)

    avg_sdr = torch.mean(torch.tensor(total_sdr_list))
    avg_sisdr = torch.mean(torch.tensor(total_sisdr_list))
    avg_time = np.mean(inference_times)

    print(f"Streaming Total SDR: {avg_sdr:.4f}")
    print(f"Streaming Total SI-SDR: {avg_sisdr:.4f}")
    print(f"Average inference time per sample: {avg_time*1000:.2f} ms")
    
    return avg_sdr, avg_sisdr, avg_time


def test(args):
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")
    
    model = ComplexMTASSStreamingLightning.load_from_checkpoint(
        args.ckpt_path,
        model_class=StreamingComplexMTASS,
        loss_class=Complex_MTASS_model_Streaming,
    )
    model.to(device)
    model.eval()
    model.freeze()

    test_dataset = WavDataset(args.wav_dir)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,
        shuffle=False, 
        num_workers=1
    )

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.test_non_streaming:
        non_streaming_sdr, non_streaming_sisdr = test_non_streaming(args, model, test_loader, device)
    
    streaming_sdr, streaming_sisdr, avg_time = test_streaming(args, model, test_loader, device, chunk_size=args.chunk_size)
    
    if args.test_non_streaming:
        print("\n" + "="*60)
        print("COMPARISON:")
        print("="*60)
        print(f"Non-streaming SI-SDR: {non_streaming_sisdr:.4f}")
        print(f"Streaming SI-SDR:     {streaming_sisdr:.4f}")
        print(f"Difference:           {abs(non_streaming_sisdr - streaming_sisdr):.4f}")
        if abs(non_streaming_sisdr - streaming_sisdr) < 0.1:
            print("✓ Streaming and non-streaming results match!")
        else:
            print("⚠ Warning: Results differ significantly")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Streaming Test for Complex MTASS Model from WAV files')
    parser.add_argument('--wav_dir', type=str, required=True, help='Directory with test wav files (each sample in subdir)')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint .ckpt')
    parser.add_argument('--output_dir', type=str, default='./test_wav_streaming_results', help='Folder to save wavs')
    parser.add_argument('--num_sources', type=int, choices=[2, 3, 4, 5], required=True,
                       help='混合声源数量: 2, 3, 4 或 5')
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--chunk_size', type=int, default=1, 
                       help='Number of frames to process at once in streaming mode (default: 1 for frame-by-frame)')
    parser.add_argument('--test_non_streaming', action='store_true', default=True,
                       help='Also test non-streaming inference for comparison')
    parser.add_argument('--save_n_samples', type=int, default=10,
                       help='Number of samples to save audio files for')
    
    args = parser.parse_args()

    test(args)

