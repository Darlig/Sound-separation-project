#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np
import scipy.io.wavfile as wav
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DNN_models.Complex_MTASS_model import ComplexMTASSLightning
from DNN_models.Complex_MTASS import Complex_MTASS
from DNN_models.Complex_MTASS_Solver import Complex_MTASS_model


def sdr_cost(estimated, target, eps=1e-8):
    estimated = torch.from_numpy(estimated).float()
    target = torch.from_numpy(target).float()

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
    return sdr.item()


def sisdr_cost(estimated, target, eps=1e-8):
    estimated = torch.from_numpy(estimated).float()
    target = torch.from_numpy(target).float()

    dot = torch.sum(estimated * target, dim=-1, keepdim=True) + eps
    s_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps

    scale = dot / s_energy
    target_scaled = scale * target
    e_noise = estimated - target_scaled
    target_pow = torch.sum(target_scaled ** 2, dim=-1) + eps
    noise_pow = torch.sum(e_noise ** 2, dim=-1) + eps

    sisdr = 10 * torch.log10(target_pow / noise_pow)
    return sisdr.item()


def Inverse_STFT(inputs, win_len, win_hop, fft_len, device):
    cutoff = fft_len // 2 + 1
    real_part = inputs[:, :cutoff, :]
    imag_part = inputs[:, cutoff:, :]

    complex_spec = torch.complex(real_part, imag_part)
    istft_window = torch.hamming_window(win_len, device=device)

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


def STFT(wav_data, win_len, win_hop, fft_len, device):
    batch_size, num_samples = wav_data.shape

    if num_samples <= win_len:
        nf = 1
    else:
        nf = (num_samples - win_len + win_hop - 1) // win_hop + 1
    pad_length = (nf - 1) * win_hop + win_len
    padding_size = pad_length - num_samples
    if padding_size > 0:
        wav_data = torch.nn.functional.pad(wav_data, (0, padding_size), mode='constant', value=0.0)

    window = torch.hamming_window(win_len, device=device)
    complex_spec = torch.stft(
        wav_data,
        n_fft=fft_len,
        hop_length=win_hop,
        win_length=win_len,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=False
    )
    print(f"[DEBUG STFT] input_shape={wav_data.shape}, complex_spec.shape={complex_spec.shape}")
    real_part = complex_spec[:, :, :, 0]
    imag_part = complex_spec[:, :, :, 1]
    features = torch.cat([real_part, imag_part], dim=1)
    print(f"[DEBUG STFT] output_shape={features.shape}")
    return features


def wav_write(data, path, filename, fs):
    full_path = os.path.join(path, filename)
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    wav.write(full_path, fs, (data * 32767).astype(np.int16))


def get_existing_classes(sample_dir):
    gt_map = {
        'speech_gt.wav': 'speech',
        'music_gt.wav': 'music',
        'others_gt.wav': 'others'
    }
    existing_classes = []
    gt_paths = {}
    eps = 1e-8
    for gt_file, class_name in gt_map.items():
        gt_path = os.path.join(sample_dir, gt_file)
        if os.path.exists(gt_path):
            fs, gt_data = wav.read(gt_path)
            if gt_data.dtype != np.float32:
                gt_data = gt_data.astype(np.float32) / 32767.0
            if np.sum(gt_data ** 2) > eps:
                existing_classes.append(class_name)
                gt_paths[class_name] = (fs, gt_data)
    return existing_classes, gt_paths


def process_offline(model, mixture_path, existing_classes, device, win_len=512, win_inc=256, fft_len=512):
    fs_read, audio_data = wav.read(mixture_path)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    audio_tensor = torch.from_numpy(audio_data).float().to(device)
    audio_tensor = audio_tensor.unsqueeze(0)

    print(f"[DEBUG process] mixture_path={mixture_path}, audio_data.shape={audio_data.shape}, tensor.shape={audio_tensor.shape}")

    X1 = STFT(audio_tensor, win_len, win_inc, fft_len, device)

    with torch.no_grad():
        Z1, Z2, Z3 = model(X1)

    results = {}
    if 'speech' in existing_classes:
        speech_wav = Inverse_STFT(Z1, win_len, win_inc, fft_len, device)
        results['speech'] = speech_wav.squeeze().detach().cpu().numpy()
    if 'music' in existing_classes:
        music_wav = Inverse_STFT(Z2, win_len, win_inc, fft_len, device)
        results['music'] = music_wav.squeeze().detach().cpu().numpy()
    if 'others' in existing_classes:
        others_wav = Inverse_STFT(Z3, win_len, win_inc, fft_len, device)
        results['others'] = others_wav.squeeze().detach().cpu().numpy()

    mixture_wav = Inverse_STFT(X1, win_len, win_inc, fft_len, device)
    results['mixture'] = mixture_wav.squeeze().detach().cpu().numpy()

    return results, fs_read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', type=str, required=True, help='Directory with test wav samples')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint .ckpt')
    parser.add_argument('--output_dir', type=str, default='./wav_offline_results', help='Folder to save results')
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to test')
    parser.add_argument('--rename_output', action='store_true', default=True, help='Rename output dir with classes suffix')

    args = parser.parse_args()

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = ComplexMTASSLightning.load_from_checkpoint(
        args.ckpt_path,
        model_class=Complex_MTASS,
        loss_class=Complex_MTASS_model,
    )
    model.to(device)
    model.eval()
    model.freeze()
    print("Model loaded!")

    sample_dirs = []
    for item in sorted(os.listdir(args.wav_dir)):
        item_path = os.path.join(args.wav_dir, item)
        if os.path.isdir(item_path):
            mixture_path = os.path.join(item_path, 'mixture.wav')
            if os.path.exists(mixture_path):
                sample_dirs.append(item_path)

    if args.num_samples is not None:
        sample_dirs = sample_dirs[:args.num_samples]

    print(f"Found {len(sample_dirs)} samples")

    os.makedirs(args.output_dir, exist_ok=True)

    all_speech_sdr = []
    all_music_sdr = []
    all_others_sdr = []
    all_speech_sisdr = []
    all_music_sisdr = []
    all_others_sisdr = []

    win_len = 512
    win_inc = 256
    fft_len = 512

    for sample_idx, sample_dir in enumerate(tqdm(sample_dirs, desc="Processing samples")):
        sample_name = os.path.basename(sample_dir)
        mixture_path = os.path.join(sample_dir, 'mixture.wav')

        existing_classes, gt_paths = get_existing_classes(sample_dir)

        if args.rename_output and existing_classes:
            classes_suffix = '-'.join(existing_classes)
            output_sample_name = f"{sample_name}_{classes_suffix}"
        else:
            output_sample_name = sample_name

        output_sample_dir = os.path.join(args.output_dir, output_sample_name)
        os.makedirs(output_sample_dir, exist_ok=True)

        results, fs = process_offline(model, mixture_path, existing_classes, device, win_len, win_inc, fft_len)

        if 'speech' in existing_classes and 'speech' in results:
            speech_es = results['speech']
            fs_gt, speech_gt = gt_paths['speech']
            min_len = min(len(speech_es), len(speech_gt))
            speech_sdr = sdr_cost(speech_es[:min_len], speech_gt[:min_len])
            speech_sisdr = sisdr_cost(speech_es[:min_len], speech_gt[:min_len])
            all_speech_sdr.append(speech_sdr)
            all_speech_sisdr.append(speech_sisdr)
            wav_write(speech_es, output_sample_dir, 'speech_es.wav', fs)

        if 'music' in existing_classes and 'music' in results:
            music_es = results['music']
            fs_gt, music_gt = gt_paths['music']
            min_len = min(len(music_es), len(music_gt))
            music_sdr = sdr_cost(music_es[:min_len], music_gt[:min_len])
            music_sisdr = sisdr_cost(music_es[:min_len], music_gt[:min_len])
            all_music_sdr.append(music_sdr)
            all_music_sisdr.append(music_sisdr)
            wav_write(music_es, output_sample_dir, 'music_es.wav', fs)

        if 'others' in existing_classes and 'others' in results:
            others_es = results['others']
            fs_gt, others_gt = gt_paths['others']
            min_len = min(len(others_es), len(others_gt))
            others_sdr = sdr_cost(others_es[:min_len], others_gt[:min_len])
            others_sisdr = sisdr_cost(others_es[:min_len], others_gt[:min_len])
            all_others_sdr.append(others_sdr)
            all_others_sisdr.append(others_sisdr)
            wav_write(others_es, output_sample_dir, 'others_es.wav', fs)

        wav_write(results['mixture'], output_sample_dir, 'mixture.wav', fs)

    print("\n" + "="*60)
    print("SDR Statistics (Offline):")
    print("="*60)

    if all_speech_sdr:
        print(f"Speech SDR:    {np.mean(all_speech_sdr):.2f} +/- {np.std(all_speech_sdr):.2f}")
        print(f"Speech SI-SDR: {np.mean(all_speech_sisdr):.2f} +/- {np.std(all_speech_sisdr):.2f}")

    if all_music_sdr:
        print(f"Music SDR:     {np.mean(all_music_sdr):.2f} +/- {np.std(all_music_sdr):.2f}")
        print(f"Music SI-SDR:  {np.mean(all_music_sisdr):.2f} +/- {np.std(all_music_sisdr):.2f}")

    if all_others_sdr:
        print(f"Others SDR:    {np.mean(all_others_sdr):.2f} +/- {np.std(all_others_sdr):.2f}")
        print(f"Others SI-SDR: {np.mean(all_others_sisdr):.2f} +/- {np.std(all_others_sisdr):.2f}")

    all_sdr = all_speech_sdr + all_music_sdr + all_others_sdr
    all_sisdr = all_speech_sisdr + all_music_sisdr + all_others_sisdr

    if all_sdr:
        print(f"\nTotal Average SDR:    {np.mean(all_sdr):.2f} +/- {np.std(all_sdr):.2f}")
        print(f"Total Average SI-SDR: {np.mean(all_sisdr):.2f} +/- {np.std(all_sisdr):.2f}")

    print("="*60)


if __name__ == '__main__':
    main()
