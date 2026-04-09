#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io.wavfile as wav
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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


class StreamingSTFT:
    def __init__(self, win_len=512, win_inc=256, fft_len=512, device='cpu'):
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.device = device

        self.window = torch.hamming_window(win_len, device=device)
        self.input_buffer = torch.zeros(win_len, device=device)

    def reset(self):
        self.input_buffer = torch.zeros(self.win_len, device=self.device)

    def process(self, new_samples):
        new_samples = torch.from_numpy(new_samples).float().to(self.device)
        num_new = len(new_samples)

        frames = []

        if self.input_buffer.nonzero().numel() == 0 and num_new >= self.win_len:
            first_win = new_samples[:self.win_len]
            self.input_buffer = first_win
            framed = self.input_buffer * self.window
            spec = torch.fft.rfft(framed, n=self.fft_len)
            frames.append(spec)
            remaining_start = self.win_len
        else:
            remaining_start = 0

        for i in range(remaining_start, num_new, self.win_inc):
            chunk_end = min(i + self.win_inc, num_new)
            chunk = new_samples[i:chunk_end]

            self.input_buffer = torch.roll(self.input_buffer, -len(chunk))
            self.input_buffer[-len(chunk):] = chunk

            framed = self.input_buffer * self.window
            spec = torch.fft.rfft(framed, n=self.fft_len)
            frames.append(spec)

        if frames:
            return torch.stack(frames, dim=-1)
        return None


class StreamingISTFT:
    def __init__(self, win_len=512, win_inc=256, fft_len=512, device='cpu'):
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.device = device

        self.window = torch.hamming_window(win_len, device=device)
        self.output_buffer = torch.zeros(win_len, device=device)
        self.prev_samples = torch.zeros(win_len - win_inc, device=device)

    def reset(self):
        self.output_buffer = torch.zeros(self.win_len, device=self.device)
        self.prev_samples = torch.zeros(self.win_len - self.win_inc, device=self.device)

    def process(self, spec_frames):
        num_frames = spec_frames.shape[-1]

        output_samples = []

        for i in range(num_frames):
            spec = spec_frames[..., i]
            framed = torch.fft.irfft(spec, n=self.fft_len)
            framed = framed * self.window

            self.output_buffer[:self.win_len - self.win_inc] = self.prev_samples
            self.output_buffer[self.win_len - self.win_inc:] = 0
            self.output_buffer += framed

            output_samples.append(self.output_buffer[:self.win_inc].clone())
            self.prev_samples = self.output_buffer[self.win_inc:].clone()

        if output_samples:
            return torch.cat(output_samples, dim=0).cpu().numpy()
        return None


class RealTimeAudioSeparator:
    def __init__(self, model, win_len=512, win_inc=256, fft_len=512,
                 history_size=256, chunk_size=32, device='cpu'):
        self.model = model
        self.model.eval()
        self.device = device

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.history_size = history_size
        self.chunk_size = chunk_size

        self.stft = StreamingSTFT(win_len, win_inc, fft_len, device)
        self.istft_speech = StreamingISTFT(win_len, win_inc, fft_len, device)
        self.istft_concert = StreamingISTFT(win_len, win_inc, fft_len, device)
        self.istft_bird = StreamingISTFT(win_len, win_inc, fft_len, device)

        self.spec_buffer = None

    def reset(self):
        self.stft.reset()
        self.istft_speech.reset()
        self.istft_concert.reset()
        self.istft_bird.reset()
        self.spec_buffer = None

    def _init_spec_buffer(self, first_spec):
        pad_size = self.history_size - first_spec.shape[-1]
        if pad_size > 0:
            padded = F.pad(first_spec, (pad_size, 0))
            self.spec_buffer = padded
        else:
            self.spec_buffer = first_spec[..., -self.history_size:]

    def _process_spec_chunk(self, new_spec):
        if self.spec_buffer is None:
            self._init_spec_buffer(new_spec)
        else:
            self.spec_buffer = torch.cat(
                [self.spec_buffer, new_spec], dim=-1
            )[..., -self.history_size:]

        with torch.no_grad():
            z1, z2, z3 = self.model(self.spec_buffer.unsqueeze(0))

        out1 = z1[..., -new_spec.shape[-1]:].squeeze(0)
        out2 = z2[..., -new_spec.shape[-1]:].squeeze(0)
        out3 = z3[..., -new_spec.shape[-1]:].squeeze(0)

        return out1, out2, out3

    def _spec_to_ri(self, spec):
        real = spec.real
        imag = spec.imag
        return torch.cat([real, imag], dim=0)

    def _ri_to_spec(self, ri):
        cutoff = self.fft_len // 2 + 1
        real = ri[:cutoff]
        imag = ri[cutoff:]
        return torch.complex(real, imag)

    def process(self, audio_samples):
        spec_frames = self.stft.process(audio_samples)

        if spec_frames is None:
            return None, None, None

        ri_spec = self._spec_to_ri(spec_frames)

        z1_ri, z2_ri, z3_ri = self._process_spec_chunk(ri_spec)

        z1_spec = self._ri_to_spec(z1_ri)
        z2_spec = self._ri_to_spec(z2_ri)
        z3_spec = self._ri_to_spec(z3_ri)

        speech_samples = self.istft_speech.process(z1_spec)
        concert_samples = self.istft_concert.process(z2_spec)
        bird_samples = self.istft_bird.process(z3_spec)

        return speech_samples, concert_samples, bird_samples

    def process_file(self, input_path, output_dir, fs=16000):
        self.reset()

        fs_read, audio_data = wav.read(input_path)
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]

        os.makedirs(output_dir, exist_ok=True)

        buffer_size = self.win_inc * 100
        speech_output = []
        concert_output = []
        bird_output = []

        for i in range(0, len(audio_data), buffer_size):
            chunk = audio_data[i:i+buffer_size]
            speech, concert, bird = self.process(chunk)

            if speech is not None:
                speech_output.append(speech)
            if concert is not None:
                concert_output.append(concert)
            if bird is not None:
                bird_output.append(bird)

        speech_out = None
        concert_out = None
        bird_out = None

        if speech_output:
            speech_out = np.concatenate(speech_output)
            wav.write(os.path.join(output_dir, 'speech_es.wav'), fs,
                     (speech_out * 32767).astype(np.int16))

        if concert_output:
            concert_out = np.concatenate(concert_output)
            wav.write(os.path.join(output_dir, 'concert_es.wav'), fs,
                     (concert_out * 32767).astype(np.int16))

        if bird_output:
            bird_out = np.concatenate(bird_output)
            wav.write(os.path.join(output_dir, 'bird_es.wav'), fs,
                     (bird_out * 32767).astype(np.int16))

        return speech_out, concert_out, bird_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', type=str, required=True, help='Directory with test wav files')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to Complex_MTASS checkpoint .ckpt')
    parser.add_argument('--output_dir', type=str, default='./wav_streaming_offline_model_results_speech_concert_bird', help='Folder to save results')
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--history_size', type=int, default=256, help='History buffer size (frames)')
    parser.add_argument('--chunk_size', type=int, default=32, help='Chunk size (frames)')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to test')

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

    separator = RealTimeAudioSeparator(
        model,
        win_len=512,
        win_inc=256,
        fft_len=512,
        history_size=args.history_size,
        chunk_size=args.chunk_size,
        device=device
    )

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
    all_concert_sdr = []
    all_bird_sdr = []
    all_speech_sisdr = []
    all_concert_sisdr = []
    all_bird_sisdr = []

    for sample_idx, sample_dir in enumerate(tqdm(sample_dirs, desc="Processing samples")):
        sample_name = os.path.basename(sample_dir)
        output_sample_dir = os.path.join(args.output_dir, sample_name)

        mixture_path = os.path.join(sample_dir, 'mixture.wav')
        speech_gt_path = os.path.join(sample_dir, 'speech_gt.wav')
        concert_gt_path = os.path.join(sample_dir, 'concert_gt.wav')
        bird_gt_path = os.path.join(sample_dir, 'bird_gt.wav')

        fs, speech_gt = wav.read(speech_gt_path)
        fs, concert_gt = wav.read(concert_gt_path)
        fs, bird_gt = wav.read(bird_gt_path)

        if speech_gt.dtype != np.float32:
            speech_gt = speech_gt.astype(np.float32) / 32767.0
            concert_gt = concert_gt.astype(np.float32) / 32767.0
            bird_gt = bird_gt.astype(np.float32) / 32767.0

        eps = 1e-8
        valid_speech = np.sum(speech_gt ** 2) > eps
        valid_concert = np.sum(concert_gt ** 2) > eps
        valid_bird = np.sum(bird_gt ** 2) > eps

        speech_es, concert_es, bird_es = separator.process_file(mixture_path, output_sample_dir, fs)

        if valid_speech and speech_es is not None:
            min_len = min(len(speech_es), len(speech_gt))
            speech_sdr = sdr_cost(speech_es[:min_len], speech_gt[:min_len])
            speech_sisdr = sisdr_cost(speech_es[:min_len], speech_gt[:min_len])
            all_speech_sdr.append(speech_sdr)
            all_speech_sisdr.append(speech_sisdr)

        if valid_concert and concert_es is not None:
            min_len = min(len(concert_es), len(concert_gt))
            concert_sdr = sdr_cost(concert_es[:min_len], concert_gt[:min_len])
            concert_sisdr = sisdr_cost(concert_es[:min_len], concert_gt[:min_len])
            all_concert_sdr.append(concert_sdr)
            all_concert_sisdr.append(concert_sisdr)

        if valid_bird and bird_es is not None:
            min_len = min(len(bird_es), len(bird_gt))
            bird_sdr = sdr_cost(bird_es[:min_len], bird_gt[:min_len])
            bird_sisdr = sisdr_cost(bird_es[:min_len], bird_gt[:min_len])
            all_bird_sdr.append(bird_sdr)
            all_bird_sisdr.append(bird_sisdr)

    print("\n" + "=" * 60)
    print("SDR Statistics (Streaming with Complex_MTASS Offline Model):")
    print("=" * 60)

    if all_speech_sdr:
        print(f"Speech SDR:     {np.mean(all_speech_sdr):.2f} +/- {np.std(all_speech_sdr):.2f}")
        print(f"Speech SI-SDR:  {np.mean(all_speech_sisdr):.2f} +/- {np.std(all_speech_sisdr):.2f}")

    if all_concert_sdr:
        print(f"Concert SDR:    {np.mean(all_concert_sdr):.2f} +/- {np.std(all_concert_sdr):.2f}")
        print(f"Concert SI-SDR: {np.mean(all_concert_sisdr):.2f} +/- {np.std(all_concert_sisdr):.2f}")

    if all_bird_sdr:
        print(f"Bird SDR:       {np.mean(all_bird_sdr):.2f} +/- {np.std(all_bird_sdr):.2f}")
        print(f"Bird SI-SDR:    {np.mean(all_bird_sisdr):.2f} +/- {np.std(all_bird_sisdr):.2f}")

    all_sdr = all_speech_sdr + all_concert_sdr + all_bird_sdr
    all_sisdr = all_speech_sisdr + all_concert_sisdr + all_bird_sisdr

    if all_sdr:
        print(f"\nTotal Average SDR:    {np.mean(all_sdr):.2f} +/- {np.std(all_sdr):.2f}")
        print(f"Total Average SI-SDR: {np.mean(all_sisdr):.2f} +/- {np.std(all_sisdr):.2f}")

    print("=" * 60)


if __name__ == '__main__':
    main()
