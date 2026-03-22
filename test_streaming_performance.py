
#!/usr/bin/env python3
import os
import sys
import time
import argparse
import torch
import numpy as np
import scipy.io.wavfile as wav
from tqdm import tqdm

project_root = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(project_root, 'model_constrcution')
os.chdir(model_dir)
sys.path.insert(0, model_dir)

from DNN_models.Complex_MTASS_model_streaming import ComplexMTASSLightningStreaming
from DNN_models.Complex_MTASS_streaming import Complex_MTASS_Streaming
from DNN_models.Complex_MTASS_Solver import Complex_MTASS_model


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
        
        for i in range(0, num_new, self.win_inc):
            chunk_end = min(i + self.win_inc, num_new)
            chunk = new_samples[i:chunk_end]
            
            self.input_buffer = torch.roll(self.input_buffer, -len(chunk))
            self.input_buffer[-len(chunk):] = chunk
            
            if i + self.win_inc <= num_new or i == 0:
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
        self.istft_music = StreamingISTFT(win_len, win_inc, fft_len, device)
        self.istft_others = StreamingISTFT(win_len, win_inc, fft_len, device)
        
        self.spec_buffer = None
        
    def reset(self):
        self.stft.reset()
        self.istft_speech.reset()
        self.istft_music.reset()
        self.istft_others.reset()
        self.spec_buffer = None
    
    def _init_spec_buffer(self, first_spec):
        import torch.nn.functional as F
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
        music_samples = self.istft_music.process(z2_spec)
        others_samples = self.istft_others.process(z3_spec)
        
        return speech_samples, music_samples, others_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_path', type=str, default="./test_2mix_wavs/sample0/mixture.wav", help='Input wav file')
    parser.add_argument('--ckpt_path', type=str, default="experiments/small_test_streaming_cpu/checkpoints/epoch=0-step=2.ckpt", help='Model checkpoint')
    parser.add_argument('--history_size', type=int, default=256, help='History buffer size (frames)')
    parser.add_argument('--chunk_size', type=int, default=32, help='Chunk size (frames)')
    parser.add_argument('--simulate_real_time', action='store_true', help='Simulate real-time with sleep')
    
    args = parser.parse_args()
    
    fs = 16000
    frame_duration = 256 / fs
    chunk_duration = args.chunk_size * frame_duration
    
    print("="*60)
    print("Audio Chunk Streaming Performance Test")
    print("="*60)
    print(f"Sampling rate: {fs} Hz")
    print(f"Frame duration: {frame_duration*1000:.1f} ms")
    print(f"Chunk size: {args.chunk_size} frames = {chunk_duration*1000:.1f} ms")
    print(f"History size: {args.history_size} frames = {args.history_size*frame_duration*1000:.1f} ms")
    print()
    
    device = 'cpu'
    print(f"Using device: {device}")
    
    print("\nLoading model...")
    model = ComplexMTASSLightningStreaming.load_from_checkpoint(
        args.ckpt_path,
        model_class=Complex_MTASS_Streaming,
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
    
    print(f"\nLoading audio file: {args.wav_path}")
    fs_read, audio_data = wav.read(args.wav_path)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    
    audio_duration = len(audio_data) / fs
    print(f"Audio duration: {audio_duration:.2f} seconds")
    
    samples_per_chunk = args.chunk_size * 256
    
    num_chunks = (len(audio_data) + samples_per_chunk - 1) // samples_per_chunk
    print(f"Number of chunks: {num_chunks}")
    
    print("\n" + "="*60)
    print("Running performance test...")
    print("="*60)
    
    separator.reset()
    
    chunk_times = []
    first_chunk_latency = None
    
    for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
        start_idx = chunk_idx * samples_per_chunk
        end_idx = min(start_idx + samples_per_chunk, len(audio_data))
        chunk_data = audio_data[start_idx:end_idx]
        
        if args.simulate_real_time and chunk_idx > 0:
            simulated_duration = len(chunk_data) / fs
            time.sleep(simulated_duration * 0.9)
        
        start_time = time.time()
        
        speech, music, others = separator.process(chunk_data)
        
        end_time = time.time()
        chunk_time = end_time - start_time
        chunk_times.append(chunk_time)
        
        if first_chunk_latency is None:
            first_chunk_latency = chunk_time
    
    print("\n" + "="*60)
    print("Performance Results")
    print("="*60)
    
    if chunk_times:
        avg_chunk_time = np.mean(chunk_times)
        std_chunk_time = np.std(chunk_times)
        max_chunk_time = np.max(chunk_times)
        min_chunk_time = np.min(chunk_times)
        
        total_process_time = np.sum(chunk_times)
        rtf = total_process_time / audio_duration
        
        print(f"Audio duration: {audio_duration:.2f} s")
        print(f"Total processing time: {total_process_time:.4f} s")
        print(f"RTF (Real-Time Factor): {rtf:.4f}")
        print()
        print(f"First chunk latency: {first_chunk_latency*1000:.2f} ms")
        print(f"Avg chunk time: {avg_chunk_time*1000:.2f} ms +/- {std_chunk_time*1000:.2f} ms")
        print(f"Min chunk time: {min_chunk_time*1000:.2f} ms")
        print(f"Max chunk time: {max_chunk_time*1000:.2f} ms")
        print()
        print(f"Chunk size: {args.chunk_size} frames = {chunk_duration*1000:.1f} ms")
        print(f"  Avg processing per chunk: {avg_chunk_time*1000:.2f} ms")
        print(f"  {'✓ Real-time capable' if avg_chunk_time < chunk_duration else '✗ Not real-time capable'}")
        print(f"  (processing time {avg_chunk_time*1000:.1f} ms < chunk duration {chunk_duration*1000:.1f} ms)")
    
    print("="*60)


if __name__ == "__main__":
    main()
