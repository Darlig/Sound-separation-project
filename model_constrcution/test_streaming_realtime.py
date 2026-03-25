
import os
import argparse
import torch
import numpy as np
import scipy.io.wavfile as wav
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DNN_models.Complex_MTASS_model_Streaming import ComplexMTASSStreamingLightning
from DNN_models.Complex_MTASS_Streaming import StreamingComplexMTASS
from DNN_models.Complex_MTASS_Solver_Streaming import Complex_MTASS_model_Streaming

win_len = 512
win_inc = 256
fft_len = 512
fs = 16000


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


class WavDataset(Dataset):
    def __init__(self, wav_dir):
        self.wav_dir = wav_dir
        self.sample_files = []
        for file in sorted(os.listdir(wav_dir)):
            if file.endswith('.wav'):
                self.sample_files.append(os.path.join(wav_dir, file))
    
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        wav_path = self.sample_files[idx]
        fs_read, mixture_wav = wav.read(wav_path)
        
        if mixture_wav.dtype != np.float32:
            mixture_wav = mixture_wav.astype(np.float32) / 32767.0
        
        if len(mixture_wav.shape) > 1:
            mixture_wav = mixture_wav[:, 0]
        
        mixture_tensor = torch.from_numpy(mixture_wav).float().unsqueeze(0).unsqueeze(0)
        X1 = STFT(mixture_tensor)
        
        audio_length = mixture_wav.shape[0] / fs
        
        return X1.squeeze(0), audio_length, os.path.basename(wav_path)


def test_realtime(args):
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
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
        num_workers=0
    )

    os.makedirs(args.output_dir, exist_ok=True)
    
    total_audio_length = 0.0
    total_inference_time = 0.0
    sample_stats = []
    
    print("\n" + "="*60)
    print(f"Starting real-time streaming test (chunk_size={args.chunk_size})...")
    print("="*60)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing"):
            X1, audio_length, filename = batch
            X1 = X1.to(device)
            num_frames = X1.shape[-1]
            
            model.model.reset_streaming_state()
            
            start_time = time.time()
            
            Z1, Z2, Z3 = Complex_MTASS_model_Streaming.streaming_separation(
                model.model, X1, chunk_size=args.chunk_size
            )
            
            end_time = time.time()
            
            inference_time = end_time - start_time
            rt_factor = inference_time / audio_length
            
            total_audio_length += audio_length
            total_inference_time += inference_time
            sample_stats.append((filename, audio_length, inference_time, rt_factor))
            
            print(f"  {filename}: audio_length={audio_length:.2f}s, inference={inference_time:.3f}s, RT={rt_factor:.3f}")
    
    print("\n" + "="*60)
    print("STATISTICS:")
    print("="*60)
    avg_rt = total_inference_time / total_audio_length
    print(f"Average RT:    {avg_rt:.4f}")
    print(f"Total audio: {total_audio_length:.2f}s")
    print(f"Total inference: {total_inference_time:.3f}s")
    
    if avg_rt < 1.0:
        print("\n✅ REAL-TIME COMPLIANT! Average RT < 1.0")
    else:
        print("\n⚠️ NOT REAL-TIME! Average RT >= 1.0")
    
    output_file = os.path.join(args.output_dir, 'realtime_stats.txt')
    with open(output_file, 'w') as f:
        f.write(f"chunk_size = {args.chunk_size}\n")
        f.write(f"avg_rt = {avg_rt:.4f}\n")
        f.write(f"total_audio_length = {total_audio_length:.2f}\n")
        f.write(f"total_inference_time = {total_inference_time:.3f}\n")
        f.write("\n")
        for filename, audio_length, inference_time, rt_factor in sample_stats:
            f.write(f"{filename} {audio_length:.2f} {inference_time:.3f} {rt_factor:.4f}\n")
    
    print(f"\nStatistics saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test real-time performance of streaming inference')
    parser.add_argument('--wav_dir', type=str, required=True, help='Directory with wav files to test')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint .ckpt')
    parser.add_argument('--output_dir', type=str, default='./streaming_realtime_results', help='Folder to save results')
    parser.add_argument('--chunk_size', type=int, default=1, 
                       help='Number of frames to process at once in streaming mode (default: 1 for frame-by-frame)')
    parser.add_argument('--use_cuda', action='store_true', default=True)
    
    args = parser.parse_args()
    
    test_realtime(args)
