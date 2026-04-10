#!/usr/bin/env python3
import argparse
import csv
import os

import librosa
import numpy as np
import scipy.io.wavfile as wav
from tqdm import tqdm

TARGET_SAMPLE_RATE = 16000
CATEGORY_FILES = {
    'speech': 'speech_gt.wav',
    'concert': 'concert_gt.wav',
    'bird': 'bird_gt.wav',
}


def load_wav(path, target_sample_rate=16000, max_length=160000):
    wav_data, _ = librosa.core.load(path, sr=target_sample_rate)
    if len(wav_data) > max_length:
        wav_data = wav_data[0:max_length]
    if len(wav_data) < max_length:
        wav_data = np.pad(wav_data, (0, max_length - len(wav_data)), 'constant')
    return wav_data


def mix_audios(audios, snrs):
    target = audios[0]
    target_energy = np.sum(target ** 2)
    mixed = target.copy()
    scaled_audios = [target]

    for i in range(1, len(audios)):
        noise = audios[i]
        noise_energy = np.sum(noise ** 2)
        snr_db = float(snrs[i])
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt((target_energy / snr_linear) / (noise_energy + 1e-8))
        scaled_noise = noise * scale
        mixed += scaled_noise
        scaled_audios.append(scaled_noise)

    max_value = np.max(np.abs(mixed))
    if max_value > 1:
        mixed *= 0.9 / max_value
        scaled_audios = [audio * 0.9 / max_value for audio in scaled_audios]

    return mixed, scaled_audios


def parse_csv(csv_path):
    src_names = []
    src_labels = []
    src_snrs = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)
        for row in reader:
            names = []
            labels = []
            snrs = []
            for i in range(0, len(row), 3):
                if i < len(row):
                    names.append(row[i])
                if i + 1 < len(row):
                    labels.append(row[i + 1])
                if i + 2 < len(row):
                    snrs.append(row[i + 2])
            src_names.append(names)
            src_labels.append(labels)
            src_snrs.append(snrs)

    return src_names, src_labels, src_snrs


def generate_wavs_from_csv(csv_path, output_dir, num_samples=None):
    os.makedirs(output_dir, exist_ok=True)

    src_names, src_labels, src_snrs = parse_csv(csv_path)

    if num_samples is not None:
        src_names = src_names[:num_samples]
        src_labels = src_labels[:num_samples]
        src_snrs = src_snrs[:num_samples]

    print(f"Generating {len(src_names)} wav files...")

    for idx in tqdm(range(len(src_names))):
        names = src_names[idx]
        labels = src_labels[idx]
        snrs = src_snrs[idx]

        audios = []
        valid = True
        for name in names:
            if os.path.exists(name):
                audios.append(load_wav(name, TARGET_SAMPLE_RATE))
            else:
                print(f"Warning: File not found: {name}")
                valid = False
                break

        if not valid or len(audios) < 2:
            continue

        mixed_wav, scaled_sources = mix_audios(audios, snrs)

        category_wavs = {
            'speech': np.zeros_like(mixed_wav),
            'concert': np.zeros_like(mixed_wav),
            'bird': np.zeros_like(mixed_wav),
        }

        for i, label in enumerate(labels):
            label_lower = label.lower()
            current_source = scaled_sources[i]

            if 'speech' in label_lower:
                category_wavs['speech'] += current_source
            elif 'concert' in label_lower:
                category_wavs['concert'] += current_source
            elif 'bird' in label_lower:
                category_wavs['bird'] += current_source
            else:
                raise ValueError(f"Unknown category label: {label}")

        sample_dir = os.path.join(output_dir, f"sample{idx}")
        os.makedirs(sample_dir, exist_ok=True)

        wav.write(
            os.path.join(sample_dir, 'mixture.wav'),
            TARGET_SAMPLE_RATE,
            (mixed_wav * 32767).astype(np.int16),
        )

        for category, filename in CATEGORY_FILES.items():
            wav.write(
                os.path.join(sample_dir, filename),
                TARGET_SAMPLE_RATE,
                (category_wavs[category] * 32767).astype(np.int16),
            )

    print(f"Generated {len(src_names)} samples to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--output_dir', type=str, default='./test_wavs_speech_concert_bird', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to generate')

    args = parser.parse_args()

    generate_wavs_from_csv(args.csv_path, args.output_dir, args.num_samples)
