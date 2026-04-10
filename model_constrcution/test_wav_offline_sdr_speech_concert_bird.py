#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from wav_offline_eval_core import (
    collect_sample_dirs,
    get_existing_classes,
    get_sample_index,
    load_offline_model,
    parse_csv_metadata,
    print_bucket_stats,
    print_category_stats,
    process_offline,
    sdr_cost,
    sisdr_cost,
    wav_write,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', type=str, required=True, help='Directory with test wav samples')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint .ckpt')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./wav_offline_results_speech_concert_bird',
        help='Folder to save results',
    )
    parser.add_argument('--csv_path', type=str, default=None, help='CSV metadata path generated for the wav samples')
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to test')
    parser.add_argument('--rename_output', action='store_true', default=True, help='Rename output dir with classes suffix')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    categories = ['speech', 'concert', 'bird']
    gt_filename_map = {
        'speech': 'speech_gt.wav',
        'concert': 'concert_gt.wav',
        'bird': 'bird_gt.wav',
    }
    est_filename_map = {
        'speech': 'speech_es.wav',
        'concert': 'concert_es.wav',
        'bird': 'bird_es.wav',
    }

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = load_offline_model(args.ckpt_path, device)
    print("Model loaded!")

    sample_dirs = collect_sample_dirs(args.wav_dir, args.num_samples)
    print(f"Found {len(sample_dirs)} samples")

    sample_category_counts = None
    if args.csv_path is not None:
        sample_category_counts = parse_csv_metadata(args.csv_path, categories)

    all_metrics = {category: {'sdr': [], 'sisdr': []} for category in categories}
    bucket_metrics = None
    if sample_category_counts is not None:
        bucket_metrics = {}
        for category in categories:
            bucket_metrics[f'{category}_single'] = {'sdr': [], 'sisdr': []}
            bucket_metrics[f'{category}_multi'] = {'sdr': [], 'sisdr': []}

    win_len = 512
    win_inc = 256
    fft_len = 512

    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        sample_name = os.path.basename(sample_dir)
        mixture_path = os.path.join(sample_dir, 'mixture.wav')

        existing_classes, gt_paths = get_existing_classes(sample_dir, categories, gt_filename_map)

        if args.rename_output and existing_classes:
            classes_suffix = '-'.join(existing_classes)
            output_sample_name = f"{sample_name}_{classes_suffix}"
        else:
            output_sample_name = sample_name

        output_sample_dir = os.path.join(args.output_dir, output_sample_name)
        os.makedirs(output_sample_dir, exist_ok=True)

        category_counts = None
        if sample_category_counts is not None:
            csv_sample_idx = get_sample_index(sample_name)
            if csv_sample_idx >= len(sample_category_counts):
                raise IndexError(
                    f"Sample index {csv_sample_idx} from '{sample_name}' exceeds csv size {len(sample_category_counts)}"
                )
            category_counts = sample_category_counts[csv_sample_idx]

        results, fs = process_offline(
            model,
            mixture_path,
            existing_classes,
            categories,
            device,
            win_len,
            win_inc,
            fft_len,
            debug=False,
        )

        for category in categories:
            if category in existing_classes and category in results:
                estimate = results[category]
                _, target = gt_paths[category]
                min_len = min(len(estimate), len(target))
                category_sdr = sdr_cost(estimate[:min_len], target[:min_len])
                category_sisdr = sisdr_cost(estimate[:min_len], target[:min_len])
                all_metrics[category]['sdr'].append(category_sdr)
                all_metrics[category]['sisdr'].append(category_sisdr)
                wav_write(estimate, output_sample_dir, est_filename_map[category], fs)

                if category_counts is not None and category_counts[category] >= 1:
                    bucket_name = f"{category}_single" if category_counts[category] == 1 else f"{category}_multi"
                    bucket_metrics[bucket_name]['sdr'].append(category_sdr)
                    bucket_metrics[bucket_name]['sisdr'].append(category_sisdr)

        wav_write(results['mixture'], output_sample_dir, 'mixture.wav', fs)

    print("\n" + "=" * 60)
    print("SDR Statistics (Offline, Speech/Concert/Bird):")
    print("=" * 60)

    for category in categories:
        print_category_stats(category.capitalize(), all_metrics[category]['sdr'], all_metrics[category]['sisdr'])

    if bucket_metrics is not None:
        print("\nCategory Count Breakdown:")
        for category in categories:
            print_bucket_stats(
                f"{category.capitalize()} Single-Source",
                bucket_metrics[f'{category}_single']['sdr'],
                bucket_metrics[f'{category}_single']['sisdr'],
            )
            print_bucket_stats(
                f"{category.capitalize()} Multi-Source",
                bucket_metrics[f'{category}_multi']['sdr'],
                bucket_metrics[f'{category}_multi']['sisdr'],
            )

    all_sdr = []
    all_sisdr = []
    for category in categories:
        all_sdr.extend(all_metrics[category]['sdr'])
        all_sisdr.extend(all_metrics[category]['sisdr'])

    if all_sdr:
        print(f"\nTotal Average SDR:    {np.mean(all_sdr):.2f} +/- {np.std(all_sdr):.2f}")
        print(f"Total Average SI-SDR: {np.mean(all_sisdr):.2f} +/- {np.std(all_sisdr):.2f}")

    print("=" * 60)


if __name__ == '__main__':
    main()
