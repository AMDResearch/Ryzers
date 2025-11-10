#!/usr/bin/env python3
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import cv2
import torch
import numpy as np
import argparse
import time

def load_midas_model(device):
    """Load MiDaS small model."""
    print(f"Loading MiDaS model on {device}...")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.to(device)
    model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    print(f"Model loaded successfully")
    return model, transform

def process_frame(frame, model, transform, device):
    """Process a single frame and return depth map."""
    input_batch = transform(frame).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_map = 255 - depth_map
    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_COOL)

    return depth_colored

def main():
    parser = argparse.ArgumentParser(description='GPU Depth Estimation using MiDaS')
    parser.add_argument('-i', '--input', required=True, help='Input video file')
    parser.add_argument('-o', '--output', default='depth_output.mp4', help='Output video file')
    args = parser.parse_args()

    # Use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Load model
    model, transform = load_midas_model(device)

    # Open video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.input}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")

    # Setup output video (side-by-side)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width * 2, height))
    print(f"Writing output to: {args.output}\n")

    # Process video
    frame_count = 0
    inference_times = []

    print("Processing video...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Measure inference time
            start_time = time.time()
            depth_colored = process_frame(frame_rgb, model, transform, device)
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)

            # Create side-by-side visualization
            combined = np.hstack([frame, depth_colored])
            writer.write(combined)

            # Progress update every 30 frames
            if frame_count % 30 == 0:
                avg_time = np.mean(inference_times[-30:])
                print(f"Frame {frame_count}/{total_frames} | {avg_time:.1f}ms/frame ({1000/avg_time:.1f} fps)")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        writer.release()

        # Print summary
        if inference_times:
            print(f"\n{'='*60}")
            print(f"Processed {frame_count} frames")
            print(f"Average inference: {np.mean(inference_times):.2f}ms ({1000/np.mean(inference_times):.1f} fps)")
            print(f"{'='*60}")
            print(f"Output saved: {args.output}\n")

    return 0

if __name__ == '__main__':
    exit(main())
