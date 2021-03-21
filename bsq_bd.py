#!/usr/bin/env python3

import sys
import math
import numpy as np
import argparse
from pathlib import Path
import subprocess
from subprocess import PIPE, STDOUT
from typing import Tuple, List, Dict
from collections import deque
import json


def bdsnr(metric_set1, metric_set2):
    rate1 = [x[0] for x in metric_set1]
    psnr1 = [x[1] for x in metric_set1]
    rate2 = [x[0] for x in metric_set2]
    psnr2 = [x[1] for x in metric_set2]

    log_rate1 = list(map(math.log, rate1))
    log_rate2 = list(map(math.log, rate2))

    # Best cubic poly fit for graph represented by log_ratex, psrn_x.
    poly1 = np.polyfit(log_rate1, psnr1, 3)
    poly2 = np.polyfit(log_rate2, psnr2, 3)

    # Integration interval.
    min_int = max([min(log_rate1), min(log_rate2)])
    max_int = min([max(log_rate1), max(log_rate2)])

    # Integrate poly1, and poly2.
    p_int1 = np.polyint(poly1)
    p_int2 = np.polyint(poly2)

    # Calculate the integrated value over the interval we care about.
    int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
    int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)

    # Calculate the average improvement.
    if max_int != min_int:
        avg_diff = (int2 - int1) / (max_int - min_int)
    else:
        avg_diff = 0.0
    return avg_diff


def bdrate(metric_set1, metric_set2):
    rate1 = [x[0] for x in metric_set1]
    psnr1 = [x[1] for x in metric_set1]
    rate2 = [x[0] for x in metric_set2]
    psnr2 = [x[1] for x in metric_set2]

    log_rate1 = list(map(math.log, rate1))
    log_rate2 = list(map(math.log, rate2))

    # Best cubic poly fit for graph represented by log_ratex, psrn_x.
    poly1 = np.polyfit(psnr1, log_rate1, 2)
    poly2 = np.polyfit(psnr2, log_rate2, 2)

    # Integration interval.
    min_int = max([min(psnr1), min(psnr2)])
    max_int = min([max(psnr1), max(psnr2)])

    # find integral
    p_int1 = np.polyint(poly1)
    p_int2 = np.polyint(poly2)

    # Calculate the integrated value over the interval we care about.
    int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
    int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)

    # Calculate the average improvement.
    avg_exp_diff = (int2 - int1) / (max_int - min_int)

    # In really bad formed data the exponent can grow too large.
    # clamp it.
    if avg_exp_diff > 200:
        avg_exp_diff = 200

    # Convert to a percentage.
    avg_diff = (math.exp(avg_exp_diff) - 1) * 100

    return avg_diff


def run_encode(pipe):
    """Run encode with provided ffmpeg and encoder command"""

    encoder_history = deque(maxlen=20)

    while True:
        line = pipe.stdout.readline().strip()

        if len(line) == 0 and pipe.poll() is not None:
            break

        if len(line) == 0:
            continue

        if line:
            encoder_history.append(line.decode())

    if pipe.returncode != 0 and pipe.returncode != -2:
        tb = sys.exc_info()[2]
        print("\n".join(encoder_history))
        raise RuntimeError("Error in processing encoding pipe").with_traceback(tb)


def read_json_file(pth: Path) -> Dict:
    with open(pth) as fl:
        return json.load(fl)


def read_metrics(js: Dict) -> Dict:
    new = {}
    for key in ("VMAF score", "PSNR score", "SSIM score", "MS-SSIM score"):
        new[key.split()[0]] = round(js.pop(key), 4)
    return new


def make_pipe(source: Path, encoder_command: List, bit_depth=8):

    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        source.resolve(),
        "-pix_fmt",
        "yuv420p" if bit_depth == 8 else "yuv420p10le",
        "-f",
        "yuv4mpegpipe",
        "-",
    ]

    ffmpeg_pipe = subprocess.Popen(ffmpeg_command, stdout=PIPE, stderr=STDOUT)
    pipe = subprocess.Popen(
        encoder_command, stdin=ffmpeg_pipe.stdout, stdout=PIPE, stderr=STDOUT
    )
    return pipe


def calculate_metrics(source: Path, probe: Path):
    fl = Path(f"{probe}.json")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-r",
        "24",
        "-i",
        source.as_posix(),
        "-r",
        "24",
        "-i",
        probe,
        "-filter_complex",
        f"[0:v]setpts=PTS-STARTPTS[reference];[1:v]setpts=PTS-STARTPTS[distorted];[distorted][reference]libvmaf=psnr=1:ssim=1:ms_ssim=1:log_path={fl.as_posix()}:log_fmt=json",
        "-f",
        "null",
        "-",
    ]

    p = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT)

    run_encode(p)

    return fl


def benchmark(source: Path, encoder: str):

    # by https://tools.ietf.org/id/draft-ietf-netvc-testing-08.html#rfc.section.4.3
    libaom_q = (20, 32, 43, 55)
    hevc_q = (15, 20, 25, 30, 35)

    results = {"aom": {}}

    if encoder == "aom":

        for q in libaom_q:
            probe = f"{q}_{source}"
            command = [
                "aomenc",
                "--passes=1",
                "-t",
                "16",
                "--end-usage=q",
                "--cpu-used=6",
                f"--cq-level={q}",
                "-o",
                probe,
                "-",
            ]
            print(f":: Encoding aom {q}")
            pipe = make_pipe(source, command)
            run_encode(pipe)
            fl = calculate_metrics(source, probe)
            js = read_json_file(fl)
            js = read_metrics(js)
            results["aom"][q] = dict(js)
            with open("data.json", "w") as outfile:

                json.dump(results, outfile)


def print_metrics():
    # Vmaf
    vmaf0 = [(x[4], round(x[5], 3)) for x in data if x[2] == i]
    vmaf1 = [(x[4], round(x[5], 3)) for x in data if x[2] == ii]

    # PSNR
    psnr0 = [(x[4], x[6], 3) for x in data if x[2] == i]
    psnr1 = [(x[4], x[6], 3) for x in data if x[2] == ii]

    # SSIM
    ssim0 = [(x[4], x[7]) for x in data if x[2] == i]
    ssim1 = [(x[4], x[7]) for x in data if x[2] == ii]

    # MS-SSIM
    mssim0 = [(x[4], x[8]) for x in data if x[2] == i]
    mssim1 = [(x[4], x[8]) for x in data if x[2] == ii]

    print("Vmaf BD rate:", bdrate(vmaf1, vmaf0))
    print("PSNR BD rate:", bdrate(psnr1, psnr0))
    print("SSIM BD rate:", bdrate(ssim1, ssim0))
    print("MS-SSIM BD rate:", bdrate(mssim1, mssim0))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    main_group = parser.add_argument_group("Functions")
    main_group.add_argument(
        "--function",
        "-f",
        nargs="+",
        default="benchmark",
        type=str,
        help="What to do",
    )
    main_group.add_argument("--input", "-i", required=True, type=Path)
    main_group.add_argument("--encoder", "-e", required=True, type=str)

    parsed = vars(parser.parse_args())
    if not parsed["input"]:
        parser.print_help()
        sys.exit()

    benchmark(parsed["input"], parsed["encoder"])
