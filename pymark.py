#!/usr/bin/env python3

import sys
import math
import numpy as np
import argparse
import re
import matplotlib
from matplotlib import plt
from pathlib import Path
import subprocess
from subprocess import PIPE, STDOUT
from typing import List, Dict
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


def get_bitrate(fl: Path):
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        fl,
        "-c",
        "copy",
        "-f",
        "null",
        "-",
    ]

    pipe = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT)

    encoder_history = []

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
        raise RuntimeError("Error in getting bitrate").with_traceback(tb)

    # Get size
    match = re.findall(r"video:([0-9]+)", "\n".join(encoder_history))
    size = int(match[-1])

    # Get time
    match = re.findall(
        r"time=([0-9+]+):([0-9+]+):([0-9+]+.[0-9+]+)", "\n".join(encoder_history)
    )
    h, m, s = match[0]
    bitrate = round(size / (int(h) * 3600 + int(m) * 60 + float(s)), 2)
    return bitrate


def read_metrics(js: Dict) -> Dict:
    # reads metrics from json and gets bitrate of probe file
    new = {}
    for key in ("VMAF score", "PSNR score", "SSIM score", "MS-SSIM score"):
        new[key.split()[0]] = round(js.pop(key), 4)

    new["BITRATE"] = js.pop("BITRATE")

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
        f"[0:v]setpts=PTS-STARTPTS[reference];\
          [1:v]setpts=PTS-STARTPTS[distorted];\
          [distorted][reference]\
          libvmaf=psnr=1:ssim=1:ms_ssim=1:log_path={fl.as_posix()}:log_fmt=json",
        "-f",
        "null",
        "-",
    ]

    p = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT)

    run_encode(p)

    return fl


def bench_routine(source, command, probe):
    pipe = make_pipe(source, command)
    run_encode(pipe)
    fl = calculate_metrics(source, probe)
    js = read_json_file(fl)
    # add bitrate
    bitrate = get_bitrate(probe)
    js["BITRATE"] = bitrate
    js = read_metrics(js)
    return js


def benchmark(source: Path, encoder: list):

    assert isinstance(source, Path)
    # by https://tools.ietf.org/id/draft-ietf-netvc-testing-08.html#rfc.section.4.3
    libaom_q = (20, 32, 43, 55)
    # It makes sense
    hevc_q = (15, 20, 25, 30, 35)

    results = dict()

    if "aom" in encoder:
        results["aom"] = dict()
        for q in libaom_q:
            probe = f"{q}_{source.with_suffix('.ivf')}"
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
            js = bench_routine(source, command, probe)
            results["aom"][q] = dict(js)
            with open("data.json", "w") as outfile:

                json.dump(results, outfile)

    if "x265" in encoder:
        results["x265"] = dict()
        for q in hevc_q:
            probe = f"{q}_{source.with_suffix('.ivf')}"
            command = [
                "x265",
                "--log-level",
                "0",
                "--no-progress",
                "--y4m",
                "--preset",
                "fast",
                "--crf",
                f"{q}",
                "-o",
                probe,
                "-",
            ]
            print(f":: Encoding x265 {q}")
            js = bench_routine(source, command, probe)
            results["x265"][q] = dict(js)
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


def plot_range(data, encoder, m_place):
    dt = [x for x in data if x[2] == cpu]
    x = sorted([round(x[4]) for x in dt])
    y = sorted([x[m_place] for x in dt])
    time = sum([x[1] for x in dt]) / len(dt)
    f = interpolate.interp1d(x, y, kind="linear")
    xnew = np.linspace(min(x), max(x), max(x) - min(x))
    plt.plot(xnew, f(xnew), label=f"{encoder} {cpu}", linewidth=3)


def plot(data: Dict, codecs: List):

    for metric, place in [("VMAF", 5), ("PSNR", 6), ("SSIM", 7)]:

        plot_range(aom_data, "Blues", "Aomenc", place)
        plot_range(vvc_data, "Reds", "VVC", place)
        # Plot

        plt.xticks(
            [x for x in range(0, 40000, 100)],
            [int(x) for x in range(0, 40000, 100)],
            fontsize=22,
        )

        if metric in ("VMAF", "PSNR"):
            plt.yticks([x for x in range(0, 101, 1)], fontsize=28)
            [plt.axhline(i, color="grey", linewidth=0.5) for i in range(1, 100, 2)]
            [plt.axhline(i, color="black", linewidth=1) for i in range(0, 100, 2)]
        else:
            [
                plt.axhline(i / 100, color="grey", linewidth=0.5)
                for i in range(61, 1000, 2)
            ]
            [
                plt.axhline(i / 100, color="black", linewidth=1)
                for i in range(62, 1000, 2)
            ]
            plt.yticks([x / 100 for x in range(0, 1000, 1)], fontsize=28)

        [plt.axvline(i, color="grey", linewidth=0.3) for i in range(0, 40000, 500)]
        plt.ylabel(metric.capitalize(), size=32)
        plt.xlabel("Bit rate, Kbps", size=24)
        plt.title(f"{' vs '.join(codecs)}, latest git 10.12.2020 {metric}", size=28)
        plt.legend(prop={"size": 19}, loc="lower right")

        # if metric in ('VMAF', 'PSNR'):
        low_ylim = [x[place] for x in data]
        low_ylim = np.percentile(sorted(low_ylim), 10)

        bitrates = [x[4] for x in data]
        plt.xlim(int(np.percentile(bitrates, 10)), int(np.percentile(bitrates, 85)))

        if metric in ("VMAF"):
            high_ylim = 100
        elif metric in ("SSIM"):
            high_ylim = np.percentile(sorted([x[place] for x in data]), 90)
        elif metric in ("PSNR"):
            high_ylim = np.percentile(sorted([x[place] for x in data]), 90)

        plt.ylim((low_ylim, high_ylim))
        plt.margins(0)
        plt.subplots_adjust(left=0.045, right=0.99, top=0.965, bottom=0.065)
        plt.show()


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
    main_group.add_argument("--input", "-i", nargs="+", required=True, type=Path)
    main_group.add_argument("--encoder", "-e", nargs="+", type=str)

    parsed = vars(parser.parse_args())
    if not parsed["input"]:
        parser.print_help()
        sys.exit()

    if "benchmark" in parsed["function"]:
        enc = parsed["encoder"]

        if not Path(parsed["input"][0]).exists():
            print("No input file/Can't reach")
            print(Path(parsed["input"][0]))
            sys.exit()

        if not enc:
            print("No encoder selected")
            sys.exit()

        benchmark(parsed["input"][0], enc)

    elif "plot" in parsed["function"]:
        data = Path(parsed["input"])

        if not data.exists():
            print("No input file/Can't reach")
            print(Path(parsed["input"][0]))
            sys.exit()
