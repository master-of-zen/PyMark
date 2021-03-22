# PyMark
Simple benchmarker
Calculates BSQ-rate and BD-rate 

# Usage
2 modes:
- Benchmark (default) - benchmarks and generates data file (json)
- Plot - reads data file and prints 
(recommended to use y4m if found desync issues for metric calculations)

```
-i --input   - input file (could be video or json file )
-e --encoder - encoder to run benchmark for [aom, x265] 
-m --metric  - select which metric to show [VMAF, PSNR, SSIM, MS-SSIM]
-r --rates   - select  which rate modes to calculate [BD, BSQ]
-p --plot    - makes plot for each processed metric
```

# example of usage

Behaviour depends on input, files with `.json` suffix will be processed in metrics/plot mode,
everyting else in benchmark mode

#### Benchmark 2 encoders from y4m

`./pymark.py -i shrek.y4m  -e x265 aom ` `data.json` will be generated with results of aomenc and x265 encoders

#### Print metrics
`./pymark.py -i data.json -m VMAF PSNR -r BD BSQ ` will print rates for BD and BSQ, for VMAF and PSNR 
```
VMAF BD rate: -66.2935
VMAF BSQ rate: 0.355
PSNR BD rate: -66.9217
PSNR BSQ rate: 0.35
```

#### Plots
adding `-p` will make plot for each processed metric

# Requirements
`ffmpeg with libvmaf`
`python`
Encoders `(x265, aomenc)`