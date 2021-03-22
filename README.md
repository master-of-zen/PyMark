# PyMark
Simple benchmarker
Calculates BSQ-rate and BD-rate 

# Usage
2 modes:
- Benchmark (default) - benchmarks and generates data file (json)
- Plot - reads data file and prints 
(recommended to use y4m if found desync issues for metric calculations)

```
-f --function - what to do [`benchmark`(default), `process`]
-i --input - input file (could be video or json file depending on usage)
-e --encoder ENCODER [ENCODER ...], -e ENCODER [ENCODER ...]
-m --metric METRIC [METRIC ...], -m METRIC [METRIC ...]
```

# example of usage

#### Benchmark 2 encoders from y4m

`./pymark.py -i shrek.y4m  -r x265 aom ` `data.json` will be generated with results

#### Print metrics
`./pymark.py -i data.json -f process -m VMAF PSNR -r BD BSQ ` will print rates for BD and BSQ, for VMAF and PSNR 
```
VMAF BD rate: -66.2935
VMAF BSQ rate: 0.355
PSNR BD rate: -66.9217
PSNR BSQ rate: 0.35
```

#### Plots
adding `-p` will make plot for each processed metric

# Requirements
`ffmpeg`
`python`
Encoders `(x265, aomenc)`