# PyMark
Simple benchmarker
Calculates BSQ-rate and BD-rate 

# Usage
2 modes:
- Benchmark (default) - benchmarks and generates data file (json)
- Plot - reads data file and prints 

```
-f --function - what to do [`benchmark`(default), `plot`]
-i --input - input file (could be video or json file depending on usage)
-e --encoder ENCODER [ENCODER ...], -e ENCODER [ENCODER ...]
-m --metric METRIC [METRIC ...], -m METRIC [METRIC ...]```



# Requirements
`ffmpeg`
`python`
Encoders `(x265, aomenc)`