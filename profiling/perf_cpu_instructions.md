# Linux perf CPU Profiling Instructions

## Overview

`perf` is the standard Linux performance analysis tool. It provides hardware counter sampling, call-graph profiling, and cache miss analysis — essential for understanding CPU-bound decode behavior.

## Prerequisites

- Linux kernel with perf support
- `perf` installed (`sudo apt install linux-tools-common linux-tools-$(uname -r)`)
- Root or `perf_event_paranoid` set appropriately: `echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid`

## Basic CPU Profile

```bash
# Sample CPU stacks during a benchmark run
perf record -g -F 999 -- \
  python -m bench.run_bench \
    --config configs/bench_default.yaml \
    --override backend=hf device=cpu benchmark.trials=1

# View flamegraph-style report
perf report --stdio
```

## Cache Miss Analysis

```bash
# Count L1/L2/LLC cache misses
perf stat -e cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses -- \
  python -m bench.run_bench \
    --config configs/bench_default.yaml \
    --override backend=hf device=cpu benchmark.trials=1 generation.output_length=32
```

## Key Metrics for Decode Analysis

| Counter | What It Tells You |
|---------|-------------------|
| `cache-misses / cache-references` | Cache miss rate; >5% suggests memory-bound |
| `LLC-load-misses` | Last-level cache misses → main memory accesses |
| `instructions / cycles` (IPC) | <1.0 IPC indicates memory stalls |
| `branch-misses` | Usually low for NN inference (predictable loops) |

## Interpreting Results

- **High LLC miss rate**: Model weights + KV cache exceed LLC → bandwidth-bound.
- **Low IPC**: CPU stalled waiting for memory; consistent with large model weights.
- **Matrix multiply dominance**: `sgemm` or `sdot` routines should dominate.

## Generating Flamegraphs

```bash
# Install flamegraph tools
git clone https://github.com/brendangregg/FlameGraph.git

# Generate
perf script | FlameGraph/stackcollapse-perf.pl | FlameGraph/flamegraph.pl > flamegraph.svg
```

## macOS Alternative

On macOS, use Instruments (Xcode) or `sample`:
```bash
# Quick sample of running process
sample $(pgrep -f "bench.run_bench") 5 -file results/cpu_sample.txt
```

Note: macOS lacks hardware counter access equivalent to Linux perf.
