# torch.profiler Operator Summary

Model: `sshleifer/tiny-gpt2` | Device: `cpu` | Decode tokens: 8

## How to Read This Table

- **cpu_time_total**: Wall-clock time spent in this operator on CPU (microseconds).
- **cuda_time_total**: Time spent on GPU kernels (only with CUDA profiling).
- **calls**: Number of invocations across all decode steps.
- Top operators by CPU time reveal where compute is spent.
- For memory-bound decode, `aten::mm` / `aten::addmm` (matrix multiply) dominate.

## Top Operators

```
-----------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                 Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-----------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        decode_step_0        32.50%       7.415ms        47.30%      10.794ms      10.794ms             1  
                                        decode_step_1         5.58%       1.272ms         9.25%       2.111ms       2.111ms             1  
                                        decode_step_6         5.07%       1.158ms         8.83%       2.015ms       2.015ms             1  
                                        decode_step_2         4.30%     981.272us         7.74%       1.766ms       1.766ms             1  
                                      aten::embedding         0.42%      95.123us         7.30%       1.666ms     104.126us            16  
                                        decode_step_5         4.03%     919.812us         7.22%       1.647ms       1.647ms             1  
                                        decode_step_4         3.53%     805.674us         6.77%       1.545ms       1.545ms             1  
                                   aten::index_select         5.78%       1.320ms         6.65%       1.516ms      94.774us            16  
                                        decode_step_3         3.50%     797.693us         6.45%       1.472ms       1.472ms             1  
                                        decode_step_7         3.42%     779.926us         6.44%       1.470ms       1.470ms             1  
                                          aten::addmm         3.76%     858.973us         5.60%       1.279ms      19.977us            64  
                                         aten::linear         0.11%      25.403us         5.00%       1.141ms     142.615us             8  
                                         aten::matmul         0.22%      49.582us         4.67%       1.066ms     133.302us             8  
                                             aten::mm         4.33%     989.055us         4.35%     991.766us     123.971us             8  
                                     aten::layer_norm         1.84%     420.531us         4.14%     944.250us      23.606us            40  
                   aten::scaled_dot_product_attention         0.29%      66.801us         3.38%     771.730us      48.233us            16  
                                         aten::argmax         3.15%     718.882us         3.18%     724.637us      90.580us             8  
    aten::_scaled_dot_product_flash_attention_for_cpu         1.44%     327.783us         3.09%     704.929us      44.058us            16  
                              aten::native_layer_norm         1.40%     318.483us         2.30%     523.719us      13.093us            40  
                                            aten::mul         1.04%     237.863us         2.29%     523.578us       8.181us            64  
                                      aten::transpose         1.75%     400.420us         2.24%     511.873us       3.047us           168  
                                             aten::to         0.35%      80.352us         1.68%     384.488us       4.369us            88  
                                       aten::_to_copy         0.71%     160.960us         1.33%     304.136us       4.752us            64  
                                            aten::add         0.98%     223.444us         1.31%     298.800us       4.150us            72  
                                          aten::split         0.35%      80.380us         1.25%     285.035us      17.815us            16  
                                           aten::view         1.20%     273.448us         1.20%     273.448us       0.834us           328  
                                          aten::empty         1.06%     242.890us         1.06%     242.890us       1.265us           192  
                                     aten::as_strided         1.06%     242.799us         1.06%     242.799us       0.706us           344  
                                          aten::copy_         1.06%     240.805us         1.06%     240.805us       1.881us           128  
                                         aten::expand         0.80%     182.599us         0.99%     225.536us       3.524us            64  
-----------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 22.819ms

```

Raw CSV: `results/WSL_Windows/summary/torch_profile_ops.csv`
