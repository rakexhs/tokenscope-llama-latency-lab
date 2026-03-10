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
                                        decode_step_0        13.46%       1.775ms        22.55%       2.973ms       2.973ms             1  
                                        decode_step_5        13.35%       1.760ms        19.94%       2.629ms       2.629ms             1  
                                        decode_step_1         5.96%     785.365us        11.02%       1.453ms       1.453ms             1  
                                        decode_step_6         5.43%     716.162us         9.84%       1.298ms       1.298ms             1  
                                        decode_step_2         5.46%     719.825us         9.81%       1.293ms       1.293ms             1  
                   aten::scaled_dot_product_attention         1.52%     200.162us         9.28%       1.223ms      76.458us            16  
                                        decode_step_3         4.87%     642.622us         9.21%       1.215ms       1.215ms             1  
                                        decode_step_4         4.87%     642.243us         8.93%       1.178ms       1.178ms             1  
                                        decode_step_7         4.74%     625.121us         8.70%       1.147ms       1.147ms             1  
                                         aten::argmax         8.19%       1.080ms         8.21%       1.082ms     135.296us             8  
    aten::_scaled_dot_product_flash_attention_for_cpu         6.01%     791.863us         7.76%       1.023ms      63.948us            16  
                                         aten::linear         0.14%      18.288us         5.51%     726.535us      90.817us             8  
                                         aten::matmul         0.28%      37.006us         5.15%     679.455us      84.932us             8  
                                             aten::mm         4.75%     626.075us         4.76%     627.369us      78.421us             8  
                                          aten::addmm         2.28%     300.417us         4.16%     549.074us       8.579us            64  
                                     aten::layer_norm         0.67%      88.209us         3.24%     426.710us      10.668us            40  
                              aten::native_layer_norm         1.35%     177.752us         2.57%     338.501us       8.463us            40  
                                            aten::mul         1.01%     133.663us         2.42%     319.454us       4.991us            64  
                                      aten::transpose         1.69%     222.385us         2.33%     307.260us       1.829us           168  
                                      aten::embedding         0.34%      44.583us         2.14%     282.536us      17.658us            16  
                                             aten::to         0.40%      52.498us         1.96%     258.371us       2.936us            88  
                                   aten::index_select         1.02%     134.292us         1.61%     212.913us      13.307us            16  
                                       aten::_to_copy         0.84%     111.253us         1.56%     205.873us       3.217us            64  
                                           aten::view         1.47%     193.297us         1.47%     193.297us       0.589us           328  
                                            aten::add         0.86%     112.919us         1.34%     177.001us       2.458us            72  
                                          aten::empty         1.25%     164.583us         1.25%     164.583us       0.857us           192  
                                          aten::copy_         1.23%     162.411us         1.23%     162.411us       1.269us           128  
                                     aten::as_strided         1.16%     153.500us         1.16%     153.500us       0.446us           344  
                                          aten::split         0.22%      29.542us         1.04%     136.791us       8.549us            16  
                                         aten::expand         0.81%     106.206us         1.00%     132.329us       2.068us            64  
-----------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 13.185ms

```

Raw CSV: `results/Macbook_M1/summary/torch_profile_ops.csv`
