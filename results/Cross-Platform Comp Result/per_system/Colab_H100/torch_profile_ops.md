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
                                        decode_step_0        18.81%       2.158ms        31.30%       3.590ms       3.590ms             1  
                                        decode_step_1         5.82%     667.582us        11.24%       1.289ms       1.289ms             1  
                                        decode_step_6         5.36%     614.217us        10.46%       1.200ms       1.200ms             1  
                                        decode_step_2         5.08%     582.809us         9.84%       1.129ms       1.129ms             1  
                                        decode_step_7         4.90%     562.381us         9.51%       1.091ms       1.091ms             1  
                                        decode_step_5         4.94%     566.182us         9.47%       1.086ms       1.086ms             1  
                                        decode_step_3         4.70%     538.490us         9.17%       1.052ms       1.052ms             1  
                                        decode_step_4         4.73%     542.640us         9.00%       1.032ms       1.032ms             1  
                                         aten::argmax         6.90%     791.844us         6.94%     795.881us      99.485us             8  
                   aten::scaled_dot_product_attention         0.53%      60.594us         6.86%     786.343us      49.146us            16  
    aten::_scaled_dot_product_flash_attention_for_cpu         4.01%     460.427us         6.33%     725.749us      45.359us            16  
                                     aten::layer_norm         2.63%     301.456us         5.98%     685.724us      17.143us            40  
                                          aten::addmm         3.47%     398.378us         5.96%     683.666us      10.682us            64  
                                      aten::embedding         0.52%      59.436us         4.15%     476.450us      29.778us            16  
                              aten::native_layer_norm         2.07%     237.157us         3.35%     384.268us       9.607us            40  
                                   aten::index_select         2.31%     265.322us         3.34%     382.940us      23.934us            16  
                                            aten::mul         1.40%     160.727us         3.32%     381.248us       5.957us            64  
                                         aten::linear         0.21%      23.517us         2.83%     325.079us      40.635us             8  
                                      aten::transpose         2.08%     238.066us         2.83%     324.936us       1.934us           168  
                                             aten::to         0.53%      61.326us         2.53%     290.195us       3.298us            88  
                                         aten::matmul         0.37%      42.273us         2.31%     264.612us      33.077us             8  
                                       aten::_to_copy         1.09%     124.756us         2.00%     228.869us       3.576us            64  
                                          aten::split         0.39%      44.441us         1.79%     204.810us      12.801us            16  
                                             aten::mm         1.75%     200.238us         1.76%     202.261us      25.283us             8  
                                            aten::add         1.22%     139.362us         1.72%     196.779us       2.733us            72  
                                           aten::view         1.68%     193.061us         1.68%     193.061us       0.589us           328  
                                     aten::as_strided         1.60%     183.558us         1.60%     183.558us       0.534us           344  
                                         aten::expand         1.17%     134.506us         1.47%     168.396us       2.631us            64  
                                          aten::empty         1.44%     164.896us         1.44%     164.896us       0.859us           192  
                                         aten::narrow         0.55%      63.255us         1.40%     160.369us       3.341us            48  
-----------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 11.469ms

```

Raw CSV: `results/Colab_H100/summary/torch_profile_ops.csv`
