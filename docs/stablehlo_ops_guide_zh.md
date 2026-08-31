# StableHLO 全部 108 个算子中文指南

> 基于 StableHLO 官方 `main` 分支提交 `b7c389d080653facae294ec49f6f0374e0672f0a`（2026-08-28）。
>
> 官方规范：[StableHLO Specification](https://openxla.org/stablehlo/spec?hl=zh-cn#ops)

## 阅读约定

- 示例主要展示**值和形状如何变化**，不是每一处都写成完整 MLIR。
- `tensor<2x3xf32>` 表示形状 `[2,3]`、元素类型 `f32`。
- StableHLO 是纯张量语义；layout、kernel、CUDA stream 等由后续 XLA 编译阶段决定。
- 动态版本通常与静态版本语义相同，只是 shape、padding、slice size 等参数改为运行时 tensor。

---

## 1. 基础逐元素算术与数学函数

这些算子通常对 tensor 的每个元素独立计算，因此最容易被 XLA 融合成一个 loop kernel。

### `abs`

绝对值。例：`[-2.0, 0.0, 3.0] -> [2.0, 0.0, 3.0]`。复数输入得到模长，例如 `abs(3+4i)=5`。

### `add`

逐元素加法。例：`[1,2] + [10,20] -> [11,22]`。`stablehlo.add` **不执行隐式广播**：规范对非量化 tensor 写作 `type(lhs) = type(rhs) = type(result)`。当前 MLIR 实现分别检查相同元素类型和兼容 shape，因此 `tensor<?x3xf32>` 与 `tensor<2x3xf32>` 这类动态维度细化可以兼容，但 `[2,3]` 与 `[3]` 这种 rank/shape 差异不是广播，仍然非法。如果前端语言允许 NumPy 风格广播，前端必须先显式插入 `broadcast_in_dim`，再执行 `add`。

### `atan2`

逐元素计算带象限信息的反正切 `atan2(y,x)`。例：`atan2(1,1)=π/4`，`atan2(1,-1)=3π/4`。

### `cbrt`

立方根。例：`[-8,1,27] -> [-2,1,3]`。

### `ceil`

向正无穷取整。例：`[-1.7,1.2] -> [-1,2]`。

### `clamp`

把值限制在 `[min,max]`。例：`clamp(0,[−2,0.5,8],6) -> [0,0.5,6]`。

### `cosine`

逐元素余弦。例：`cosine([0,π]) -> [1,-1]`。

### `divide`

逐元素除法。例：`[6,8] / [2,4] -> [3,2]`。整数除法和除零行为要按元素类型语义理解。

### `exponential`

逐元素 `e^x`。例：`[0,1] -> [1,e]`。

### `exponential_minus_one`

逐元素 `exp(x)-1`，即常见的 `expm1`；对很小的 `x` 比直接 `exp(x)-1` 更稳定。

### `floor`

向负无穷取整。例：`[-1.2,1.8] -> [-2,1]`。

### `is_finite`

判断是否为有限值。例：`[1,+inf,nan] -> [true,false,false]`。

### `log`

自然对数。例：`log([1,e]) -> [0,1]`。

### `log_plus_one`

计算 `log(1+x)`，即 `log1p`。例：`log_plus_one(0.001)` 比 `log(1.001)` 更适合保持小数精度。

### `logistic`

Sigmoid：`1 / (1 + exp(-x))`。例：`logistic(0)=0.5`。

### `maximum`

逐元素最大值。例：`maximum([1,5],[3,2]) -> [3,5]`。

### `minimum`

逐元素最小值。例：`minimum([1,5],[3,2]) -> [1,2]`。

### `multiply`

逐元素乘法。例：`[1,2,3] * [10,20,30] -> [10,40,90]`。

### `negate`

逐元素取负。例：`[1,-2] -> [-1,2]`。

### `power`

逐元素乘方。例：`power([2,3],[3,2]) -> [8,9]`。

### `remainder`

逐元素余数。例：`remainder([7,8],[3,3]) -> [1,2]`。负数行为不要直接假设等同 Python `%`。

### `round_nearest_afz`

舍入到最近整数，正好位于中点时**远离 0**。例：`[1.5,-1.5,2.4] -> [2,-2,2]`。

### `round_nearest_even`

银行家舍入，中点舍入到偶数。例：`[1.5,2.5,-1.5] -> [2,2,-2]`。

### `rsqrt`

倒数平方根 `1/sqrt(x)`。例：`rsqrt([1,4]) -> [1,0.5]`。

### `sign`

符号函数。例：`[-3,0,7] -> [-1,0,1]`；复数通常得到单位方向值。

### `sine`

逐元素正弦。例：`sine([0,π/2]) -> [0,1]`。

### `sqrt`

逐元素平方根。例：`sqrt([1,4,9]) -> [1,2,3]`。

### `subtract`

逐元素减法。例：`[10,20] - [1,3] -> [9,17]`。

### `tan`

逐元素正切。例：`tan([0,π/4]) -> [0,1]`。

### `tanh`

逐元素双曲正切。例：`tanh(0)=0`，常见于激活函数和 LSTM。

---

## 2. 比较、布尔与位运算

### `and`

布尔或整数逐元素按位与。例：`[true,false] and [true,true] -> [true,false]`；`6 & 3 = 2`。

### `compare`

按指定方向比较，如 `EQ/NE/LT/LE/GT/GE`。例：`compare LT [1,4] [2,3] -> [true,false]`。浮点还需要关注比较模式和 NaN。

### `count_leading_zeros`

统计整数二进制前导 0。例：8 位整数 `00010000` 的结果是 `3`。

### `not`

布尔逻辑非或整数按位取反。例：`not [true,false] -> [false,true]`。

### `or`

布尔或整数逐元素按位或。例：`4 | 3 = 7`。

### `popcnt`

统计整数二进制中 1 的数量。例：`13 = 1101₂`，所以 `popcnt(13)=3`。

### `select`

逐元素条件选择：`pred ? on_true : on_false`。例：`select([T,F],[1,2],[10,20]) -> [1,20]`。

### `shift_left`

左移。例：`3 << 2 = 12`。

### `shift_right_arithmetic`

算术右移，保留有符号数符号位。例：8 位 `-8 >> 1 = -4`。

### `shift_right_logical`

逻辑右移，高位补 0。例：8 位 `11111000 >> 1 -> 01111100`。

### `xor`

布尔异或或整数按位异或。例：`6 xor 3 = 5`。

---

## 3. 类型、复数、精度与量化

### `bitcast_convert`

保持底层 bit，改变解释方式；不是数值转换。例：`f32 1.0` 的 bit `0x3f800000` bitcast 成 `ui32` 得 `1065353216`。若元素 bit 宽改变，还可能改变末尾维度。

### `complex`

用实部和虚部构造复数。例：`complex([1,2],[3,4]) -> [1+3i,2+4i]`。

### `convert`

数值类型转换。例：`f32 [1.8,-2.2] -> si32 [1,-2]`；与 `bitcast_convert` 的核心区别是它转换数值而不是保留 bit 模式。

### `imag`

提取复数虚部。例：`imag([1+3i,2-4i]) -> [3,-4]`。

### `real`

提取复数实部。例：`real([1+3i,2-4i]) -> [1,2]`。

### `reduce_precision`

在保持 tensor 类型不变的情况下，模拟较低浮点精度。例：把 `f32` 临时限制为较少 exponent/mantissa bits，用于模拟低精度误差。

### `uniform_quantize`

把浮点 tensor 按量化类型里的 scale 和 zero-point 转成整数存储。例：scale=`0.1`、zero-point=`0` 时，`[0.2,1.3] -> [2,13]`。

### `uniform_dequantize`

量化值恢复到 expressed floating type。例：scale=`0.1`、zero-point=`0` 时，量化 `[2,13] -> f32 [0.2,1.3]`。

---

## 4. 常量、Shape 与数据重排

### `constant`

在 IR 中创建字面量 tensor。例：`dense<[1,2,3]> : tensor<3xi32>`。

### `broadcast_in_dim`

把低 rank tensor 广播到高 rank，并显式指定原维度映射。例：`bias:[3]` 通过 `broadcast_dimensions=[1]` 广播成 `[2,3]`：两行都为 bias。StableHLO 的逐元素二元 op 不隐式广播，因此 `tensor<2x3xf32> + tensor<3xf32>` 必须先把后者广播成 `tensor<2x3xf32>`。

### `dynamic_broadcast_in_dim`

与 `broadcast_in_dim` 相同，但输出 shape 是运行时 tensor。例：运行时 `output_dimensions=[batch,3]`，把 `[3]` bias 广播成 `[batch,3]`。

### `concatenate`

沿某个维度拼接。例：`[1,2] concat [3,4,5] -> [1,2,3,4,5]`。

### `dynamic_iota`

与 `iota` 相同，但输出 shape 运行时确定。例：运行时 shape `[B,4]`、维度 0，得到每一行编号 `0..B-1`。

### `dynamic_pad`

与 `pad` 相同，但 low/high/interior padding 都是运行时值。

### `dynamic_reshape`

与 `reshape` 相同，但结果 shape 由运行时 `output_shape` 指定。元素总数仍必须一致。

### `dynamic_slice`

从运行时起点取固定大小切片。例：`x=[0,1,2,3,4]`，`start=2`、`size=2`，结果 `[2,3]`。

### `dynamic_update_slice`

从运行时位置覆盖一块区域。例：`[0,0,0,0,0]` 在 `start=2` 写入 `[7,8]`，结果 `[0,0,7,8,0]`。

### `get_dimension_size`

读取某一维的运行时长度。例：动态 shape `[?,128]` 的 dim 0 实际是 32，返回标量 `32`。

### `get_tuple_element`

从 tuple 取第 N 项。例：tuple `(tensorA,tensorB)`，index 1 得到 `tensorB`。

### `iota`

沿指定维度生成递增坐标。例：shape `[2,3]`、iota dim 1，结果 `[[0,1,2],[0,1,2]]`。

### `pad`

增加边缘或元素间 padding。例：`[1,2]`，low=1、high=2、pad_value=0，得到 `[0,1,2,0,0]`；interior=1 得 `[1,0,2]`。

### `reshape`

只改变 shape，不改变元素线性顺序。例：`[[1,2,3],[4,5,6]] : [2,3] -> [1,2,3,4,5,6] : [6]`。

### `reverse`

沿指定维度反转。例：`[1,2,3,4] -> [4,3,2,1]`。

### `slice`

使用静态 `start/limit/stride` 切片。例：`[0,1,2,3,4,5]`，`start=1,limit=6,stride=2 -> [1,3,5]`。

### `transpose`

重排维度。例：矩阵 `[2,3]` 使用 permutation `[1,0]` 后成为 `[3,2]`。

### `tuple`

把多个值包装成 tuple。例：`tuple(logits,kv_cache,token)`；它不是把 tensor 在某一维拼起来。

---

## 5. 线性代数、卷积与频域运算

### `cholesky`

对 Hermitian/Symmetric positive-definite 矩阵做 Cholesky 分解。例：`A = L·Lᵀ`，输出下三角 `L`。

### `convolution`

通用 N 维卷积，可描述 CNN 卷积、分组卷积、步长、padding、dilation。例：`input[N,H,W,C] * kernel[KH,KW,C,O] -> output[N,OH,OW,O]`。

### `dynamic_conv`

与 `convolution` 功能相同，但 `padding` 是运行时 tensor。适合输入尺寸决定 SAME padding 的场景。

### `dot_general`

广义点积，通过 `contracting_dimensions` 和 `batching_dimensions` 描述矩阵乘、batch matmul 等。例：`[B,M,K] × [B,K,N] -> [B,M,N]`。

### `fft`

快速傅里叶变换，支持 FFT/IFFT/RFFT/IRFFT。例：实数时间信号 `[N]` 经 RFFT 得频域复数 `[N/2+1]`。

### `triangular_solve`

解三角线性方程。例：给定下三角 `L` 和 `B`，求 `X` 使 `L·X=B`；常与 Cholesky 配套。

---

## 6. Gather、Scatter 与索引更新

### `gather`

按照索引从 operand 抽取切片，是 `index_select`、embedding lookup、高维 gather 的统一形式。例：embedding `table[5,2]`、indices `[3,1]`，结果为第 3、1 行，shape `[2,2]`。

### `dynamic_gather`

与 `gather` 相同，但 `slice_sizes` 在运行时给出。例：索引起点固定，但每次请求抽取的窗口大小不同。

### `scatter`

按照索引把 updates 合并回 operand，合并函数可以是覆盖、加法、最大值等。例：base=`[0,0,0,0]`，indices=`[1,3]`，updates=`[5,7]`，覆盖式结果 `[0,5,0,7]`。

### `select_and_scatter`

先对滑动窗口用 `select` 选中位置，再把 source 值 scatter 回这些位置。典型用途是 max-pool backward：前向窗口最大值的位置接收梯度。

---

## 7. Reduction、窗口与高阶张量计算

### `map`

把一个小 computation 映射到 tensor 各元素。例：body=`x*x+1`，输入 `[1,2,3]`，输出 `[2,5,10]`。现代编译流程中很多 map 会被规范化为普通逐元素 HLO。

### `reduce`

沿指定维度用二元函数聚合。例：`[[1,2,3],[4,5,6]]` 沿 dim 1 做 add，结果 `[6,15]`。reducer 也可以是 max、min 或多输入 tuple reducer。

### `reduce_window`

在滑动窗口内做 reduce。例：一维 `[1,3,2,4]`，window=2、stride=2、max，结果 `[3,4]`；max-pool、avg-pool 可由它表达。

### `sort`

沿指定维度排序，可同时携带多个 tensor。例：keys=`[3,1,2]`、values=`[30,10,20]`，按 keys 排序后得到 keys=`[1,2,3]`、values=`[10,20,30]`。

---

## 8. Batch Normalization

### `batch_norm_inference`

推理 BN：使用已有 mean/variance/scale/offset。公式近似 `y=(x-mean)/sqrt(var+epsilon)*scale+offset`。

### `batch_norm_training`

训练 BN：从当前 batch 计算 mean 和 variance，同时输出归一化结果。例：输入 `[N,H,W,C]`，按 feature dim `C` 统计。

### `batch_norm_grad`

BN 反向传播，输入 `x/scale/mean/variance/grad_output`，输出对 input、scale、offset 的梯度。

---

## 9. 控制流、封装与编译边界

### `case`

多分支控制流，通过整数 branch index 选择 region。例：index=0 执行 `x+1`，index=1 执行 `x*2`，其余走最后一个分支。

### `if`

二分支控制流，predicate 为布尔标量。例：`if training then dropout(x) else x`。

### `while`

循环，包含 `cond` 和 `body` region。例：状态 `(i,sum)` 从 `(0,0)` 开始，在 `i<10` 时执行 `(i+1,sum+i)`。

### `composite`

用一组 StableHLO op 的 decomposition 表示一个有名字、有版本的复合操作。例：`my.gelu@v1` 可以展开成 `multiply + erf + add`；编译器不认识名字时仍能展开执行。

### `custom_call`

调用实现定义的外部能力。例：调用 cuDNN、用户 kernel 或 host callback。`call_target_name` 指定实现，`backend_config` 带配置；如果标记 `has_side_effect`，优化器不能随意删除或重排。

### `optimization_barrier`

阻止某些优化跨越边界，但数值结果与输入相同。例：`y=barrier(x)`，可用于防止 constant folding、强制观察调度或构造 benchmark。

---

## 10. 异步操作

### `async_start`

请求异步执行一个受支持的 collective 或 slice 操作，返回 `future<T>`。例：启动 `all_gather(x)` 后继续做独立计算；后端可以选择实际仍同步执行，因此异步不是绝对保证。

### `async_done`

等待 `future<T>` 完成并取回 `T`。例：`future = async_start(all_gather(x))`，稍后 `gathered = async_done(future)`。

---

## 11. 随机数

### `rng`

按 UNIFORM 或 NORMAL 分布生成随机 tensor。例：`rng(a=0,b=1,shape=[2,3],UNIFORM)`。具体算法和隐藏状态由实现决定，规范中已将它视为趋向废弃的接口。

### `rng_bit_generator`

显式输入、输出 RNG state，生成随机 bit tensor。例：`(new_state,bits)=rng_bit_generator(state,shape=[1024])`；比 `rng` 更适合要求可复现和显式状态管理的实现。

---

## 12. Token、设备 I/O 与点对点通信

### `after_all`

把多个 token 合并成一个 token，表达“这些副作用都完成后再继续”。例：`t = after_all(send_token, outfeed_token)`。

### `infeed`

从设备外部 infeed queue 接收数据，同时传递 token 保证顺序。例：设备循环中读取下一训练 batch。

### `outfeed`

把数据写到设备外部 outfeed queue，并使用 token 保证顺序。例：把设备上的调试 tensor 发送给 host。

### `send`

通过 channel 向另一 process/device 发送 tensor，返回 token 或发送状态。常与匹配的 `recv` 配对。

### `recv`

从 channel 接收 tensor。例：pipeline stage 1 `send(activation)`，stage 2 `recv()` 得到 activation。

---

## 13. 集合通信与分布式执行

先区分两个概念：

- replica：通常执行同一程序、处理不同数据副本。
- partition：同一个逻辑 tensor/程序经过 SPMD 切分后的不同分片。

### `all_gather`

每个 process 提供一个分片，沿指定维度拼接，并让组内所有 process 都得到完整结果。例：P0=`[a,b]`、P1=`[c,d]`，结果两边都是 `[a,b,c,d]`。

### `all_reduce`

组内所有 process 对 tensor 做 reduce，并让所有 process 都得到结果。例：P0=`[1,2]`、P1=`[10,20]`，sum 后两边都是 `[11,22]`。

### `all_to_all`

每个 process 把输入沿 `split_dimension` 切块，互相交换，再沿 `concat_dimension` 拼接。常用于 tensor parallel 中从一种 sharding layout 转换到另一种 layout。

### `collective_broadcast`

组内指定 root process 的 tensor 广播给所有成员。例：P0 持有模型配置 `[7,8]`，广播后 P0/P1/P2 都得到 `[7,8]`。

### `collective_permute`

按静态 source-target pair 重排数据。例：pairs=`[(0,1),(1,2),(2,0)]`，形成 ring rotation：P0→P1、P1→P2、P2→P0。

### `collective_reduce`

组内 reduce，但结果只保存在 root process；非 root 的规范结果为零 tensor。例：P0/P1/P2 分别持有 `1/2/3`，root=P0，sum 后 P0 得 `6`，其他得到 `0`。

### `reduce_scatter`

逻辑上先 all-reduce，再把结果沿指定维度切分给各 process。例：两个 process 对 `[a0,a1]`、`[b0,b1]` 求和后，P0 得 `a0+b0`，P1 得 `a1+b1`。

### `replica_id`

返回当前 replica 的整数编号。例：4 个 replicas 分别返回 `0,1,2,3`。

### `partition_id`

返回当前 SPMD partition 的整数编号。例：一个 tensor 切成 8 片，每个 partition 可用 ID 计算自己负责的区间。

---

## 14. 最容易混淆的几组

### `convert` 与 `bitcast_convert`

```text
convert:         数值 1.0f -> 整数 1
bitcast_convert: bit 0x3f800000 -> 整数 1065353216
```

### `reshape` 与 `transpose`

```text
reshape:   不改变线性元素顺序，只重新解释 shape
transpose: 改变维度坐标映射，通常改变访问顺序
```

### `slice` 与 `dynamic_slice`

```text
slice:         起点、终点、stride 编译期已知
dynamic_slice: 起点运行时给出，但 slice size 通常静态
```

### `gather` 与 `dynamic_gather`

```text
gather:         slice_sizes 是属性/常量
dynamic_gather: slice_sizes 是运行时 tensor
```

### `reduce` 与 `reduce_window`

```text
reduce:        整条指定维度聚合，例如 [M,N] -> [M]
reduce_window: 每个滑动窗口聚合，例如 pooling
```

### `all_reduce` 与 `reduce_scatter` 与 `collective_reduce`

```text
all_reduce:          所有人得到完整 reduce 结果
reduce_scatter:      每人得到 reduce 结果的一片
collective_reduce:   只有 root 得到完整 reduce 结果
```

### `composite` 与 `custom_call`

```text
composite:   有 StableHLO decomposition，可以展开成标准 op
custom_call: 语义由外部实现定义，可能无法用 StableHLO 完整展开
```

### `rng` 与 `rng_bit_generator`

```text
rng:               隐式状态、实现定义，复现性较弱
rng_bit_generator: 显式输入输出 state，更容易获得确定性
```

---

## 15. 从 StableHLO 到 XLA 的学习顺序

建议按下面顺序真正动手：

1. `constant/add/multiply/compare/select`
2. `broadcast_in_dim/reshape/transpose/slice/pad`
3. `dot_general/convolution`
4. `reduce/reduce_window/sort`
5. `gather/scatter`
6. `if/while/case`
7. `all_reduce/all_gather/reduce_scatter`
8. `custom_call/composite/async_start`

每组都可以写一个 `.mlir`，然后观察：

```text
StableHLO
  ↓ stablehlo-opt
canonicalized StableHLO
  ↓ StableHLO -> HLO
XLA HLO
  ↓ HLO passes
optimized HLO
```
