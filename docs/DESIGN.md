# Design notes

This file explains how the chip computes a binary-weight Transformer layer and why it is built this way. The RTL is in `rtl/`, the hardware-exact model in `software/hw_model_exact.py`, and the test-vector generator in `scripts/export_all.py`.

## 1. What runs on the chip and what runs on the host

The model is a 4-layer Transformer encoder with d_model = 32, an FFN width of 32 and a sequence length of 16 tokens (a 16x16 MNIST image read as 16 rows of 16 pixels, padded to 32). All weights inside the layers are binary (+1/-1) and all activations between blocks are INT8. The model has 26,492 parameters.

The chip holds the datapath of one layer. It has three modes, chosen on `mode_i` and started with a pulse on `start_i`:

| Mode | What it computes | Cycles |
| --- | --- | --- |
| PROJ | One 32x32 BitLinear for all 16 tokens (used four times per layer: Q, K, V, O) | about 1,283 per matrix |
| ATTN | Q.K scores, ConSmax and P.V for all 16 queries, two queries per invocation | 8,192 |
| FFN | FFN1, LeakyReLU and FFN2, token-pipelined | about 2,743 |

One layer therefore takes about 4 x 1,283 + 8,192 + 2,743, roughly 16,000 cycles, or about 0.16 ms at 100 MHz.

The host does the rest in FP32 or plain integer arithmetic: the input projection and positional embedding, the two residual adds (saturated to [-127, 127]), the mean pool and the classifier. It also runs the chip four times, streaming in the next layer's weights each time and feeding the layer output back as the next input. In this repository the host is the testbench `tb/tb_4layer.v` together with `scripts/run_e2e.py`.

## 2. Why only one layer, and the pad budget

The four layers have the same shape and differ only in their weights, betas and ConSmax constant. One layer's datapath can therefore run all four if the host streams in a new set of weights for each pass, and the silicon area stays that of a single layer. The cost is time and I/O: each layer is a separate run, and all weights cross the pads every time.

The chip sits in a 40-pin side-braze package. With 8 pins used for power and ground, the design uses 28 signal pads:

- 23 inputs: `clk`, `rst_n`, `mode_i[1:0]`, `start_i`, `data_a_i[7:0]`, `data_b_i[7:0]`, `weight_a_i`, `weight_b_i`
- 5 outputs: `serial_a_o[1:0]`, `serial_b_o[1:0]`, `output_valid_o`

The chip keeps no weights between runs: every layer's weights, and the K and V tensors for attention, come in through the pads. That makes the pad count the main constraint on the interface. The team reduced it in steps during integration, from 52 to 51, 48, 40 and finally 28 signal pads. The steps included moving V off-chip, removing a 4-cycle Q/K load latency, sharing flip-flops between the ATTN and FFN modes, and preloading two queries per attention call. The final interface works like this:

- Weights arrive one bit per cycle per BitLinear on `weight_a_i` and `weight_b_i`.
- Activations, betas, K and V arrive a byte at a time on the two 8-bit data buses.
- Results leave 2 bits per cycle, so an INT8 value takes 4 cycles on each serial output.
- Each mode times itself from internal counters. The host only sets `mode_i`, pulses `start_i` and watches `output_valid_o`; there are no per-token handshake pins.
- All pads are registered at the core boundary (`Transformer_Layer_Top.v`, input and output boundary registers) so that pad delay does not eat into the 10 ns cycle.

The pad ring itself (I/O, power and corner cells) is a foundry-library wrapper around `Transformer_Layer_Top` and is not part of this repository.

## 3. BitLinear without multipliers

With weights restricted to +1 and -1, a dot product needs only additions and negations. `INT8_MAC_Engine.v` processes 8 lanes per cycle:

- A weight bit of 1 passes the activation unchanged. A weight bit of 0 (meaning -1) passes its bitwise inverse `~a`.
- In two's complement `~a = -a - 1`, so each inverted lane is off by -1. The engine counts the zero weight bits (a 3-bit popcount) and adds that count back after an 8-input adder tree.
- A 13-bit accumulator sums four 8-lane chunks to form one 32-element dot product.

The activation buffer (`INT8_Act_Buffer_Stats.v`) holds the 32 activations of a token and rotates them by 8 each cycle so the same 8 lanes see every chunk. It maps -128 to -127, which keeps every INT8 value symmetric and avoids the one value whose negation overflows.

Two BitLinear datapaths are instantiated (`u_bl0`, `u_bl1`). In PROJ mode both load the same weight matrix and split the tokens 8 and 8. In FFN mode `u_bl0` runs FFN1 and `u_bl1` runs FFN2.

## 4. RMSNorm instead of LayerNorm, with a Newton inverse square root

Each BitLinear normalises its input by the root of its sum of squares, then scales by a learned constant:

    y = round( mac * beta / sqrt(sum x^2) )

This is RMSNorm folded into the output scale. LayerNorm would also need the mean and a subtraction per element. RMSNorm needs only the sum of squares, which the activation buffer accumulates while the token is loading, so the statistic is ready when the MAC starts.

`Newton_Scale_Generator.v` computes `beta / sqrt(sum x^2)` in a pipeline (stages 0 to 6 in the source) with no divider and no square-root unit:

1. A leading-zero detector finds the exponent `E` of the 19-bit sum of squares, and the value is normalised to a mantissa `M` in [1, 2).
2. A 32-entry table indexed by the top 5 mantissa bits gives a first estimate `y0` of `1/sqrt(M)`.
3. One Newton-Raphson step refines it: `y1 = y0 * (3/2 - M/2 * y0^2)`, in fixed point, with the constant `0x6000` standing for 3/2.
4. If the exponent is odd, the result is multiplied by `0x2D41`, which is 1/sqrt(2) with 14 fractional bits. The even part of the exponent becomes a shift.
5. The result is multiplied by beta (16-bit, Q4.12) and normalised to a 9-bit mantissa plus a shift amount.

`Dequant_Formatter.v` multiplies the MAC result by the 9-bit scale, shifts with one guard bit for round-half-up, and saturates to [-127, 127]. These are the only true multipliers in the BitLinear path, one per output value, against the 32 additions of the dot product.

## 5. ConSmax instead of softmax

Softmax needs the row maximum, an exponential per score and a division by the row sum. The maximum and the sum each need a full pass over the scores before any probability is available. ConSmax replaces the maximum and the sum with one learned constant `C` per layer: `p = exp2(s) * C`. Each probability depends only on its own score, so the attention core streams.

`consmax_core.v` evaluates it like this:

- The Q.K score from `stage1_qk_serial.v` (32 cycles per score, one dimension per cycle) is quantised to 7 bits: negative scores give 0, large scores saturate at 127.
- `2^(s/64)` is split into the high and low 4 bits of the score and read from two 16-entry tables: `2^(hi/4)` and `2^(lo/64)`, both in Q8.8. Their product is the exponential.
- The product is multiplied by `C` and clipped to [0, 127].

Two 16-entry tables replace one 128-entry table at the cost of one 16x16 multiply. `tb/consmax_lut/` keeps a copy of the table contents; the RTL has them as constants.

`stage3_pv_serial.v` accumulates `p * V` into 32 accumulators of 19 bits each and shifts the result right by 3 (`P_SHIFT`) before clipping to INT8. On the last of the 16 key positions it writes out each finished accumulator in the same cycle it is updated, so there is no separate dump phase. In the top level, a 32-byte buffer and a 2-bit serialiser stream each query's output while the next query's scores are computed.

## 6. LeakyReLU and the FFN ping-pong buffer

The FFN activation is LeakyReLU with slope 1/8, computed by an arithmetic shift:

    y = x >= 0 ? x : (x + 4) >>> 3

The `+4` makes the shift round to nearest instead of towards minus infinity, so small negative values do not all collapse to -1.

FFN1 and FFN2 run at the same time on the two BitLinear datapaths. FFN1's LeakyReLU outputs go into a 64-byte token buffer (`tok_buf`), which holds two tokens. FFN2 starts once the first token is complete, reads it from one half while FFN1 writes the next token into the other, and so stays one token behind. FFN1's output never leaves the chip. In ATTN mode the same buffer holds the two preloaded query vectors, since the two modes never run together.

## 7. A PyTorch model that matches the RTL bit for bit

The model was trained with the hardware datapath inside its forward pass (hardware-aware quantisation-aware training). `software/hw_model_exact.py` computes each block's forward output with integer NumPy code that copies the RTL operation by operation: the same Newton table and constants, the same bit widths and truncations, the same ConSmax tables, the same rounding and saturation. The backward pass uses a smooth float approximation of each block through straight-through estimators. The weights are trained for the arithmetic the chip performs, with no separate post-training quantisation step.

Two details needed attention during training:

- The attention score is floored before the ConSmax lookup. A plain `torch.floor` has zero gradient, so Q and K stopped learning. A straight-through estimator around the floor fixed this.
- Both residual adds saturate to [-127, 127], the same range as every other INT8 value in the design.

`scripts/export_all.py` contains the same integer functions. It runs the trained checkpoint on one MNIST image and writes every intermediate tensor of every layer as hex files: inputs, binary weights, betas, the Q/K/V/O, attention and FFN outputs, the residuals and the layer output. The testbenches load these files and compare each stage, so a mismatch points to the block that caused it.

## 8. Verification

- Per-mode testbenches: `tb_proj.v` (Q/K/V projection), `tb_attn.v` (attention), `tb_ffn.v` (FFN and residual), each checking every output byte.
- `tb_layer.v` chains all modes through one layer.
- `tb_4layer.v` runs all four layers with the chip output of each layer fed into the next, checks every stage, and writes the final layer output to `tb/pattern_full/rtl_l3_out.hex`. For MNIST test image 67 all 512 output bytes equal `tb/pattern_L3/layer_gold.hex`, and the host classifier predicts 4, the correct label.
- The team ran these tests in Synopsys VCS. The 4-layer end-to-end test was also repeated on the post-layout netlist with SDF back-annotation at 100 MHz, with zero errors. Those netlist and timing files are not in this repository.

## 9. Debugging history

The commit history of the project shows how the design converged:

- The first Newton-Raphson implementation lost precision. After the fix and a retrain, all 512 attention outputs matched the model.
- A 12-bit Newton version was tried and dropped for the 9-bit scale that is in the chip.
- The P.V shift was moved to the final dump and the accumulator was widened to 19 bits, so the result is rounded once at the end.
- The Q projection was overlapped with the Q.K stage, and later the Q/K projections were pipelined with attention.
- The pad count went from 52 to 28 as described in section 2.
- For gate-level simulation the testbenches were changed to use only pad signals. FFN2 starts inside the chip at a fixed point (when FFN1 has produced its first token), so the testbenches drive FFN2's weights after a fixed delay (`F2_KICK_DELAY = 171` cycles) instead of probing an internal signal.

## 10. Known limits

- One layer in silicon. Every layer's weights and the K and V tensors are streamed in through pads on each run, so pad bandwidth limits throughput more than the datapath does.
- The input projection, residuals, mean pool and classifier run on the host.
- Sequence length (16) and width (32) are fixed by parameters and buffer sizes.
- The synthesis and place-and-route scripts, the netlists and the layout are not part of this repository.
- The chip was submitted for fabrication in June 2026. This repository has no silicon measurements.
