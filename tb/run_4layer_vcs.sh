#!/bin/bash
# 4-layer E2E (VCS). Run from tb/. Self-cleaning.
vcs -full64 -sverilog -timescale=1ns/1ps -o simv_4l -l l4_compile.log \
    tb_4layer.v ../rtl/transformer/Transformer_Layer_Top.v \
    ../rtl/bitlinear/*.v ../rtl/attention/*.v > /dev/null 2>&1
./simv_4l -l l4_sim.log > /dev/null 2>&1
grep -E 'Layer|errors=|E2E' l4_sim.log
rm -rf simv_4l simv_4l.daidir csrc ucli.key l4_compile.log l4_sim.log \
       AN.DB DVEfiles .vcsmx_rebuild
