#!/bin/bash
# Phase-3 FFN regression (VCS). Run from tb/. Self-cleaning.
vcs -full64 -sverilog -timescale=1ns/1ps -o simv_ffn -l ffn_compile.log \
    tb_ffn.v ../rtl/transformer/Transformer_Layer_Top.v \
    ../rtl/bitlinear/*.v ../rtl/attention/*.v > /dev/null 2>&1
./simv_ffn -l ffn_sim.log > /dev/null 2>&1
grep -E 'FFN:|PHASE-3|ffn2\]' ffn_sim.log
rm -rf simv_ffn simv_ffn.daidir csrc ucli.key ffn_compile.log ffn_sim.log \
       AN.DB DVEfiles .vcsmx_rebuild
