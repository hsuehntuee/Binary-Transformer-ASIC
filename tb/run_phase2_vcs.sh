#!/bin/bash
# Phase-2 ATTN regression (VCS). Run from tb/. Self-cleaning.
vcs -full64 -sverilog -timescale=1ns/1ps -o simv_attn -l attn_compile.log \
    tb_attn.v ../rtl/transformer/Transformer_Layer_Top.v \
    ../rtl/bitlinear/*.v ../rtl/attention/*.v > /dev/null 2>&1
./simv_attn -l attn_sim.log > /dev/null 2>&1
grep -E 'ATTN:|PHASE-2|idx=' attn_sim.log
rm -rf simv_attn simv_attn.daidir csrc ucli.key attn_compile.log attn_sim.log \
       AN.DB DVEfiles .vcsmx_rebuild
