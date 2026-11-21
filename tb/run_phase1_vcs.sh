#!/bin/bash
# Phase-1 PROJ regression (VCS). Run from tb/. Self-cleaning — leaves no
# build artifacts behind (per project workflow: VCS only, no /tmp/shared dumps).
vcs -full64 -sverilog -timescale=1ns/1ps -o simv_proj -l proj_compile.log \
    tb_proj.v ../rtl/transformer/Transformer_Layer_Top.v \
    ../rtl/bitlinear/*.v ../rtl/attention/*.v > /dev/null 2>&1
./simv_proj -l proj_sim.log > /dev/null 2>&1
grep -E 'PROJ |PHASE-1|idx=' proj_sim.log
rm -rf simv_proj simv_proj.daidir csrc ucli.key proj_compile.log proj_sim.log \
       AN.DB DVEfiles .vcsmx_rebuild
