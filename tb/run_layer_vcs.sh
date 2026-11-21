#!/bin/bash
# Single-layer integration test (VCS). Run from tb/. Self-cleaning.
vcs -full64 -sverilog -timescale=1ns/1ps -o simv_layer -l layer_compile.log \
    tb_layer.v ../rtl/transformer/Transformer_Layer_Top.v \
    ../rtl/bitlinear/*.v ../rtl/attention/*.v > /dev/null 2>&1
./simv_layer -l layer_sim.log > /dev/null 2>&1
grep -E 'errors=|LAYER E2E' layer_sim.log
rm -rf simv_layer simv_layer.daidir csrc ucli.key layer_compile.log layer_sim.log \
       AN.DB DVEfiles .vcsmx_rebuild
