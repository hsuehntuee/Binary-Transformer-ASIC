`timescale 1ns / 1ps
module attention_core_top #(
    parameter DIM=32,
    ACC_W=19,
    P_SHIFT=3
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              attn_pair_start_i,
    input  wire              qk_valid_i,
    input  wire signed [7:0] q_dim_i,
    input  wire signed [7:0] k_dim_i,
    input  wire [15:0]       beta_scale,
    input  wire signed [7:0] v_data_in,
    output wire              dump_valid,
    output wire signed [7:0] dump_data
);
    wire s_valid, p_valid;
    wire signed [20:0] s_mac;
    wire [7:0] p_ij;

    // Shift s_mac >>>4 then clamp to [0,127]. Without the shift, INT8 dot
    // products (~514k max) saturate at 127 → all attention scores equal.
    wire [7:0] s_quant = s_mac[20] ? 8'd0 : (|s_mac[19:11]) ? 8'd127 : s_mac[10:4];

    stage1_qk_serial #(.DIM(DIM), .ACC_W(21)) u_stg1(
        clk, rst_n, attn_pair_start_i, qk_valid_i, q_dim_i, k_dim_i, s_valid, s_mac
    );
    consmax_core u_stg2(clk, rst_n, s_valid, s_quant, beta_scale, p_valid, p_ij);
    stage3_pv_serial #(.DIM(DIM), .ACC_W(ACC_W), .P_SHIFT(P_SHIFT)) u_stg3(
        clk, rst_n, p_valid, p_ij, v_data_in, dump_valid, dump_data
    );
endmodule
