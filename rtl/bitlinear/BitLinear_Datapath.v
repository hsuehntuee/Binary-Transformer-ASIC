`timescale 1ns / 1ps

module BitLinear_Datapath (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        ctrl_load_beta_i,
    input  wire        ctrl_load_act_wt0_i,
    input  wire        ctrl_load_wt_next_i,
    input  wire        ctrl_compute_i,
    input  wire        ctrl_act_lock_i,
    input  wire        ctrl_act_clear_i,
    input  wire        ctrl_mac_clear_i,
    input  wire        ctrl_dequant_start_i,
    input  wire [1:0]  ctrl_chunk_idx_i,

    input  wire        config_pad_i,
    input  wire signed [7:0] act_pad_i,

    output wire [1:0]  dot_pad_o,
    output wire        out_valid_o,
    output wire [7:0]  dequant_out_o,
    output wire        dequant_valid_o,
    output wire        dequant_pre_valid_o,
    output wire [18:0] stat_sum_sq_o
);

    reg [15:0] sys_beta;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) sys_beta <= 16'd0;
        else if (ctrl_load_beta_i) sys_beta <= {sys_beta[7:0], act_pad_i};
    end

    wire [63:0]  stat_act_chunk;
    wire [18:0]  stat_sum_sq;
    wire [7:0]   wt_chunk_curr;
    wire signed [12:0] mac_result;
    
    assign stat_sum_sq_o = stat_sum_sq;

    INT8_Act_Buffer_Stats u_act_stats (
        .clk(clk), .rst_n(rst_n),
        .load_en_i(ctrl_load_act_wt0_i), 
        .rotate_en_i(ctrl_compute_i),   
        .clear_stats_i(ctrl_act_clear_i),
        .act_pad_i(act_pad_i), 
        .act_chunk_o(stat_act_chunk), 
        .sum_sq_o(stat_sum_sq) 
    );

    Weight_Buffer u_weight_buf (
        .clk(clk), .rst_n(rst_n),
        .load_wt0_en_i(ctrl_load_act_wt0_i), .load_next_en_i(ctrl_load_wt_next_i), 
        .config_pad_i(config_pad_i), .act_pad_i(act_pad_i),
        .chunk_idx_i(ctrl_chunk_idx_i), 
        .col_switch_i(ctrl_dequant_start_i), 
        .wt_chunk_curr_o(wt_chunk_curr)
    );

    INT8_MAC_Engine u_mac (
        .clk(clk), .rst_n(rst_n),
        .compute_en_i(ctrl_compute_i), .clear_i(ctrl_mac_clear_i),
        .act_chunk_i(stat_act_chunk),  
        .wt_chunk_i(wt_chunk_curr),
        .accum_result_o(mac_result)
    );

    Dequant_Formatter u_dequant (
        .clk(clk), .rst_n(rst_n), .start_i(ctrl_dequant_start_i), 
        .accum_result_i(mac_result), .sum_sq_i(stat_sum_sq), 
        .update_i(ctrl_act_lock_i), 
        .beta_i(sys_beta),
        .dequant_out_o(dequant_out_o), 
        .dequant_valid_o(dequant_valid_o),
        .pre_valid_o(dequant_pre_valid_o) 
    );

    // 2-bit Shifter
    reg [7:0] shift_out_reg;
    reg [1:0] shift_cnt;
    reg       out_valid_reg;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_out_reg <= 8'd0; shift_cnt <= 2'd0; out_valid_reg <= 1'b0;
        end else begin
            if (dequant_valid_o) begin
                shift_out_reg <= dequant_out_o;
                shift_cnt     <= 2'd3;
                out_valid_reg <= 1'b1;
            end else if (out_valid_reg) begin
                shift_out_reg <= {shift_out_reg[5:0], 2'b00};
                if (shift_cnt == 2'd0) out_valid_reg <= 1'b0;
                else shift_cnt <= shift_cnt - 2'd1;
            end
        end
    end

    assign dot_pad_o   = shift_out_reg[7:6];
    assign out_valid_o = out_valid_reg;

endmodule
