`timescale 1ns / 1ps

module Dequant_Formatter (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start_i,        
    input  wire signed [12:0] accum_result_i, 
    input  wire [18:0] sum_sq_i,       
    input  wire        update_i,      
    input  wire [15:0] beta_i,     
    
    output reg  [7:0]  dequant_out_o,  
    output reg         dequant_valid_o,
    output wire        pre_valid_o     
);

    wire [8:0] trimmed_scale;
    wire [5:0] adj_shift;

    Newton_Scale_Generator_Pipelined u_scale_gen(
        .clk(clk), .rst_n(rst_n),
        .sum_sq_i(sum_sq_i), .update_i(update_i), .beta_i(beta_i),
        .scale_o(trimmed_scale), .total_shift_o(adj_shift)
    );

    // 3-cycle delay aligns accum with scale generator pipeline
    reg signed [12:0] accum_result_q1, accum_result_q2, accum_result_q3;
    reg               start_q1, start_q2, start_q3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum_result_q1 <= 13'sd0; accum_result_q2 <= 13'sd0; accum_result_q3 <= 13'sd0; 
            start_q1 <= 1'b0; start_q2 <= 1'b0; start_q3 <= 1'b0; 
        end else begin
            accum_result_q1 <= accum_result_i;
            accum_result_q2 <= accum_result_q1;
            accum_result_q3 <= accum_result_q2; 
            
            start_q1 <= start_i; 
            start_q2 <= start_q1;
            start_q3 <= start_q2; 
        end
    end

    wire signed [22:0] scaled_mac = accum_result_q3 * $signed({1'b0, trimmed_scale});

    // Guard bit for round-to-nearest: LSB after shift is the round bit
    wire signed [24:0] ext_mac_guard = {scaled_mac[22], scaled_mac, 1'b0};
    wire signed [24:0] shifted_guard = ext_mac_guard >>> adj_shift;
    
    wire round_bit = shifted_guard[0];
    wire signed [23:0] shifted_base = shifted_guard[24:1];
    
    wire signed [23:0] shifted_mac = shifted_base + round_bit;

    wire is_pos_ovf = (shifted_mac[23] == 1'b0) && (|shifted_mac[22:7]);
    wire is_neg_ovf = (shifted_mac[23] == 1'b1) && (~&shifted_mac[22:7]);

    wire [7:0] sat_mac = is_pos_ovf ? 8'sd127 :
                         is_neg_ovf ? -8'sd128 : shifted_mac[7:0];
                         
    wire is_neg_128 = (sat_mac == 8'h80);
    wire [7:0] final_int8 = {sat_mac[7:1], sat_mac[0] | is_neg_128};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dequant_out_o   <= 8'd0;
            dequant_valid_o <= 1'b0;
        end else begin
            dequant_out_o   <= final_int8;
            dequant_valid_o <= start_q3; 
        end
    end

    assign pre_valid_o = start_q3; 

endmodule