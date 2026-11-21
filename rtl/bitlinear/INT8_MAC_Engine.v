`timescale 1ns / 1ps

module INT8_MAC_Engine (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        compute_en_i, 
    input  wire        clear_i,     
    input  wire [63:0] act_chunk_i,  
    input  wire [7:0]  wt_chunk_i,   
    output wire signed [12:0] accum_result_o
);
    // 1's complement MAC: wt=1 → keep act, wt=0 (meaning −1) → ~act, then add back popcount(zeros)
    wire signed [7:0] mult_base [0:7];
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : mac_lane
            wire signed [7:0] act = act_chunk_i[i*8 +: 8];
            assign mult_base[i] = wt_chunk_i[i] ? act : ~act;
        end
    endgenerate

    wire [3:0] wt_zeros = (~wt_chunk_i[0]) + (~wt_chunk_i[1]) + (~wt_chunk_i[2]) + (~wt_chunk_i[3]) +
                          (~wt_chunk_i[4]) + (~wt_chunk_i[5]) + (~wt_chunk_i[6]) + (~wt_chunk_i[7]);

    wire signed [8:0]  sum_L1 [0:3];
    wire signed [9:0]  sum_L2 [0:1];
    wire signed [10:0] tree_sum_base;
    
    assign sum_L1[0] = mult_base[0] + mult_base[1];
    assign sum_L1[1] = mult_base[2] + mult_base[3];
    assign sum_L1[2] = mult_base[4] + mult_base[5]; 
    assign sum_L1[3] = mult_base[6] + mult_base[7];
    
    assign sum_L2[0] = sum_L1[0] + sum_L1[1];
    assign sum_L2[1] = sum_L1[2] + sum_L1[3];
    
    assign tree_sum_base = sum_L2[0] + sum_L2[1];
    
    wire signed [11:0] tree_sum_total = tree_sum_base + $signed({2'b00, wt_zeros});
    
    reg signed [12:0] accum;
    assign accum_result_o = accum + tree_sum_total;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) accum <= 13'sd0;
        else if (compute_en_i) begin
            if (clear_i) accum <= tree_sum_total;
            else         accum <= accum + tree_sum_total;
        end
    end
endmodule