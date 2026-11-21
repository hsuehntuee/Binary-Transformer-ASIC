`timescale 1ns / 1ps

module INT8_Act_Buffer_Stats (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         load_en_i,     
    input  wire         rotate_en_i,   
    input  wire         clear_stats_i,
    input  wire signed [7:0] act_pad_i,   
    
    output wire [63:0]  act_chunk_o,   
    output wire [18:0]  sum_sq_o     
);
    reg signed [7:0] act_reg [0:31];
    reg [18:0] sum_x2;

    wire is_neg_128 = (act_pad_i == 8'h80);
    wire signed [7:0] safe_act = {act_pad_i[7:1], act_pad_i[0] | is_neg_128};
    
    wire signed [15:0] sq_val_full = safe_act * safe_act;
    wire [13:0] sq_val_14 = sq_val_full[13:0];

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 32; i = i + 1) act_reg[i] <= 8'sd0;
            sum_x2 <= 19'd0; 
        end else begin
            if (load_en_i) begin
                act_reg[0] <= safe_act;
                for (i = 1; i < 32; i = i + 1) act_reg[i] <= act_reg[i-1];
                
                if (clear_stats_i) sum_x2 <= sq_val_14;
                else               sum_x2 <= sum_x2 + sq_val_14;
            end else if (rotate_en_i) begin
                for (i = 0; i < 24; i = i + 1) act_reg[i] <= act_reg[i+8];
                for (i = 24; i < 32; i = i + 1) act_reg[i] <= act_reg[i-24];
            end else if (clear_stats_i) begin
                sum_x2 <= 19'd0;
            end
        end
    end

    assign sum_sq_o = sum_x2 + sq_val_14;
    
    genvar g;
    generate
        for (g = 0; g < 8; g = g + 1) begin : out_chunk
            assign act_chunk_o[g*8 +: 8] = act_reg[g];
        end
    endgenerate
endmodule