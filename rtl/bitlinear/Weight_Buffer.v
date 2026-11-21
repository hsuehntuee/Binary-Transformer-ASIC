`timescale 1ns / 1ps

module Weight_Buffer (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        load_wt0_en_i,   
    input  wire        load_next_en_i,  
    input  wire        config_pad_i,    
    input  wire [7:0]  act_pad_i,       
    
    input  wire [1:0]  chunk_idx_i,     
    input  wire        col_switch_i,    
    output wire [7:0]  wt_chunk_curr_o  
);
    reg [31:0] bank_0, bank_1;
    reg compute_ptr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) compute_ptr <= 1'b0;
        else if (col_switch_i) compute_ptr <= ~compute_ptr;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bank_0 <= 32'd0;
            bank_1 <= 32'd0;
        end
        else if (load_wt0_en_i) begin
            bank_0 <= {bank_0[30:0], config_pad_i};
        end else if (load_next_en_i) begin
            if (compute_ptr) bank_0[chunk_idx_i*8 +: 8] <= act_pad_i;
            else             bank_1[chunk_idx_i*8 +: 8] <= act_pad_i;
        end
    end

    wire [31:0] current_bank = compute_ptr ? bank_1 : bank_0;
    assign wt_chunk_curr_o = current_bank[chunk_idx_i * 8 +: 8];

endmodule