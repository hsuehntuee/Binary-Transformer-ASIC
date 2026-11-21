`timescale 1ns / 1ps

module consmax_core #(
    parameter EXP_SHIFT = 8, 
    parameter SCALE_SHIFT = 8
)(
    input  wire        clk, 
    input  wire        rst_n, 
    input  wire        valid_in, 
    input  wire [7:0]  score_in,
    input  wire [15:0] beta_scale, 
    output reg         valid_out, 
    output reg  [7:0]  p_out
);

    wire [3:0] idx_msb = score_in[7:4]; 
    wire [3:0] idx_lsb = score_in[3:0];

    // ==========================================
    // ConSmax exp-approximation LUTs (Synthesizable)
    // ==========================================
    
    reg [15:0] val_msb;
    always @(*) begin
        case (idx_msb)
            4'd0:  val_msb = 16'h0100;
            4'd1:  val_msb = 16'h0130;
            4'd2:  val_msb = 16'h016A;
            4'd3:  val_msb = 16'h01AE;
            4'd4:  val_msb = 16'h0200;
            4'd5:  val_msb = 16'h0260;
            4'd6:  val_msb = 16'h02D4;
            4'd7:  val_msb = 16'h035D;
            4'd8:  val_msb = 16'h0400;
            4'd9:  val_msb = 16'h04C1;
            4'd10: val_msb = 16'h05A8;
            4'd11: val_msb = 16'h06BA;
            4'd12: val_msb = 16'h0800;
            4'd13: val_msb = 16'h0983;
            4'd14: val_msb = 16'h0B50;
            4'd15: val_msb = 16'h0D74;
            default: val_msb = 16'h0100;
        endcase
    end

    reg [15:0] val_lsb;
    always @(*) begin
        case (idx_lsb)
            4'd0:  val_lsb = 16'h0100;
            4'd1:  val_lsb = 16'h0102;
            4'd2:  val_lsb = 16'h0105;
            4'd3:  val_lsb = 16'h0108;
            4'd4:  val_lsb = 16'h010B;
            4'd5:  val_lsb = 16'h010E;
            4'd6:  val_lsb = 16'h0111;
            4'd7:  val_lsb = 16'h0114;
            4'd8:  val_lsb = 16'h0117;
            4'd9:  val_lsb = 16'h011A;
            4'd10: val_lsb = 16'h011D;
            4'd11: val_lsb = 16'h0120;
            4'd12: val_lsb = 16'h0123;
            4'd13: val_lsb = 16'h0126;
            4'd14: val_lsb = 16'h0129;
            4'd15: val_lsb = 16'h012D;
            default: val_lsb = 16'h0100;
        endcase
    end

    // ==========================================
    // Datapath & Pipeline Registers
    // ==========================================

    wire [31:0] mult_lut = val_msb * val_lsb;
    
    reg [23:0] exp_reg; 
    reg [15:0] beta_reg; 
    reg        valid_exp;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin 
            exp_reg   <= 0; 
            beta_reg  <= 0; 
            valid_exp <= 0;
        end else begin 
            exp_reg   <= mult_lut[EXP_SHIFT +: 24]; 
            beta_reg  <= beta_scale; 
            valid_exp <= valid_in;
        end
    end

    wire [39:0] mult_scale = exp_reg * beta_reg;
    wire [23:0] s_wire = mult_scale[SCALE_SHIFT +: 24];
    
    // Saturation & ReLU
    wire [7:0] q_wire = (s_wire > 127) ? 8'd127 : (s_wire[23]) ? 8'd0 : s_wire[7:0];

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin 
            p_out     <= 0; 
            valid_out <= 0;
        end else begin 
            p_out     <= q_wire; 
            valid_out <= valid_exp; 
        end
    end

endmodule