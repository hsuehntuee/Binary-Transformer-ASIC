`timescale 1ns / 1ps

module Newton_Scale_Generator_Pipelined (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [18:0] sum_sq_i, 
    input  wire        update_i, 
    input  wire [15:0] beta_i,   
    
    output reg  [8:0]  scale_o,       
    output reg  [5:0]  total_shift_o  
);
    // stage 0: input buffer (isolate external adder tree from internal LZD)
    reg [18:0] sum_sq_q0;
    reg        update_q0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_sq_q0 <= 19'd0;
            update_q0 <= 1'b0;
        end else begin
            sum_sq_q0 <= sum_sq_i;
            update_q0 <= update_i;
        end
    end

    // stage 1: LZD + initial approximation LUT
    reg [4:0] E;
    always @(*) begin
        if(sum_sq_q0[18]) E=18;
        else if(sum_sq_q0[17]) E=17; else if(sum_sq_q0[16]) E=16;
        else if(sum_sq_q0[15]) E=15; else if(sum_sq_q0[14]) E=14; else if(sum_sq_q0[13]) E=13;
        else if(sum_sq_q0[12]) E=12;
        else if(sum_sq_q0[11]) E=11; else if(sum_sq_q0[10]) E=10;
        else if(sum_sq_q0[9])  E=9;  else if(sum_sq_q0[8])  E=8;  else if(sum_sq_q0[7])  E=7;
        else if(sum_sq_q0[6])  E=6;  else if(sum_sq_q0[5])  E=5;  else if(sum_sq_q0[4])  E=4;
        else if(sum_sq_q0[3])  E=3;  else if(sum_sq_q0[2])  E=2;
        else if(sum_sq_q0[1])  E=1;  
        else E=0;
    end

    wire [18:0] sum_norm = sum_sq_q0 << (18 - E);
    
    wire [4:0]  lut_idx = sum_norm[17:13];
    reg [9:0] y0_lut;
    always @(*) begin
        case(lut_idx)
            5'd0:y0_lut=10'd254;
            5'd1:y0_lut=10'd250; 5'd2:y0_lut=10'd247; 5'd3:y0_lut=10'd243;
            5'd4:y0_lut=10'd240; 5'd5:y0_lut=10'd236; 5'd6:y0_lut=10'd233; 5'd7:y0_lut=10'd230;
            5'd8:y0_lut=10'd228; 5'd9:y0_lut=10'd225; 5'd10:y0_lut=10'd222;5'd11:y0_lut=10'd220;
            5'd12:y0_lut=10'd217;5'd13:y0_lut=10'd215;5'd14:y0_lut=10'd212;5'd15:y0_lut=10'd210;
            5'd16:y0_lut=10'd208;5'd17:y0_lut=10'd206;5'd18:y0_lut=10'd204;5'd19:y0_lut=10'd202;
            5'd20:y0_lut=10'd200;5'd21:y0_lut=10'd198;5'd22:y0_lut=10'd196;5'd23:y0_lut=10'd194;
            5'd24:y0_lut=10'd193;5'd25:y0_lut=10'd191;5'd26:y0_lut=10'd189;5'd27:y0_lut=10'd188;
            5'd28:y0_lut=10'd186;5'd29:y0_lut=10'd185;5'd30:y0_lut=10'd183;5'd31:y0_lut=10'd182;
            default:y0_lut=10'd0;
        endcase
    end

    reg [9:0]  y0_q1;       
    reg [13:0] M_frac_q1;   
    reg [4:0]  E_q1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y0_q1 <= 0;
            M_frac_q1 <= 0; E_q1 <= 0;
        end else if (update_q0) begin
            y0_q1 <= y0_lut;
            M_frac_q1 <= sum_norm[17:4]; E_q1 <= E;
        end
    end

    // stage 2: y0²
    wire [19:0] y0_sq_10 = y0_q1 * y0_q1;
    wire [15:0] y0_sq = y0_sq_10[17:2];   

    reg [15:0] y0_sq_q2;
    reg [13:0] M_frac_q2;
    reg [9:0]  y0_q2;
    reg [4:0]  E_q2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y0_sq_q2 <= 0;
            M_frac_q2 <= 0; y0_q2 <= 0; E_q2 <= 0;
        end else begin
            y0_sq_q2 <= y0_sq;
            M_frac_q2 <= M_frac_q1; y0_q2 <= y0_q1; E_q2 <= E_q1;
        end
    end

    // stage 3: M × y0²
    wire [14:0] M_full_q2 = {1'b1, M_frac_q2};
    wire [30:0] m_y0_sq_raw = M_full_q2 * y0_sq_q2; 

    reg [9:0]  y0_q3;       
    reg [15:0] half_m_y0_sq_q3; 
    reg [4:0]  E_q3;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y0_q3 <= 0;
            half_m_y0_sq_q3 <= 0; E_q3 <= 0;
        end else begin
            y0_q3 <= y0_q2;
            half_m_y0_sq_q3 <= m_y0_sq_raw[29:15];
            E_q3 <= E_q2;
        end
    end

    // stage 4: Newton iteration y1 = y0 × (3/2 − M/2 × y0²)
    wire [15:0] diff = 16'h6000 - half_m_y0_sq_q3;
    wire [25:0] y1_10x16 = y0_q3 * diff;

    reg [15:0] y1_q4;
    reg [4:0]  E_q4;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y1_q4 <= 0;
            E_q4 <= 0;
        end else begin
            y1_q4 <= y1_10x16[23:8];
            E_q4  <= E_q3;
        end
    end

    // stage 5: odd-exponent compensation + shift computation
    wire signed [5:0] E_eff = $signed({1'b0, E_q4}) - 6'sd5;
    wire is_odd = E_eff[0];
    wire [31:0] y1_comp_raw = y1_q4 * 16'h2D41;
    wire signed [5:0] shift_signed = 6'sd8 + (E_eff >>> 1);

    reg [15:0] inv_M_q5;
    reg [5:0]  shift_q5;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            inv_M_q5 <= 0;
            shift_q5 <= 0;
        end else begin
            inv_M_q5 <= is_odd ?
            y1_comp_raw[29:14] : y1_q4;
            shift_q5 <= shift_signed; 
        end
    end

    // stage 6: beta × inv_M, normalize to 9-bit scale
    wire [31:0] scale_raw = beta_i * inv_M_q5;
    reg [4:0] L;
    always @(*) begin
        if(scale_raw[31]) L=31; else if(scale_raw[30]) L=30;
        else if(scale_raw[29]) L=29;
        else if(scale_raw[28]) L=28; else if(scale_raw[27]) L=27; else if(scale_raw[26]) L=26;
        else if(scale_raw[25]) L=25; else if(scale_raw[24]) L=24;
        else if(scale_raw[23]) L=23;
        else if(scale_raw[22]) L=22; else if(scale_raw[21]) L=21; else if(scale_raw[20]) L=20;
        else if(scale_raw[19]) L=19; else if(scale_raw[18]) L=18;
        else if(scale_raw[17]) L=17;
        else if(scale_raw[16]) L=16; else if(scale_raw[15]) L=15; else if(scale_raw[14]) L=14;
        else if(scale_raw[13]) L=13; else if(scale_raw[12]) L=12;
        else if(scale_raw[11]) L=11;
        else if(scale_raw[10]) L=10; else if(scale_raw[9])  L=9;  else if(scale_raw[8])  L=8;
        else if(scale_raw[7])  L=7;
        else if(scale_raw[6])  L=6;  else if(scale_raw[5])  L=5;
        else if(scale_raw[4])  L=4;  else if(scale_raw[3])  L=3;  else if(scale_raw[2])  L=2;
        else if(scale_raw[1])  L=1;  else if(scale_raw[0])  L=0;  else L=0;
    end

    wire [5:0] K = (L > 8) ? (L - 8) : 6'd0;
    wire signed [6:0] safe_shift = 7'sd18 + $signed({1'b0, shift_q5}) - $signed({1'b0, K});
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scale_o <= 0;
            total_shift_o <= 0;
        end else begin
            scale_o       <= scale_raw >> K;
            total_shift_o <= (safe_shift < 0) ? 6'd0 : safe_shift[5:0]; 
        end
    end
endmodule