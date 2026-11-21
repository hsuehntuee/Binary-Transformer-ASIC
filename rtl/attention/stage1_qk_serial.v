`timescale 1ns / 1ps
// stage1_qk_serial — QK^T dot product, 1 dim/cyc, 32 cyc per score.
module stage1_qk_serial #(parameter DIM=32, ACC_W=21)(
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       attn_pair_start_i,
    input  wire                       qk_valid_i,
    input  wire signed [7:0]          q_dim_i,
    input  wire signed [7:0]          k_dim_i,
    output reg                        s_valid,
    output reg  signed [ACC_W-1:0]    s_mac_out
);
    reg [4:0]               dim_cnt;
    reg signed [ACC_W-1:0]  acc;
    wire signed [15:0]      prod = $signed(q_dim_i) * $signed(k_dim_i);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dim_cnt <= 0; acc <= 0; s_valid <= 0; s_mac_out <= 0;
        end else begin
            s_valid <= 1'b0;
            if (attn_pair_start_i) begin
                dim_cnt <= 0; acc <= 0;
            end
            if (qk_valid_i) begin
                if (dim_cnt == DIM - 1) begin
                    s_valid   <= 1'b1;
                    s_mac_out <= acc + prod;
                    dim_cnt   <= 0;
                    acc       <= 0;
                end else begin
                    acc     <= acc + prod;
                    dim_cnt <= dim_cnt + 1'b1;
                end
            end
        end
    end
endmodule
