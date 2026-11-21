`timescale 1ns / 1ps
// stage3_pv_serial — PV accumulator with in-COMP dump and back-to-back pairs.
//
// Per output token: 16 pairs × 32 cyc COMP each. acc[k] accumulates p*V[k] across
// pairs. On the LAST pair (t_base == DIM*(SEQ_LEN-1)), each COMP cycle ALSO emits
// the final acc[k] as a dump byte — clamp((acc[calc_cnt]+pv) >>> P_SHIFT) is the
// post-NBA value seen combinationally, so dump fully overlaps the last pair's
// compute (no separate dump state).
//
// first_pair = (t_base == 0): the COMP NBA OVERWRITES acc[k] with pv instead of
// accumulating, so acc need not be explicitly cleared at token start.
//
// attn_token_start_i is not needed — t_base auto-resets at last-pair last-iter.
module stage3_pv_serial #(
    parameter DIM     = 32,
    parameter SEQ_LEN = 16,
    parameter ACC_W   = 19,
    parameter P_SHIFT = 3
)(
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       p_valid,
    input  wire signed [7:0]          p_in,
    input  wire signed [7:0]          v_data_in,
    output reg                        dump_valid,
    output reg  signed [7:0]          dump_data
);
    reg signed [ACC_W-1:0] acc [0:DIM-1];
    reg signed [7:0]       p_reg;
    reg [4:0]              calc_cnt;
    reg [9:0]              t_base;

    localparam S_IDLE = 1'b0, S_COMP = 1'b1;
    reg state;

    wire signed [15:0]       pv     = p_reg * v_data_in;
    // $signed forces portable sign extension to ACC_W bits.
    wire signed [ACC_W-1:0]  pv_ext = $signed(pv);

    wire first_pair   = (t_base == 10'd0);
    wire is_last_pair = (t_base == DIM * (SEQ_LEN - 1));

    // Combinational "post-NBA" view of acc[calc_cnt] so the dump byte is the
    // fully-accumulated value, same cycle the acc NBA fires.
    wire signed [ACC_W-1:0]  new_acc      = first_pair ? pv_ext : (acc[calc_cnt] + pv_ext);
    wire signed [ACC_W-1:0]  new_acc_sh   = new_acc >>> P_SHIFT;
    wire signed [7:0]        new_acc_clip = (new_acc_sh >  127) ?  8'sd127 :
                                            (new_acc_sh < -127) ? -8'sd127 :
                                                                   new_acc_sh[7:0];

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; calc_cnt <= 0; t_base <= 0; p_reg <= 0;
            dump_valid <= 0; dump_data <= 0;
            for (i = 0; i < DIM; i = i + 1) acc[i] <= {ACC_W{1'b0}};
        end else begin
            dump_valid <= 1'b0;  // pulsed high only during last-pair COMP

            case (state)
                S_IDLE: begin
                    if (p_valid) begin
                        p_reg    <= p_in;
                        calc_cnt <= 5'd0;
                        state    <= S_COMP;
                    end
                end

                S_COMP: begin
                    acc[calc_cnt] <= new_acc;

                    if (is_last_pair) begin
                        dump_valid <= 1'b1;
                        dump_data  <= new_acc_clip;
                    end

                    if (calc_cnt == DIM-1) begin
                        t_base <= is_last_pair ? 10'd0 : (t_base + DIM);
                        if (p_valid) begin
                            // Back-to-back next pair (or next token's pair 0).
                            p_reg    <= p_in;
                            calc_cnt <= 5'd0;
                        end else begin
                            state <= S_IDLE;
                        end
                    end else begin
                        calc_cnt <= calc_cnt + 5'd1;
                    end
                end
            endcase
        end
    end

`ifdef DBG_STG3
    always @(posedge clk) if (dump_valid)
        $display("[stg3] dump byte=%0d data=%0d", calc_cnt, $signed(dump_data));
`endif
endmodule
