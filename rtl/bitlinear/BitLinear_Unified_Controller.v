`timescale 1ns / 1ps

// BitLinear_Unified_Controller — BL datapath FSM for multi-token (Attention)
// and single-token (FFN) modes. Enables are combinational from state/counter
// so they're zero-latency aligned with chunk_idx.
module BitLinear_Unified_Controller #(
    parameter MODE_MULTI_TOKEN = 1  // 1: Attention, 0: FFN
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start_i,
    // Last-token index for MULTI_TOKEN runs (drive 15 for full 16-token,
    // 7 for the PROJ_V half-split). Ignored when MODE_MULTI_TOKEN=0.
    input  wire [3:0]  tok_limit_i,
    output reg         done_o,

    // Datapath Control Signals
    output reg         load_beta_en_o,
    output reg         load_act_wt0_en_o,
    output reg         load_wt_next_en_o,
    output reg         compute_en_o,
    output reg         act_lock_stats_o,
    output reg         act_clear_stats_o,
    output reg         mac_clear_o,
    output reg         dequant_start_o,

    // Index Signals
    output wire [1:0]  chunk_idx_o,
    output wire [4:0]  col_idx_o,
    output wire [3:0]  token_idx_o
);

    // FSM States
    localparam IDLE   = 3'd0,
               BETA   = 3'd1,
               ACT_W0 = 3'd2,
               COMP   = 3'd3,
               FINISH = 3'd4;

    reg [2:0] state, next_state;
    reg [6:0] counter;
    reg [4:0] col_cnt;
    reg [3:0] tok_cnt;

    assign chunk_idx_o = counter[1:0];
    assign col_idx_o   = col_cnt;
    assign token_idx_o = tok_cnt;

    // Next-state logic
    always @(*) begin
        next_state = state;
        case (state)
            IDLE:   if (start_i)          next_state = BETA;
            BETA:   if (counter == 1)     next_state = ACT_W0;
            ACT_W0: if (counter == 31)    next_state = COMP;
            COMP: begin
                if (counter == 3 && col_cnt == 31) begin
                    if (MODE_MULTI_TOKEN == 0 || tok_cnt == tok_limit_i)
                        next_state = FINISH;
                    else
                        next_state = ACT_W0;
                end
            end
            FINISH: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

    // Sequential state / counter updates
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= IDLE;
            counter <= 0;
            col_cnt <= 0;
            tok_cnt <= 0;
        end else begin
            state <= next_state;
            case (state)
                IDLE: begin
                    counter <= 0;
                    col_cnt <= 0;
                    tok_cnt <= 0;
                end
                BETA: begin
                    if (counter == 1) counter <= 0;
                    else              counter <= counter + 1;
                end
                ACT_W0: begin
                    if (counter == 31) counter <= 0;
                    else               counter <= counter + 1;
                end
                COMP: begin
                    if (counter == 3) begin
                        counter <= 0;
                        if (col_cnt == 31) begin
                            col_cnt <= 0;
                            if (MODE_MULTI_TOKEN != 0 && tok_cnt != tok_limit_i)
                                tok_cnt <= tok_cnt + 1;
                        end else begin
                            col_cnt <= col_cnt + 1;
                        end
                    end else begin
                        counter <= counter + 1;
                    end
                end
                FINISH: begin
                    counter <= 0;
                    col_cnt <= 0;
                    tok_cnt <= 0;
                end
                default: ;
            endcase
        end
    end

    // Output decode — zero-latency aligned with chunk_idx
    always @(*) begin
        done_o            = 1'b0;
        load_beta_en_o    = 1'b0;
        load_act_wt0_en_o = 1'b0;
        load_wt_next_en_o = 1'b0;
        compute_en_o      = 1'b0;
        act_lock_stats_o  = 1'b0;
        act_clear_stats_o = 1'b0;
        mac_clear_o       = 1'b0;
        dequant_start_o   = 1'b0;

        case (state)
            IDLE: begin
                if (start_i) act_clear_stats_o = 1'b1;
            end
            BETA: begin
                load_beta_en_o = 1'b1;
            end
            ACT_W0: begin
                load_act_wt0_en_o = 1'b1;
                if (counter == 0)  act_clear_stats_o = 1'b1;
                if (counter == 31) act_lock_stats_o  = 1'b1;
            end
            COMP: begin
                compute_en_o      = 1'b1;
                load_wt_next_en_o = (col_cnt < 31);
                if (counter == 0) mac_clear_o     = 1'b1;
                if (counter == 3) dequant_start_o = 1'b1;
            end
            FINISH: begin
                done_o = 1'b1;
            end
            default: ;
        endcase
    end
endmodule
