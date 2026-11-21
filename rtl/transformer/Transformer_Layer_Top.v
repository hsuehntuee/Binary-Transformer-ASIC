`timescale 1ns / 1ps

// ============================================================================
// Transformer_Layer_Top — 28-pad, mode-based 1-bit transformer layer.
//
// Pads (28): clk, rst_n, mode_i[1:0], start_i, data_a_i[8], data_b_i[8],
//            weight_a_i, weight_b_i, serial_a_o[2], serial_b_o[2], output_valid_o.
//
// The chip self-times each mode from internal counters; the TB sets mode_i,
// pulses start_i, and watches output_valid_o — no per-token/pair handshake pads.
// Residual-1/2 (x+o, res1+ffn2) are computed off-chip in the TB.
//
// Modes (mode_i):
//   PROJ : Q/K/V/O. Both BLs load the SAME matrix; 16 tokens split 8/8 (BL0 ->
//          tokens 0-7 on serial_a, BL1 -> tokens 8-15 on serial_b). One ctrl.
//   ATTN : Q preloaded into tok_buf (2 queries/invocation); K full-byte on
//          data_a, V full-byte on data_b; stage1 -> consmax -> stage3, result
//          serialized on serial_a.
//   FFN  : F1 on BL0 || F2 on BL1, on-chip LeakyReLU + ping-pong buffer,
//          token-pipelined; F2 result on serial_a.
//
// Beta: each BL run takes its 16-bit beta on its own data pad during the 2-cyc
// BETA state (MSB byte first); attn_beta via a 2-cyc ATTN preamble on data_a.
// ============================================================================
module Transformer_Layer_Top #(
    parameter DIM     = 32,
    parameter SEQ_LEN = 16
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire [1:0]        mode_i,
    input  wire              start_i,
    input  wire signed [7:0] data_a_i,
    input  wire signed [7:0] data_b_i,
    input  wire              weight_a_i,
    input  wire              weight_b_i,
    output wire [1:0]        serial_a_o,
    output wire [1:0]        serial_b_o,
    output wire              output_valid_o
);
    localparam MODE_IDLE = 2'd0, MODE_PROJ = 2'd1, MODE_ATTN = 2'd2, MODE_FFN = 2'd3;

    // -------------------------------------------------------------------------
    // Input boundary registers (STA closure). Data pads are 1-cyc delayed; the
    // BL ctrl signals get a matching 1-cyc delay (below) so they stay aligned.
    // -------------------------------------------------------------------------
    reg [1:0]        mode_i_r;
    reg              start_i_r;
    reg signed [7:0] data_a_i_r, data_b_i_r;
    reg              weight_a_i_r, weight_b_i_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mode_i_r <= 2'd0; start_i_r <= 1'b0;
            data_a_i_r <= 8'sd0; data_b_i_r <= 8'sd0;
            weight_a_i_r <= 1'b0; weight_b_i_r <= 1'b0;
        end else begin
            mode_i_r     <= mode_i;     start_i_r    <= start_i;
            data_a_i_r   <= data_a_i;   data_b_i_r   <= data_b_i;
            weight_a_i_r <= weight_a_i; weight_b_i_r <= weight_b_i;
        end
    end

    wire proj_mode = (mode_i_r == MODE_PROJ);
    wire ffn_mode  = (mode_i_r == MODE_FFN);

    // =========================================================================
    // PROJ controller — single FSM, drives BOTH BLs in lockstep (token-split
    // 8+8). FFN controllers — F1 on BL0, F2 on BL1, both MULTI_TOKEN (16 tok),
    // F2 started one token behind F1 (pipeline). The BL ctrl is mode-muxed:
    // BL0 <= PROJ|F1, BL1 <= PROJ|F2.
    // =========================================================================
    wire        c_load_beta, c_load_act_wt0, c_load_wt_next, c_compute;
    wire        c_act_lock, c_act_clear, c_mac_clear, c_dequant_start;
    wire [1:0]  c_chunk_idx;

    BitLinear_Unified_Controller #(.MODE_MULTI_TOKEN(1)) u_proj_ctrl (
        .clk(clk), .rst_n(rst_n),
        .start_i(start_i_r & proj_mode), .tok_limit_i(4'd7), .done_o(),
        .load_beta_en_o(c_load_beta), .load_act_wt0_en_o(c_load_act_wt0),
        .load_wt_next_en_o(c_load_wt_next), .compute_en_o(c_compute),
        .act_lock_stats_o(c_act_lock), .act_clear_stats_o(c_act_clear),
        .mac_clear_o(c_mac_clear), .dequant_start_o(c_dequant_start),
        .chunk_idx_o(c_chunk_idx), .col_idx_o(), .token_idx_o()
    );

    // ---- FFN sequencing: F1 starts on FFN start; F2 starts after F1's first
    //      token is buffered (f1_cnt reaches 32). ----
    wire signed [7:0] bl0_dequant_out;  wire bl0_dequant_valid;
    reg [9:0] f1_cnt;   // BL0 F1 leaky-output count (write index into tok_buf)
    reg [9:0] f2_cnt;   // BL1 F2 act-load count    (read index from tok_buf)
    reg       f2_kick, f2_armed;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin f1_cnt<=0; f2_kick<=0; f2_armed<=0; end
        else begin
            f2_kick <= 1'b0;
            if (!ffn_mode) begin f1_cnt<=0; f2_armed<=0; end
            else begin
                if (bl0_dequant_valid) f1_cnt <= f1_cnt + 10'd1;
                if (!f2_armed && f1_cnt >= 10'd32) begin f2_kick<=1'b1; f2_armed<=1'b1; end
            end
        end
    end

    wire f1_load_beta, f1_load_act_wt0, f1_load_wt_next, f1_compute;
    wire f1_act_lock, f1_act_clear, f1_mac_clear, f1_dequant_start;
    wire [1:0] f1_chunk;
    BitLinear_Unified_Controller #(.MODE_MULTI_TOKEN(1)) u_f1_ctrl (
        .clk(clk), .rst_n(rst_n),
        .start_i(start_i_r & ffn_mode), .tok_limit_i(4'd15), .done_o(),
        .load_beta_en_o(f1_load_beta), .load_act_wt0_en_o(f1_load_act_wt0),
        .load_wt_next_en_o(f1_load_wt_next), .compute_en_o(f1_compute),
        .act_lock_stats_o(f1_act_lock), .act_clear_stats_o(f1_act_clear),
        .mac_clear_o(f1_mac_clear), .dequant_start_o(f1_dequant_start),
        .chunk_idx_o(f1_chunk), .col_idx_o(), .token_idx_o()
    );

    wire f2_load_beta, f2_load_act_wt0, f2_load_wt_next, f2_compute;
    wire f2_act_lock, f2_act_clear, f2_mac_clear, f2_dequant_start;
    wire [1:0] f2_chunk;
    BitLinear_Unified_Controller #(.MODE_MULTI_TOKEN(1)) u_f2_ctrl (
        .clk(clk), .rst_n(rst_n),
        .start_i(f2_kick), .tok_limit_i(4'd15), .done_o(),
        .load_beta_en_o(f2_load_beta), .load_act_wt0_en_o(f2_load_act_wt0),
        .load_wt_next_en_o(f2_load_wt_next), .compute_en_o(f2_compute),
        .act_lock_stats_o(f2_act_lock), .act_clear_stats_o(f2_act_clear),
        .mac_clear_o(f2_mac_clear), .dequant_start_o(f2_dequant_start),
        .chunk_idx_o(f2_chunk), .col_idx_o(), .token_idx_o()
    );

    // ---- mode-muxed BL control (BL0 <= PROJ|F1, BL1 <= PROJ|F2) ----
    wire bl0_lb  = proj_mode ? c_load_beta    : (ffn_mode ? f1_load_beta    : 1'b0);
    wire bl0_law = proj_mode ? c_load_act_wt0 : (ffn_mode ? f1_load_act_wt0 : 1'b0);
    wire bl0_lwn = proj_mode ? c_load_wt_next : (ffn_mode ? f1_load_wt_next : 1'b0);
    wire bl0_cmp = proj_mode ? c_compute      : (ffn_mode ? f1_compute      : 1'b0);
    wire bl0_alk = proj_mode ? c_act_lock     : (ffn_mode ? f1_act_lock     : 1'b0);
    wire bl0_acl = proj_mode ? c_act_clear    : (ffn_mode ? f1_act_clear    : 1'b0);
    wire bl0_mcl = proj_mode ? c_mac_clear    : (ffn_mode ? f1_mac_clear    : 1'b0);
    wire bl0_dqs = proj_mode ? c_dequant_start: (ffn_mode ? f1_dequant_start: 1'b0);
    wire [1:0] bl0_chk = proj_mode ? c_chunk_idx : (ffn_mode ? f1_chunk : 2'd0);

    wire bl1_lb  = proj_mode ? c_load_beta    : (ffn_mode ? f2_load_beta    : 1'b0);
    wire bl1_law = proj_mode ? c_load_act_wt0 : (ffn_mode ? f2_load_act_wt0 : 1'b0);
    wire bl1_lwn = proj_mode ? c_load_wt_next : (ffn_mode ? f2_load_wt_next : 1'b0);
    wire bl1_cmp = proj_mode ? c_compute      : (ffn_mode ? f2_compute      : 1'b0);
    wire bl1_alk = proj_mode ? c_act_lock     : (ffn_mode ? f2_act_lock     : 1'b0);
    wire bl1_acl = proj_mode ? c_act_clear    : (ffn_mode ? f2_act_clear    : 1'b0);
    wire bl1_mcl = proj_mode ? c_mac_clear    : (ffn_mode ? f2_mac_clear    : 1'b0);
    wire bl1_dqs = proj_mode ? c_dequant_start: (ffn_mode ? f2_dequant_start: 1'b0);
    wire [1:0] bl1_chk = proj_mode ? c_chunk_idx : (ffn_mode ? f2_chunk : 2'd0);

    // 1-cyc delay aligns ctrl with the pad-registered data they consume.
    reg d0_lb,d0_law,d0_lwn,d0_cmp,d0_alk,d0_acl,d0_mcl,d0_dqs; reg [1:0] d0_chk;
    reg d1_lb,d1_law,d1_lwn,d1_cmp,d1_alk,d1_acl,d1_mcl,d1_dqs; reg [1:0] d1_chk;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            {d0_lb,d0_law,d0_lwn,d0_cmp,d0_alk,d0_acl,d0_mcl,d0_dqs,d0_chk} <= 0;
            {d1_lb,d1_law,d1_lwn,d1_cmp,d1_alk,d1_acl,d1_mcl,d1_dqs,d1_chk} <= 0;
        end else begin
            {d0_lb,d0_law,d0_lwn,d0_cmp,d0_alk,d0_acl,d0_mcl,d0_dqs,d0_chk} <=
              {bl0_lb,bl0_law,bl0_lwn,bl0_cmp,bl0_alk,bl0_acl,bl0_mcl,bl0_dqs,bl0_chk};
            {d1_lb,d1_law,d1_lwn,d1_cmp,d1_alk,d1_acl,d1_mcl,d1_dqs,d1_chk} <=
              {bl1_lb,bl1_law,bl1_lwn,bl1_cmp,bl1_alk,bl1_acl,bl1_mcl,bl1_dqs,bl1_chk};
        end
    end

    // ---- leaky-ReLU dequant + shared token buffer (tok_buf), 64 deep ----------
    // FFN : F1 leaky outputs -> tok_buf (read back by F2 as its activation).
    // ATTN: A_QLOAD loads 2 query vectors; A_RUN reads them as the stage1 Q.
    // Modes are sequential, so the buffer time-shares with no overlap. The WRITE
    // logic lives next to the ATTN FSM (below) which supplies the Q-load address.
    wire signed [7:0] leaky_f1 =
        bl0_dequant_out[7] ? ((bl0_dequant_out + 8'sd4) >>> 3) : bl0_dequant_out;
    reg [7:0] tok_buf [0:63];
    integer li;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          f2_cnt <= 0;
        else if (!ffn_mode)  f2_cnt <= 0;
        else if (d1_law)     f2_cnt <= f2_cnt + 10'd1;   // one read per BL1 act load
    end
    // BL1 activation: F2 ACT_W0 reads tok_buf; otherwise data_b pad (beta_f2
    // during BETA, Wf2 next-col bytes during COMP).
    wire signed [7:0] bl1_act_pad = (ffn_mode & d1_law) ? tok_buf[f2_cnt[5:0]] : data_b_i_r;

    // ---- BL0: PROJ tokens 0-7 / FFN F1 ----
    wire [1:0] bl0_dot;  wire bl0_dot_valid;
    BitLinear_Datapath u_bl0 (
        .clk(clk), .rst_n(rst_n),
        .ctrl_load_beta_i(d0_lb), .ctrl_load_act_wt0_i(d0_law),
        .ctrl_load_wt_next_i(d0_lwn), .ctrl_compute_i(d0_cmp),
        .ctrl_act_lock_i(d0_alk), .ctrl_act_clear_i(d0_acl),
        .ctrl_mac_clear_i(d0_mcl), .ctrl_dequant_start_i(d0_dqs),
        .ctrl_chunk_idx_i(d0_chk),
        .config_pad_i(weight_a_i_r), .act_pad_i(data_a_i_r),
        .dot_pad_o(bl0_dot), .out_valid_o(bl0_dot_valid),
        .dequant_out_o(bl0_dequant_out), .dequant_valid_o(bl0_dequant_valid),
        .dequant_pre_valid_o(), .stat_sum_sq_o()
    );

    // ---- BL1: PROJ tokens 8-15 / FFN F2 ----
    wire [1:0] bl1_dot;  wire bl1_dot_valid;
    BitLinear_Datapath u_bl1 (
        .clk(clk), .rst_n(rst_n),
        .ctrl_load_beta_i(d1_lb), .ctrl_load_act_wt0_i(d1_law),
        .ctrl_load_wt_next_i(d1_lwn), .ctrl_compute_i(d1_cmp),
        .ctrl_act_lock_i(d1_alk), .ctrl_act_clear_i(d1_acl),
        .ctrl_mac_clear_i(d1_mcl), .ctrl_dequant_start_i(d1_dqs),
        .ctrl_chunk_idx_i(d1_chk),
        .config_pad_i(weight_b_i_r), .act_pad_i(bl1_act_pad),
        .dot_pad_o(bl1_dot), .out_valid_o(bl1_dot_valid),
        .dequant_out_o(), .dequant_valid_o(), .dequant_pre_valid_o(), .stat_sum_sq_o()
    );

    // =========================================================================
    // ATTN mode — self-timed, buffer-Q, 2 queries per invocation.
    //   start -> A_BETA  (2 cyc, capture attn_beta on data_a)
    //         -> A_QLOAD (32 cyc, load Q0 on data_a + Q1 on data_b into tok_buf)
    //         -> A_RUN   (2 q x 16 k x 32 dim = 1024 cyc; K full-byte on data_a,
    //                     V full-byte on data_b; stage1 = 32 cyc/pair)
    //         -> A_DONE  (drain pipeline + final dump/serialize).
    //
    //   Q comes from tok_buf (preloaded), K from data_a_i_r, V from data_b_i_r;
    //   qk_valid is high every A_RUN cycle (full-byte, no nibble). The TB times V
    //   to land in stage3's COMP window (see tb_attn VOFF).
    // =========================================================================
    localparam A_IDLE=3'd0, A_BETA=3'd1, A_QLOAD=3'd2, A_RUN=3'd3, A_DONE=3'd4;
    wire attn_mode = (mode_i_r == MODE_ATTN);

    reg [2:0]  a_state;
    reg        a_beta_cnt;
    reg [15:0] beta_attn_reg;
    reg [5:0]  qload_cnt;       // 0..31 Q-load index within A_QLOAD
    reg [9:0]  rcnt;            // 0..1023 within A_RUN (q<<9 | key<<5 | dim)
    reg [8:0]  drain_cnt;

    reg [7:0]  q_dim_r;         // stage1 Q operand, read from tok_buf
    reg        qk_valid_r, pair_start_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_state <= A_IDLE; a_beta_cnt <= 1'b0; beta_attn_reg <= 16'd0;
            qload_cnt <= 6'd0; rcnt <= 10'd0; drain_cnt <= 9'd0;
            q_dim_r <= 8'd0; qk_valid_r <= 1'b0; pair_start_r <= 1'b0;
        end else begin
            qk_valid_r   <= 1'b0;
            pair_start_r <= 1'b0;
            case (a_state)
                A_IDLE: begin
                    rcnt <= 10'd0; a_beta_cnt <= 1'b0;
                    drain_cnt <= 9'd0; qload_cnt <= 6'd0;
                    if (start_i_r && attn_mode) a_state <= A_BETA;
                end
                A_BETA: begin
                    beta_attn_reg <= {beta_attn_reg[7:0], data_a_i_r};
                    a_beta_cnt <= a_beta_cnt + 1'b1;
                    if (a_beta_cnt == 1'b1) a_state <= A_QLOAD;
                end
                A_QLOAD: begin
                    qload_cnt <= qload_cnt + 6'd1;
                    if (qload_cnt == 6'd31) a_state <= A_RUN;
                end
                A_RUN: begin
                    rcnt         <= rcnt + 10'd1;
                    pair_start_r <= (rcnt[4:0] == 5'd0);
                    qk_valid_r   <= 1'b1;
                    q_dim_r      <= tok_buf[{rcnt[9], rcnt[4:0]}];
                    if (rcnt == 10'd1023) a_state <= A_DONE;
                end
                A_DONE: begin
                    drain_cnt <= drain_cnt + 9'd1;
                    if (drain_cnt == 9'd300) a_state <= A_IDLE;
                end
                default: a_state <= A_IDLE;   // unreachable; recover from SEU
            endcase
        end
    end

    // tok_buf write (shared FFN/ATTN; buffer declared above next to FFN reads).
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) for (li = 0; li < 64; li = li + 1) tok_buf[li] <= 0;
        else if (ffn_mode) begin
            if (bl0_dequant_valid) tok_buf[f1_cnt[5:0]] <= leaky_f1;
        end else if (attn_mode && a_state == A_QLOAD) begin
            tok_buf[qload_cnt]         <= data_a_i_r;   // Q0[dim]
            tok_buf[6'd32 + qload_cnt] <= data_b_i_r;   // Q1[dim]
        end
    end

    wire             attn_dump_valid;
    wire signed [7:0] attn_dump_data;

    attention_core_top #(.DIM(DIM)) u_attn (
        .clk(clk), .rst_n(rst_n),
        .attn_pair_start_i(pair_start_r),
        .qk_valid_i(qk_valid_r),
        .q_dim_i(q_dim_r), .k_dim_i(data_a_i_r),
        .beta_scale(beta_attn_reg),
        .v_data_in(data_b_i_r),
        .dump_valid(attn_dump_valid), .dump_data(attn_dump_data)
    );

    // ---- dump buffer (32 bytes/query) + 2-bit serializer on serial_a ----
    // stage3 dumps 32 bytes/query at 1 byte/cyc; the 2-bit pad needs 4 cyc/byte,
    // so buffer a query's dump then stream it (128 cyc), overlapping the next
    // query's stage1 (~1024 cyc) — single buffer, drained well before refill.
    reg [7:0] attn_buf [0:31];
    reg [4:0] dump_wr;
    reg       dump_valid_d;
    integer   bi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dump_wr <= 0; dump_valid_d <= 0;
            for (bi = 0; bi < 32; bi = bi + 1) attn_buf[bi] <= 0;
        end else begin
            dump_valid_d <= attn_dump_valid;
            if (attn_dump_valid) begin
                attn_buf[dump_wr] <= attn_dump_data;
                dump_wr <= dump_wr + 5'd1;
            end else dump_wr <= 0;
        end
    end
    wire dump_done = dump_valid_d & ~attn_dump_valid;   // 32 bytes buffered

    reg       ser_go;
    reg [6:0] ser_cnt;          // 0..127 (32 bytes x 4 chunks)
    reg       attn_ser_valid;
    reg [1:0] attn_ser_chunk;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ser_go <= 0; ser_cnt <= 0; attn_ser_valid <= 0; attn_ser_chunk <= 0;
        end else begin
            if (dump_done) begin ser_go <= 1; ser_cnt <= 0; end
            if (ser_go) begin
                attn_ser_valid <= 1'b1;
                case (ser_cnt[1:0])
                    2'd0: attn_ser_chunk <= attn_buf[ser_cnt[6:2]][7:6];
                    2'd1: attn_ser_chunk <= attn_buf[ser_cnt[6:2]][5:4];
                    2'd2: attn_ser_chunk <= attn_buf[ser_cnt[6:2]][3:2];
                    2'd3: attn_ser_chunk <= attn_buf[ser_cnt[6:2]][1:0];
                endcase
                if (ser_cnt == 7'd127) ser_go <= 0;
                ser_cnt <= ser_cnt + 7'd1;
            end else attn_ser_valid <= 1'b0;
        end
    end

    // -------------------------------------------------------------------------
    // Output mux + boundary registers (STA closure).
    //   PROJ : serial_a = BL0 (tokens 0-7), serial_b = BL1 (tokens 8-15).
    //   ATTN : serial_a = dump serializer; serial_b unused.
    //   FFN  : serial_a = BL1/F2 dot; serial_b unused.
    // -------------------------------------------------------------------------
    wire [1:0] sa_pre = attn_mode ? attn_ser_chunk : (ffn_mode ? bl1_dot : bl0_dot);
    wire [1:0] sb_pre = (attn_mode | ffn_mode) ? 2'b00 : bl1_dot;
    wire       ov_pre = attn_mode ? attn_ser_valid :
                        (ffn_mode ? bl1_dot_valid : (bl0_dot_valid | bl1_dot_valid));

    reg [1:0] serial_a_r, serial_b_r;
    reg       output_valid_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            serial_a_r <= 2'd0; serial_b_r <= 2'd0; output_valid_r <= 1'b0;
        end else begin
            serial_a_r     <= sa_pre;
            serial_b_r     <= sb_pre;
            output_valid_r <= ov_pre;
        end
    end
    assign serial_a_o     = serial_a_r;
    assign serial_b_o     = serial_b_r;
    assign output_valid_o = output_valid_r;

endmodule
