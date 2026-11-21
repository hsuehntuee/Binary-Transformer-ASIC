`timescale 1ns / 1ps
// ============================================================================
// tb_attn — phase-2 TB for Transformer_Layer_Top ATTN mode (buffer-Q, v2).
//
// 8 invocations, 2 queries each. Per invocation: A_BETA captures attn_beta, then
// A_QLOAD loads 2 query vectors into tok_buf (Q0=query 2i on data_a, Q1=query
// 2i+1 on data_b), then A_RUN streams K full-byte on data_a and V full-byte on
// data_b (32 pairs = 2 q x 16 k; stage1 = 32 cyc/pair). The 2 query dumps come
// out on serial_a back-to-back (Q0 hidden under Q1 compute, Q1 a short tail) and
// are checked against golden_attn.hex (all 16 queries, 512 bytes).
//
// Locked offsets (relative to the co-aligned start+da[0] event):
//   LEAD=1  start->beta gap        beta at da[1..2]
//   KBASE=36  K[key][dim] at da[KBASE + pair*32 + dim]
//   VBASE=71  V[key][dim] at db[VBASE + pair*32 + dim]   (V lags K by 35)
// ============================================================================
module tb_attn;
    localparam DIM = 32, SEQ = 16;
    localparam [1:0] MODE_IDLE=2'd0, MODE_PROJ=2'd1, MODE_ATTN=2'd2, MODE_FFN=2'd3;
    localparam integer LEAD  = 1;
    localparam integer KBASE = 36;
    localparam integer VBASE = 71;
    localparam integer NINV  = 8;       // invocations (2 queries each = 16)
    localparam integer INVC  = 1450;    // drive cycles per invocation

    reg              clk, rst_n;
    reg [1:0]        mode;
    reg              start;
    reg signed [7:0] data_a, data_b;
    reg              weight_a, weight_b;
    wire [1:0]       serial_a, serial_b;
    wire             output_valid;

    Transformer_Layer_Top #(.DIM(DIM), .SEQ_LEN(SEQ)) dut (
        .clk(clk), .rst_n(rst_n), .mode_i(mode), .start_i(start),
        .data_a_i(data_a), .data_b_i(data_b),
        .weight_a_i(weight_a), .weight_b_i(weight_b),
        .serial_a_o(serial_a), .serial_b_o(serial_b),
        .output_valid_o(output_valid)
    );

    initial clk = 0;  always #5 clk = ~clk;

    reg signed [7:0] Q [0:511], K [0:511], V [0:511], gattn [0:511];
    reg [15:0]       abeta [0:0];
    reg [7:0]        da [0:INVC-1];
    reg [7:0]        db [0:INVC-1];
    reg signed [7:0] cap [0:511];

    // ---- capture ATTN dump on serial_a (4 chunks/byte), running across invocations ----
    reg [5:0] sh;  reg [1:0] bc;  integer cidx;
    always @(posedge clk) begin
        if (!rst_n) begin sh<=0; bc<=0; cidx<=0; end
        else if (output_valid) begin
            sh <= {sh[3:0], serial_a};
            if (bc == 2'd3) begin
                cap[cidx] <= {sh[5:0], serial_a};
                cidx <= cidx + 1; bc <= 0;
            end else bc <= bc + 1;
        end
    end

    integer p, d, ki, m;
    // Drive one invocation: queries q0 (data_a / tok_buf[0:31]) and q0+1 (data_b / tok_buf[32:63]).
    task run_pair(input integer q0);
        begin
            for (m = 0; m < INVC; m = m + 1) begin da[m]=0; db[m]=0; end
            da[LEAD+0] = abeta[0][15:8];
            da[LEAD+1] = abeta[0][7:0];
            for (d = 0; d < DIM; d = d + 1) begin              // A_QLOAD
                da[LEAD+2+d] = Q[q0*DIM + d];
                db[LEAD+2+d] = Q[(q0+1)*DIM + d];
            end
            for (p = 0; p < 2*SEQ; p = p + 1) begin             // A_RUN: 32 pairs
                ki = p % SEQ;                                    // key (shared by both q)
                for (d = 0; d < DIM; d = d + 1) begin
                    da[KBASE + p*DIM + d] = K[ki*DIM + d];
                    db[VBASE + p*DIM + d] = V[ki*DIM + d];
                end
            end
            @(negedge clk); start=1; data_a=da[0]; data_b=db[0];
            for (m = 1; m < INVC; m = m + 1) begin
                @(negedge clk); start=0; data_a=da[m]; data_b=db[m];
            end
        end
    endtask

    integer inv, errs;
    initial begin
        $readmemh("pattern_L0/q_out_8b.hex",        Q);
        $readmemh("pattern_L0/k_out_8b.hex",        K);
        $readmemh("pattern_L0/v_out_8b.hex",        V);
        $readmemh("pattern_L0/golden_attn.hex",     gattn);
        $readmemh("pattern_L0/attn_beta_scale.hex", abeta);

        rst_n=0; mode=MODE_IDLE; start=0; data_a=0; data_b=0; weight_a=0; weight_b=0;
        #24; @(negedge clk); rst_n=1; repeat(4) @(negedge clk);
        @(negedge clk); mode=MODE_ATTN;             // mode held ATTN across all invocations

        for (inv = 0; inv < NINV; inv = inv + 1)
            run_pair(2*inv);

        mode=MODE_IDLE;
        repeat(300) @(negedge clk);

        errs = 0;
        for (m = 0; m < 512; m = m + 1)
            if (cap[m] !== gattn[m]) begin
                if (errs < 10)
                    $display("  [attn] idx=%0d HW=%0d Gold=%0d",
                        m, $signed(cap[m]), $signed(gattn[m]));
                errs = errs + 1;
            end
        $display("ATTN: errors=%0d  (captured=%0d)", errs, cidx);
        if (errs == 0) $display(" [PHASE-2 ATTN PASS]");
        else           $display(" [PHASE-2 ATTN FAIL] errors=%0d", errs);
        $finish;
    end
endmodule
