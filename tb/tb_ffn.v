`timescale 1ns / 1ps
// ============================================================================
// tb_ffn — phase-3 TB for Transformer_Layer_Top FFN mode (token-pipeline:
// F1 on BL0, F2 on BL1, on-chip leaky + ping-pong buffer between them).
//
// FFN input is golden_res1 (the off-chip residual-1). F1 stream (res1 + Wf1)
// drives data_a/weight_a from FFN start; F2 stream (beta_f2 + Wf2 config/bytes)
// drives weight_b/data_b, triggered off the chip's internal f2_kick (F2's
// activations come from the on-chip leaky buffer, not the pad). The captured F2
// output (serial_a) is checked vs golden_ffn2; layer_out = clip(res1+ffn2) vs
// layer_gold. Locked offset: F2LEAD=1 (negedges after f2_kick before beta_f2).
// ============================================================================
module tb_ffn;
    localparam DIM = 32, SEQ = 16;
    localparam [1:0] MODE_IDLE=2'd0, MODE_PROJ=2'd1, MODE_ATTN=2'd2, MODE_FFN=2'd3;
    // Gate-sim safe: F2's controller starts internally at the deterministic
    // f2_kick (f1_cnt==32). Instead of watching that internal signal, the TB
    // waits a fixed number of negedges from the fork (measured 171) before
    // driving the F2 weight stream — no hierarchical dut.* reference.
    localparam integer F2_KICK_DELAY = 171;

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

    reg signed [7:0] res1 [0:511], g_ffn2 [0:511], g_layer [0:511];
    reg [0:0]        f1w [0:1023], f2w [0:1023];
    reg [15:0]       bf1 [0:0], bf2 [0:0];
    reg signed [7:0] cap [0:511];

    // ---- capture F2 output on serial_a ----
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

    // (F1 path, leaky-buffer reads, and beta_f2/weight alignment were verified
    //  during bring-up; F2LEAD=1 is the locked offset from the chip's f2_kick.
    //  Final checks below: captured F2 vs golden_ffn2, and res1+ffn2 vs layer_gold.)

    integer t, i, c, k, j;
    task drive_f1;
        begin
            @(negedge clk); mode=MODE_FFN; start=1;
            @(negedge clk); start=0;
            @(negedge clk); data_a = bf1[0][15:8];
            @(negedge clk); data_a = bf1[0][7:0];
            for (t = 0; t < 16; t = t + 1) begin
                for (i = 0; i < DIM; i = i + 1) begin
                    @(negedge clk);
                    data_a   = res1[t*DIM + i];
                    weight_a = f1w[i*DIM];
                end
                for (c = 1; c < DIM; c = c + 1)
                    for (k = 0; k < 4; k = k + 1) begin
                        @(negedge clk);
                        for (j = 0; j < 8; j = j + 1)
                            data_a[j] = f1w[(31 - k*8 - j)*DIM + c];
                    end
                @(negedge clk); data_a = 0;
                repeat (3) @(negedge clk);
            end
        end
    endtask

    integer t2, i2, c2, k2, j2;
    task drive_f2;
        begin
            @(negedge clk); data_b = bf2[0][15:8];
            @(negedge clk); data_b = bf2[0][7:0];
            for (t2 = 0; t2 < 16; t2 = t2 + 1) begin
                for (i2 = 0; i2 < DIM; i2 = i2 + 1) begin
                    @(negedge clk);
                    weight_b = f2w[i2*DIM];
                    data_b   = 0;                 // F2 activation comes from buf
                end
                for (c2 = 1; c2 < DIM; c2 = c2 + 1)
                    for (k2 = 0; k2 < 4; k2 = k2 + 1) begin
                        @(negedge clk);
                        for (j2 = 0; j2 < 8; j2 = j2 + 1)
                            data_b[j2] = f2w[(31 - k2*8 - j2)*DIM + c2];
                    end
                @(negedge clk); data_b = 0;
                repeat (3) @(negedge clk);
            end
        end
    endtask

    integer m, errs, lerrs;
    reg signed [8:0] lsum;  reg signed [7:0] lout;
    initial begin
        $readmemh("pattern_L0/golden_res1.hex", res1);
        $readmemh("pattern_L0/golden_ffn2.hex", g_ffn2);
        $readmemh("pattern_L0/layer_gold.hex",  g_layer);
        $readmemh("pattern_L0/f1_w.hex",        f1w);
        $readmemh("pattern_L0/f2_w.hex",        f2w);
        $readmemh("pattern_L0/f1_beta_16b.hex", bf1);
        $readmemh("pattern_L0/f2_beta_16b.hex", bf2);

        rst_n=0; mode=MODE_IDLE; start=0; data_a=0; data_b=0; weight_a=0; weight_b=0;
        #24; @(negedge clk); rst_n=1; repeat(4) @(negedge clk);

        fork
            drive_f1;
            begin : f2proc
                repeat (F2_KICK_DELAY) @(negedge clk);
                drive_f2;
            end
        join

        repeat (60) @(negedge clk);   // drain last F2 token (mode stays FFN)
        mode = MODE_IDLE;

        errs = 0;
        for (m = 0; m < 512; m = m + 1)
            if (cap[m] !== g_ffn2[m]) begin
                if (errs < 10)
                    $display("  [ffn2] idx=%0d HW=%0d Gold=%0d",
                        m, $signed(cap[m]), $signed(g_ffn2[m]));
                errs = errs + 1;
            end
        // layer_out = clip(res1 + ffn2, [-127,127]) computed from captured F2
        lerrs = 0;
        for (m = 0; m < 512; m = m + 1) begin
            lsum = $signed(res1[m]) + $signed(cap[m]);
            lout = (lsum > 9'sd127) ? 8'sd127 : (lsum < -9'sd127) ? -8'sd127 : lsum[7:0];
            if (lout !== g_layer[m]) lerrs = lerrs + 1;
        end
        $display("FFN: ffn2_errors=%0d layer_errors=%0d (captured=%0d)", errs, lerrs, cidx);
        if (errs == 0 && lerrs == 0) $display(" [PHASE-3 FFN PASS]");
        else                         $display(" [PHASE-3 FFN FAIL]");
        $finish;
    end
endmodule
