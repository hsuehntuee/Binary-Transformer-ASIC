`timescale 1ns / 1ps
// ============================================================================
// tb_proj — phase-1 TB for Transformer_Layer_Top PROJ mode (28-pad redesign).
//
// Drives Q, K, V as three token-split PROJ runs: both BLs load the same matrix,
// BL0 gets input tokens 0-7 on data_a, BL1 gets tokens 8-15 on data_b; betas on
// the data pads during the 2-cyc BETA state. Captures BL0 -> cap[0..255]
// (output tokens 0-7) and BL1 -> cap[256..511] (tokens 8-15), then checks each
// against pattern_L0/{q,k,v}_out_8b.hex.
//
// Run from tb/ (so $readmemh finds pattern_L0/).
// ============================================================================
module tb_proj;
    localparam DIM = 32, SEQ = 16;
    localparam [1:0] MODE_IDLE = 2'd0, MODE_PROJ = 2'd1,
                     MODE_ATTN = 2'd2, MODE_FFN  = 2'd3;

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

    // ---- off-chip memories (model params + goldens) ----
    reg signed [7:0] x_mem [0:511];                       // Q/K/V input
    reg signed [7:0] attn_in [0:511];                     // O input = golden_attn
    reg [0:0]        qw [0:1023], kw [0:1023], vw [0:1023], ow [0:1023];
    reg [15:0]       qb [0:0], kb [0:0], vb [0:0], ob [0:0];
    reg signed [7:0] qg [0:511], kg [0:511], vg [0:511], og [0:511];

    // ---- active input/matrix selected per run ----
    reg signed [7:0] x_act [0:511];
    reg [0:0]        w_act [0:1023];
    reg [15:0]       beta_act;
    reg signed [7:0] gold_act [0:511];

    reg signed [7:0] cap [0:511];

    // ---- serial capture: BL0 -> cap[0..255], BL1 -> cap[256..511] ----
    reg [5:0] sha, shb;
    reg [1:0] bc;
    integer   aidx, bidx;
    reg       cap_reset;
    always @(posedge clk) begin
        if (!rst_n || cap_reset) begin
            sha <= 0; shb <= 0; bc <= 0; aidx <= 0; bidx <= 0;
        end else if (output_valid) begin
            sha <= {sha[3:0], serial_a};
            shb <= {shb[3:0], serial_b};
            if (bc == 2'd3) begin
                cap[aidx]     <= {sha[5:0], serial_a};
                cap[256+bidx] <= {shb[5:0], serial_b};
                aidx <= aidx + 1;
                bidx <= bidx + 1;
                bc   <= 0;
            end else bc <= bc + 1;
        end
    end

    // ---- select active input/matrix (0=Q 1=K 2=V 3=O) ----
    task load_active(input integer sel);
        integer m;
        begin
            for (m = 0; m < 1024; m = m + 1)
                w_act[m] = (sel==0) ? qw[m] : (sel==1) ? kw[m] : (sel==2) ? vw[m] : ow[m];
            beta_act = (sel==0) ? qb[0] : (sel==1) ? kb[0] : (sel==2) ? vb[0] : ob[0];
            for (m = 0; m < 512; m = m + 1) begin
                gold_act[m] = (sel==0) ? qg[m] : (sel==1) ? kg[m] : (sel==2) ? vg[m] : og[m];
                x_act[m]    = (sel==3) ? attn_in[m] : x_mem[m];   // O projects attn_out
            end
        end
    endtask

    // ---- drive one PROJ run (both BLs, same matrix, 8+8 token split) ----
    task run_proj;
        integer t, i, c, k, j;
        begin
            @(negedge clk); mode = MODE_PROJ; start = 1;
            @(negedge clk); start = 0;
            @(negedge clk); data_a = beta_act[15:8]; data_b = beta_act[15:8];
            @(negedge clk); data_a = beta_act[7:0];  data_b = beta_act[7:0];
            for (t = 0; t < 8; t = t + 1) begin
                // ACT_W0: 32 activation bytes + column-0 weight config bit
                for (i = 0; i < DIM; i = i + 1) begin
                    @(negedge clk);
                    data_a   = x_act[t*DIM + i];
                    data_b   = x_act[(t+8)*DIM + i];
                    weight_a = w_act[i*DIM];
                    weight_b = w_act[i*DIM];
                end
                // COMP: next-col weight bytes for columns 1..31 (4 chunks each)
                for (c = 1; c < DIM; c = c + 1)
                    for (k = 0; k < 4; k = k + 1) begin
                        @(negedge clk);
                        for (j = 0; j < 8; j = j + 1) begin
                            data_a[j] = w_act[(31 - k*8 - j)*DIM + c];
                            data_b[j] = w_act[(31 - k*8 - j)*DIM + c];
                        end
                    end
                // column-31 COMP (no next-col load) + boundary
                @(negedge clk); data_a = 0; data_b = 0;
                repeat (3) @(negedge clk);
            end
            mode = MODE_IDLE;
        end
    endtask

    integer total_errs;
    task run_and_check(input integer sel);
        integer m, errs;
        begin
            load_active(sel);
            @(negedge clk); cap_reset = 1; @(negedge clk); cap_reset = 0;
            run_proj;
            repeat (24) @(negedge clk);   // drain dequant pipeline + last outputs
            errs = 0;
            for (m = 0; m < 512; m = m + 1)
                if (cap[m] !== gold_act[m]) begin
                    if (errs < 8)
                        $display("  [%0s] idx=%0d HW=%0d Gold=%0d",
                            (sel==0)?"Q":(sel==1)?"K":(sel==2)?"V":"O",
                            m, $signed(cap[m]), $signed(gold_act[m]));
                    errs = errs + 1;
                end
            $display("PROJ %0s: errors=%0d  (captured a=%0d b=%0d)",
                     (sel==0)?"Q":(sel==1)?"K":(sel==2)?"V":"O", errs, aidx, bidx);
            total_errs = total_errs + errs;
        end
    endtask

    initial begin
        $readmemh("pattern_L0/x_in.hex",      x_mem);
        $readmemh("pattern_L0/q_w.hex",       qw);
        $readmemh("pattern_L0/k_w.hex",       kw);
        $readmemh("pattern_L0/v_w.hex",       vw);
        $readmemh("pattern_L0/q_beta_16b.hex", qb);
        $readmemh("pattern_L0/k_beta_16b.hex", kb);
        $readmemh("pattern_L0/v_beta_16b.hex", vb);
        $readmemh("pattern_L0/q_out_8b.hex",  qg);
        $readmemh("pattern_L0/k_out_8b.hex",  kg);
        $readmemh("pattern_L0/v_out_8b.hex",  vg);
        $readmemh("pattern_L0/o_w.hex",       ow);
        $readmemh("pattern_L0/o_beta_16b.hex", ob);
        $readmemh("pattern_L0/golden_o.hex",  og);
        $readmemh("pattern_L0/golden_attn.hex", attn_in);  // O projects attn_out

        rst_n = 0; mode = MODE_IDLE; start = 0;
        data_a = 0; data_b = 0; weight_a = 0; weight_b = 0;
        cap_reset = 0; total_errs = 0;
        #24; @(negedge clk); rst_n = 1; repeat (4) @(negedge clk);

        run_and_check(0);   // Q
        run_and_check(1);   // K
        run_and_check(2);   // V
        run_and_check(3);   // O (projects golden_attn -> golden_o)

        $display("============================================");
        if (total_errs == 0) $display(" [PHASE-1 PROJ PASS]");
        else                 $display(" [PHASE-1 PROJ FAIL] total errors=%0d", total_errs);
        $display("============================================");
        $finish;
    end
endmodule
