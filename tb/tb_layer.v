`timescale 1ns / 1ps
// ============================================================================
// tb_layer — full single-layer integration TB for Transformer_Layer_Top.
// Chains all chip modes through one transformer block and checks against the
// per-stage goldens, ending at layer_gold:
//
//   PROJ Q,K,V (input x_in)            -> capture Q,K,V        (vs {q,k,v}_out)
//   ATTN (buffer-Q, 8x2 queries)      -> capture attn         (vs golden_attn)
//   PROJ O (input attn)               -> capture o            (vs golden_o)
//   res1 = clip(x_in + o)   [off-chip] ->                     (vs golden_res1)
//   FFN  (input res1)                 -> capture ffn2         (vs golden_ffn2)
//   layer = clip(res1 + ffn2) [off-chip] ->                   (vs layer_gold)
//
// Reuses the proven per-mode timing: PROJ beta @start+2/3; ATTN buffer-Q LEAD=1,
// KBASE=36, VBASE=71 (V lags K by 35); FFN F2 weights driven at a fixed
// F2_KICK_DELAY=171 from the fork (deterministic f2_kick) — pad-only, gate-safe.
// ============================================================================
module tb_layer;
    localparam DIM = 32, SEQ = 16;
    localparam [1:0] MODE_IDLE=2'd0, MODE_PROJ=2'd1, MODE_ATTN=2'd2, MODE_FFN=2'd3;
    localparam integer LEAD=1, KBASE=36, VBASE=71, INVC=1450;
    localparam integer F2_KICK_DELAY=171;

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

    // ---- params + goldens ----
    reg signed [7:0] x_in [0:511];
    reg [0:0] qw[0:1023], kw[0:1023], vw[0:1023], ow[0:1023], f1w[0:1023], f2w[0:1023];
    reg [15:0] qb[0:0], kb[0:0], vb[0:0], ob[0:0], f1b[0:0], f2b[0:0], abeta[0:0];
    reg signed [7:0] g_attn[0:511], g_o[0:511], g_res1[0:511], g_ffn2[0:511], g_layer[0:511];

    // ---- captured intermediates ----
    reg signed [7:0] Qc[0:511], Kc[0:511], Vc[0:511], Ac[0:511], Oc[0:511], res1[0:511], Fc[0:511];

    // ---- generic active input/matrix for PROJ ----
    reg signed [7:0] x_act[0:511];
    reg [0:0] w_act[0:1023];
    reg [15:0] beta_act;

    // ---- capture (cap_split: PROJ splits a/b; ATTN/FFN use a only) ----
    reg signed [7:0] cap[0:511];
    reg cap_split, cap_reset;
    reg [5:0] sha, shb;  reg [1:0] bca, bcb;  integer aidx, bidx;
    always @(posedge clk) begin
        if (!rst_n || cap_reset) begin sha<=0;shb<=0;bca<=0;bcb<=0;aidx<=0;bidx<=0; end
        else if (output_valid) begin
            sha <= {sha[3:0], serial_a};
            if (bca==2'd3) begin cap[aidx]<={sha[5:0],serial_a}; aidx<=aidx+1; bca<=0; end
            else bca <= bca+1;
            if (cap_split) begin
                shb <= {shb[3:0], serial_b};
                if (bcb==2'd3) begin cap[256+bidx]<={shb[5:0],serial_b}; bidx<=bidx+1; bcb<=0; end
                else bcb <= bcb+1;
            end
        end
    end

    // ---- PROJ drive (both BLs, same matrix, 8+8 split); leaves mode=PROJ ----
    task drive_proj;
        integer t,i,c,k,j;
        begin
            @(negedge clk); mode=MODE_PROJ; start=1;
            @(negedge clk); start=0;
            @(negedge clk); data_a=beta_act[15:8]; data_b=beta_act[15:8];
            @(negedge clk); data_a=beta_act[7:0];  data_b=beta_act[7:0];
            for (t=0;t<8;t=t+1) begin
                for (i=0;i<DIM;i=i+1) begin
                    @(negedge clk);
                    data_a=x_act[t*DIM+i]; data_b=x_act[(t+8)*DIM+i];
                    weight_a=w_act[i*DIM]; weight_b=w_act[i*DIM];
                end
                for (c=1;c<DIM;c=c+1) for(k=0;k<4;k=k+1) begin
                    @(negedge clk);
                    for(j=0;j<8;j=j+1) begin
                        data_a[j]=w_act[(31-k*8-j)*DIM+c];
                        data_b[j]=w_act[(31-k*8-j)*DIM+c];
                    end
                end
                @(negedge clk); data_a=0; data_b=0; repeat(3) @(negedge clk);
            end
        end
    endtask

    // ---- ATTN drive (buffer-Q): 8 invocations x 2 queries; K full-byte on
    //      data_a, V full-byte on data_b (lags K by VBASE-KBASE). mode held ATTN. ----
    reg [7:0] da[0:INVC-1], db[0:INVC-1];
    task drive_attn;
        integer inv,p,d,ki,m,q0;
        begin
            @(negedge clk); mode=MODE_ATTN;
            for (inv=0;inv<8;inv=inv+1) begin
                q0 = 2*inv;
                for (m=0;m<INVC;m=m+1) begin da[m]=0; db[m]=0; end
                da[LEAD+0]=abeta[0][15:8]; da[LEAD+1]=abeta[0][7:0];
                for (d=0;d<DIM;d=d+1) begin
                    da[LEAD+2+d]=Qc[q0*DIM+d];
                    db[LEAD+2+d]=Qc[(q0+1)*DIM+d];
                end
                for (p=0;p<2*SEQ;p=p+1) begin
                    ki=p%SEQ;
                    for (d=0;d<DIM;d=d+1) begin
                        da[KBASE+p*DIM+d]=Kc[ki*DIM+d];
                        db[VBASE+p*DIM+d]=Vc[ki*DIM+d];
                    end
                end
                @(negedge clk); start=1; data_a=da[0]; data_b=db[0];
                for (m=1;m<INVC;m=m+1) begin @(negedge clk); start=0; data_a=da[m]; data_b=db[m]; end
            end
        end
    endtask

    // ---- FFN drive: F1 (res1+Wf1 on data_a) + F2 (Wf2 on data_b) via f2_kick ----
    task drive_f1;
        integer t,i,c,k,j;
        begin
            @(negedge clk); mode=MODE_FFN; start=1;
            @(negedge clk); start=0;
            @(negedge clk); data_a=f1b[0][15:8];
            @(negedge clk); data_a=f1b[0][7:0];
            for (t=0;t<16;t=t+1) begin
                for (i=0;i<DIM;i=i+1) begin
                    @(negedge clk); data_a=res1[t*DIM+i]; weight_a=f1w[i*DIM];
                end
                for (c=1;c<DIM;c=c+1) for(k=0;k<4;k=k+1) begin
                    @(negedge clk);
                    for(j=0;j<8;j=j+1) data_a[j]=f1w[(31-k*8-j)*DIM+c];
                end
                @(negedge clk); data_a=0; repeat(3) @(negedge clk);
            end
        end
    endtask
    task drive_f2;
        integer t,i,c,k,j;
        begin
            @(negedge clk); data_b=f2b[0][15:8];
            @(negedge clk); data_b=f2b[0][7:0];
            for (t=0;t<16;t=t+1) begin
                for (i=0;i<DIM;i=i+1) begin @(negedge clk); weight_b=f2w[i*DIM]; data_b=0; end
                for (c=1;c<DIM;c=c+1) for(k=0;k<4;k=k+1) begin
                    @(negedge clk);
                    for(j=0;j<8;j=j+1) data_b[j]=f2w[(31-k*8-j)*DIM+c];
                end
                @(negedge clk); data_b=0; repeat(3) @(negedge clk);
            end
        end
    endtask

    // ---- helpers ----
    integer terr;
    task report(input [16*8:1] name, input integer errs);
        begin
            $display("  %0s: errors=%0d", name, errs);
            terr = terr + errs;
        end
    endtask

    integer m, e;
    reg signed [8:0] s9;
    initial begin
        $readmemh("pattern_L0/x_in.hex", x_in);
        $readmemh("pattern_L0/q_w.hex", qw); $readmemh("pattern_L0/k_w.hex", kw);
        $readmemh("pattern_L0/v_w.hex", vw); $readmemh("pattern_L0/o_w.hex", ow);
        $readmemh("pattern_L0/f1_w.hex", f1w); $readmemh("pattern_L0/f2_w.hex", f2w);
        $readmemh("pattern_L0/q_beta_16b.hex", qb); $readmemh("pattern_L0/k_beta_16b.hex", kb);
        $readmemh("pattern_L0/v_beta_16b.hex", vb); $readmemh("pattern_L0/o_beta_16b.hex", ob);
        $readmemh("pattern_L0/f1_beta_16b.hex", f1b); $readmemh("pattern_L0/f2_beta_16b.hex", f2b);
        $readmemh("pattern_L0/attn_beta_scale.hex", abeta);
        $readmemh("pattern_L0/golden_attn.hex", g_attn); $readmemh("pattern_L0/golden_o.hex", g_o);
        $readmemh("pattern_L0/golden_res1.hex", g_res1);
        $readmemh("pattern_L0/golden_ffn2.hex", g_ffn2); $readmemh("pattern_L0/layer_gold.hex", g_layer);

        rst_n=0; mode=MODE_IDLE; start=0; data_a=0; data_b=0; weight_a=0; weight_b=0;
        cap_split=0; cap_reset=0; terr=0;
        #24; @(negedge clk); rst_n=1; repeat(4) @(negedge clk);

        // ---- PROJ Q ----
        for (m=0;m<512;m=m+1) x_act[m]=x_in[m];
        for (m=0;m<1024;m=m+1) w_act[m]=qw[m];  beta_act=qb[0]; cap_split=1;
        @(negedge clk); cap_reset=1; @(negedge clk); cap_reset=0;
        drive_proj; repeat(24) @(negedge clk);
        for (m=0;m<512;m=m+1) Qc[m]=cap[m];

        // ---- PROJ K ----
        for (m=0;m<1024;m=m+1) w_act[m]=kw[m];  beta_act=kb[0];
        @(negedge clk); cap_reset=1; @(negedge clk); cap_reset=0;
        drive_proj; repeat(24) @(negedge clk);
        for (m=0;m<512;m=m+1) Kc[m]=cap[m];

        // ---- PROJ V ----
        for (m=0;m<1024;m=m+1) w_act[m]=vw[m];  beta_act=vb[0];
        @(negedge clk); cap_reset=1; @(negedge clk); cap_reset=0;
        drive_proj; repeat(24) @(negedge clk);
        for (m=0;m<512;m=m+1) Vc[m]=cap[m];

        // ---- ATTN ----
        cap_split=0;
        @(negedge clk); cap_reset=1; @(negedge clk); cap_reset=0;
        drive_attn; repeat(300) @(negedge clk);
        for (m=0;m<512;m=m+1) Ac[m]=cap[m];
        e=0; for(m=0;m<512;m=m+1) if (Ac[m]!==g_attn[m]) e=e+1;  report("attn", e);

        // ---- PROJ O (projects attn) ----
        for (m=0;m<512;m=m+1) x_act[m]=Ac[m];
        for (m=0;m<1024;m=m+1) w_act[m]=ow[m];  beta_act=ob[0]; cap_split=1;
        @(negedge clk); cap_reset=1; @(negedge clk); cap_reset=0;
        drive_proj; repeat(24) @(negedge clk);
        for (m=0;m<512;m=m+1) Oc[m]=cap[m];
        e=0; for(m=0;m<512;m=m+1) if (Oc[m]!==g_o[m]) e=e+1;  report("o", e);

        // ---- residual-1 (off-chip): res1 = clip(x_in + o) ----
        for (m=0;m<512;m=m+1) begin
            s9 = $signed(x_in[m]) + $signed(Oc[m]);
            res1[m] = (s9>9'sd127)?8'sd127:(s9<-9'sd127)?-8'sd127:s9[7:0];
        end
        e=0; for(m=0;m<512;m=m+1) if (res1[m]!==g_res1[m]) e=e+1;  report("res1", e);

        // ---- FFN ----
        cap_split=0;
        @(negedge clk); cap_reset=1; @(negedge clk); cap_reset=0;
        fork
            drive_f1;
            begin : f2p repeat(F2_KICK_DELAY) @(negedge clk); drive_f2; end  // gate-safe fixed delay
        join
        repeat(60) @(negedge clk);
        for (m=0;m<512;m=m+1) Fc[m]=cap[m];
        e=0; for(m=0;m<512;m=m+1) if (Fc[m]!==g_ffn2[m]) e=e+1;  report("ffn2", e);

        // ---- residual-2 (off-chip): layer = clip(res1 + ffn2) ----
        e=0;
        for (m=0;m<512;m=m+1) begin
            s9 = $signed(res1[m]) + $signed(Fc[m]);
            if (((s9>9'sd127)?8'sd127:(s9<-9'sd127)?-8'sd127:s9[7:0]) !== g_layer[m]) e=e+1;
        end
        report("layer", e);

        $display("============================================");
        if (terr==0) $display(" [LAYER E2E PASS]");
        else         $display(" [LAYER E2E FAIL] total errors=%0d", terr);
        $display("============================================");
        $finish;
    end
endmodule
