// ============================================================================
// PROJECT: FPGA NOVELTY DETECTION ACCELERATOR (Consolidated)
// CHIP: GOWIN TANG NANO 9K
// LOGIC: LEAKY INTEGRATOR NEURON + UART E2E BENCHMARK ECHO
// ============================================================================

module top_novelty (
    input clk,            // Pin 52 (27MHz) [cite: 1]
    input reset_n,        // Pin 4 (Active Low) [cite: 1]
    input rx_pin,         // Pin 18 (UART RX) [cite: 1]
    output tx_pin,        // Pin 17 (UART TX Echo) [cite: 1]
    output [5:0] leds     // Onboard LEDs (Novelty Visualization) [cite: 2]
);

    wire [7:0] rx_data;
    wire rx_tick;
    wire tx_busy;
    reg tx_start; [cite: 3]
    
    wire [7:0] weight;
    reg [3:0] addr = 4'd0; [cite: 3]
    reg [15:0] energy_acc = 16'd0; [cite: 3]

    // --- 1. UART RECEIVER (INPUT GATE) --- [cite: 4]
    uart_rx #(.CLK_FREQ(27000000), .BAUD(115200)) rx_inst (
        .clk(clk), .rx(rx_pin), .data(rx_data), .tick(rx_tick)
    );

    // --- 2. UART TRANSMITTER (BENCHMARK ECHO) --- [cite: 5]
    uart_tx #(.CLK_FREQ(27000000), .BAUD(115200)) tx_inst (
        .clk(clk), .start(tx_start), .data(rx_data), .tx(tx_pin), .busy(tx_busy)
    );

    // --- 3. WEIGHT MEMORY (QUANTIZED ROM) --- [cite: 6]
    weight_bram mem_inst (.clka(clk), .addra(addr), .douta(weight));

    // --- 4. NOVELTY THRESHOLDING (VISUAL OUTPUT) --- [cite: 7]
    // LEDs turn ON (Low) if energy density exceeds weight
    assign leds = (energy_acc[11:4] > weight) ? 6'b000000 : 6'b111111; [cite: 7, 8]

    // --- 5. MAIN INFERENCE ENGINE ---
    always @(posedge clk) begin
        tx_start <= 1'b0; [cite: 8]
        if (!reset_n) begin [cite: 9]
            addr <= 4'd0; [cite: 9]
            energy_acc <= 16'd0; [cite: 9]
        end else if (rx_tick) begin [cite: 10]
            addr <= addr + 4'd1; [cite: 10]
            // Leaky Integrator: E = (E/2) + New_Data [cite: 11]
            energy_acc <= (energy_acc >> 1) + {8'd0, rx_data}; [cite: 11]
            // Trigger Echo for Python E2E Timing [cite: 12]
            if (!tx_busy) tx_start <= 1'b1; [cite: 12]
        end
    end
endmodule

// --- UART RX MODULE --- [cite: 13]
module uart_rx #(parameter CLK_FREQ = 27000000, parameter BAUD = 115200) (
    input clk, input rx, output reg [7:0] data, output reg tick
);
    localparam WAIT_COUNT = CLK_FREQ / BAUD; [cite: 14]
    localparam HALF_COUNT = WAIT_COUNT / 2; [cite: 14]
    reg [31:0] count = 32'd0; [cite: 14]
    reg [3:0] state = 4'd0; [cite: 15]
    reg receiving = 1'b0; [cite: 15]
    always @(posedge clk) begin
        tick <= 1'b0; [cite: 15]
        if (!receiving) begin [cite: 16]
            if (rx == 1'b0) begin [cite: 16]
                if (count < HALF_COUNT) count <= count + 32'd1; [cite: 16]
                else begin count <= 32'd0; receiving <= 1'b1; state <= 4'd0; end [cite: 17, 18]
            end else count <= 32'd0; [cite: 18]
        end else begin [cite: 19]
            if (count < WAIT_COUNT - 1) count <= count + 32'd1; [cite: 19]
            else begin [cite: 20]
                count <= 32'd0; [cite: 20]
                if (state < 4'd8) begin data[state[2:0]] <= rx; state <= state + 4'd1; end [cite: 21]
                else begin receiving <= 1'b0; tick <= 1'b1; end [cite: 22, 23]
            end
        end
    end
endmodule

// --- UART TX MODULE --- [cite: 24]
module uart_tx #(parameter CLK_FREQ = 27000000, parameter BAUD = 115200) (
    input clk, input start, input [7:0] data, output reg tx, output reg busy
);
    localparam WAIT_COUNT = CLK_FREQ / BAUD; [cite: 24]
    reg [31:0] count = 32'd0; [cite: 24]
    reg [3:0] state = 4'd0; [cite: 24]
    reg [7:0] d_reg; [cite: 24]
    initial tx = 1'b1; [cite: 25]
    initial busy = 1'b0; [cite: 25]
    always @(posedge clk) begin
        if (!busy) begin [cite: 25]
            if (start) begin d_reg <= data; busy <= 1'b1; state <= 4'd0; count <= 32'd0; tx <= 1'b0; end [cite: 25, 26, 27]
        end else begin [cite: 27]
            if (count < WAIT_COUNT - 1) count <= count + 32'd1; [cite: 27]
            else begin [cite: 28]
                count <= 32'd0; [cite: 28]
                if (state < 4'd8) begin tx <= d_reg[state[2:0]]; state <= state + 4'd1; end [cite: 29]
                else if (state == 4'd8) begin tx <= 1'b1; state <= state + 4'd1; end [cite: 30, 31]
                else busy <= 1'b0; [cite: 31]
            end
        end
    end
endmodule

// --- WEIGHT ROM MODULE --- [cite: 32]
module weight_bram (input clka, input [3:0] addra, output reg [7:0] douta);
    reg [7:0] rom [0:15]; [cite: 33]
    initial begin
        rom[0]=8'h07; rom[1]=8'h15; rom[2]=8'h08; rom[3]=8'h12; [cite: 33]
        rom[4]=8'h1e; rom[5]=8'h21; rom[6]=8'h08; rom[7]=8'h22; [cite: 33, 34]
        rom[8]=8'h09; rom[9]=8'h22; rom[10]=8'h0a; rom[11]=8'h1a; [cite: 34]
        rom[12]=8'h1e; rom[13]=8'h0c; rom[14]=8'h1e; rom[15]=8'h06; [cite: 34]
    end
    always @(posedge clka) douta <= rom[addra]; [cite: 34]
endmodule
