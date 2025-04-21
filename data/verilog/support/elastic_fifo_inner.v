`timescale 1ns/1ps
module elastic_fifo_inner #(
  parameter NUM_SLOTS = 2,
  parameter DATA_TYPE = 32
) (
  input  clk,
  input  rst,
  input  [DATA_TYPE - 1 : 0] ins,
  input  ins_valid,
  input  outs_ready,

  output [DATA_TYPE - 1 : 0] outs,
  output outs_valid,
  output ins_ready
);

  // Internal Signal Definition
  wire ReadEn, WriteEn;
  reg [$clog2(NUM_SLOTS) - 1 : 0] Tail = 0, Head = 0;
  reg Full = 0, Empty = 1;
  reg [DATA_TYPE - 1 : 0] Memory[0 : NUM_SLOTS - 1];
  integer i;
  
  // Ready if there is space in the FIFO
  assign ins_ready = !Full || outs_ready;

  // Read if next can accept and there is sth in FIFO to read
  assign ReadEn = (outs_ready && !Empty);
  assign outs_valid = ~Empty;
  assign WriteEn = ins_valid & (!Full || outs_ready);
  assign outs = Memory[Head];

  // Initialize memory content
  initial begin
     for (i=0; i<NUM_SLOTS; i=i+1) begin
        Memory[i] = 0;
     end
  end

  always @(posedge clk) begin
    if (rst) begin
     for (i=0; i<NUM_SLOTS; i=i+1) begin
        Memory[i] <= 0;
     end
    end else if (WriteEn) begin
      Memory[Tail] <= ins;
    end
  end

  // Update Tail
  always @(posedge clk) begin
    if (rst) begin
      Tail <= 0;
    end else begin
      if (WriteEn) begin
        // This simulates the behavior of "Tail <= (Tail + 1) % NUM_SLOTS;".
        // The reminder operator might be handled differently (and also
        // incorrectly) in different synthesis tools.

        // Here and below: Appending Tail to 32 bits ("{{31{1'b0}}, Tail} ==
        // NUM_SLOTS - 1") to fix Verilator's linter warning: "Tail" is a 1-bit
        // signal, but "NUM_SLOTS - 1" is a 32-bit number.
        if ({{31{1'b0}}, Tail} == NUM_SLOTS - 1) begin
          Tail <= 0;
        end else begin
          Tail <= Tail + 1;
        end
      end
    end  
  end

  // Update Head
  always @(posedge clk) begin
    if (rst) begin
      Head <= 0;
    end else begin
      if (ReadEn) begin
        // This simulates the behavior of "Head <= (Head + 1) % NUM_SLOTS;".
        // The reminder operator might be handled differently (and also
        // incorrectly) in different synthesis tools.
        if ({{31{1'b0}}, Head} == NUM_SLOTS - 1) begin
          Head <= 0;
        end else begin
          Head <= Head + 1;
        end
      end
    end 
  end

  // Update Full
  always @(posedge clk) begin
    if (rst) begin
      Full <= 0;
    end else begin
      // If only filling but not emptying
      if (WriteEn && !ReadEn) begin
        // If the new tail index will reach head index, the FIFO is full.
        if ((({{31{1'b0}}, Tail} == NUM_SLOTS - 1) && (Head == 0)) || ((Tail + 1) == Head)) begin
            Full <= 1;
        end
      end else if (!WriteEn && ReadEn) begin
        // if only emptying but not filling
        Full <= 0;
      end
    end
  end

  // Update Empty
  always @(posedge clk) begin
    if (rst) begin
      Empty <= 1;
    end else begin
      // If only emptying but not filling
      if (!WriteEn && ReadEn) begin
        // If the new head index will reach tail index, the FIFO is empty.
        if (({{31{1'b0}}, Head} == (NUM_SLOTS - 1) && (Tail == 0)) || ((Head + 1) == Tail)) begin
          Empty <= 1;
        end
      end else if (WriteEn && !ReadEn) begin
        // If only filling but not emptying
        Empty <= 0;
      end
    end
  end

endmodule
