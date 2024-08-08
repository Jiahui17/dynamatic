library ieee;
use ieee.std_logic_1164.all;
use work.types.all;

entity lazy_fork is
  generic (
    SIZE     : integer;
    DATA_WIDTH : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channels
    outs       : out data_array (SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0);
    outs_valid : out std_logic_vector(SIZE - 1 downto 0);
    outs_ready : in  std_logic_vector(SIZE - 1 downto 0)
  );
end entity;

architecture arch of lazy_fork is
begin
  control : entity work.lazy_fork_dataless generic map (SIZE)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      ins_ready  => ins_ready,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );

  process (ins)
  begin
    for i in 0 to SIZE - 1 loop
      outs(i) <= ins;
    end loop;
  end process;
end architecture;
