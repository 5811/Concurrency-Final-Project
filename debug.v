include hashmodule.v;


// Instantiate compute grid with hard-wired inputs 
wire unsigned [0:31] nonce [0:7];
wire unsigned [0:31] hash [0:7];

assign nonce[0]=32'61000000;
genvar zeroVar;
for(zeroVar=2; zeroVar<8; zeroVar=zeroVar+1) begin: zeros
    nonce[zeroVar]=0;
end

hashModule m(
  nonce,
  hash
);


$display("%h", hash[0]);


/*
always @(posedge clock.val) begin
  $display("%d %d %d %d %d", g.gridScores[0][0], g.gridScores[0][1], g.gridScores[0][2], g.gridScores[0][3], g.gridScores[0][4]);
  $display("%d %d %d %d %d", g.gridScores[1][0], g.gridScores[1][1], g.gridScores[1][2], g.gridScores[1][3], g.gridScores[1][4]);
  $display("%d %d %d %d %d", g.gridScores[2][0], g.gridScores[2][1], g.gridScores[2][2], g.gridScores[2][3], g.gridScores[2][4]);
  $display("%d %d %d %d %d", g.gridScores[3][0], g.gridScores[3][1], g.gridScores[3][2], g.gridScores[3][3], g.gridScores[3][4]);
  $display("%d %d %d %d %d", g.gridScores[4][0], g.gridScores[4][1], g.gridScores[4][2], g.gridScores[4][3], g.gridScores[4][4]);
  $display("");
  
  count <= (count + 1);
  if (done | (&count)) begin
    $display("%d", g.score);
    $finish(1);
  end
end
*/