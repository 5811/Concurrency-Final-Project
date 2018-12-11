include hashModule.v;


wire [31:0] nonce [7:0];
wire [31:0] hash [7:0];

assign nonce[0]=32'h61000000;
genvar zeroVar;
for(zeroVar=2; zeroVar<8; zeroVar=zeroVar+1) begin: zeros
    assign nonce[zeroVar]=0;
end

wire [255:0] flattenedNonce;
wire [255:0] flattenedHash;

genvar index1;
for (index1=0; index1 < 8; index1=index1+1) begin: flatten
    assign flattenedNonce[255-32*index1:224-32*index1] = nonce[index1];
end
hashModule m(
  flattenedNonce,
  flattenedHash
);

genvar index2;
for (index2=0; index2 < 8; index2=index2+1) begin: unflatten
    assign hash[index2] = flattenedHash[255-32*index2:224-32*index2];
end

always @(posedge clock.val) begin
  $display("%h", hash[0]);
  $finish(1);
end


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