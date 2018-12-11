module MSUnit(
    input wire [31:0] w15,
    input wire [31:0] w2,
    input wire [31:0] w16,
    input wire [31:0] w7,

    output wire [31:0] w0

);
    wire [31:0] S0;
    assign S0 = {w15[6 : 0], w15[31 : 7]} ^ {w15[17 : 0], w15[31 : 17]} ^ {w15 >> 3};
    wire [31:0] S1;
    assign S1 =  {w2[16 : 0], w2[31 : 18]} ^ {w2[18 : 0], w2[31 : 19]} ^ {w2 >> 10};
    
    assign w0=w16+S0+w7+S1;

endmodule

module hashRound(
    input wire [255:0] workingInput,
    input wire [31:0] k,
    input wire [31:0] w,

    output wire [255:0] workingOutput

);

    wire [31:0] workingVariables[7:0];
    wire [31:0] nWorkingVariables[7:0];
    genvar index1;
    for (index1=0; index1 < 8; index1=index1+1) begin: unflatten
        assign workingVariables[index1] = workingInput[255-32*index1:224-32*index1];
    end

    wire [31:0] e;
    assign e=workingVariables[4];
    wire [31:0] a;
    assign a=workingVariables[0];

    wire [31:0] S1;
    assign S1 =  {e[5 : 0], e[31 : 6]} ^ {e[10 : 0], e[31 : 11]} ^ {e[24 : 0], e[31 : 25]};
    wire [31:0] ch;
    assign ch = (workingVariables[4] & workingVariables[5]) ^ ((~ workingVariables[4]) & workingVariables[6]);
    wire [31:0] temp1;
    assign temp1 = workingVariables[7] + S1 + ch + k + w;
    wire [31:0] S0;
    assign S0 = {a[1 : 0], a[31 : 2]} ^ {a[12 : 0], a[31 : 13]} ^ {a[21 : 0], a[31 : 22]};
    wire [31:0] maj;
    assign maj = (workingVariables[0] & workingVariables[1]) ^ (workingVariables[0] & workingVariables[2]) ^ (workingVariables[1] & workingVariables[2]);
    wire [31:0] temp2;
    assign temp2 = S0 + maj;

    assign nWorkingVariables[7] = workingVariables[6];
    assign nWorkingVariables[5] = workingVariables[4];
    assign nWorkingVariables[4] = workingVariables[3]+ temp1;
    assign nWorkingVariables[6] = workingVariables[5];
    assign nWorkingVariables[3] = workingVariables[2];
    assign nWorkingVariables[2] = workingVariables[1];
    assign nWorkingVariables[1] = workingVariables[0];
    assign nWorkingVariables[0] = temp1 + temp2;

    genvar index2;
    for (index2=0; index2 < 8; index2=index2+1) begin: flatten
        assign workingOutput[255-32*index2:224-32*index2] = nWorkingVariables[index2];
    end


endmodule

//hashes a nonce that is a 32 byte value
module hashModule#(

    parameter kflat =
    {
    32'h428a2f98,32'h71374491,32'hb5c0fbcf,32'he9b5dba5,32'h3956c25b,32'h59f111f1,32'h923f82a4,32'hab1c5ed5,
    32'hd807aa98,32'h12835b01,32'h243185be,32'h550c7dc3,32'h72be5d74,32'h80deb1fe,32'h9bdc06a7,32'hc19bf174,
    32'he49b69c1,32'hefbe4786,32'h0fc19dc6,32'h240ca1cc,32'h2de92c6f,32'h4a7484aa,32'h5cb0a9dc,32'h76f988da,
    32'h983e5152,32'ha831c66d,32'hb00327c8,32'hbf597fc7,32'hc6e00bf3,32'hd5a79147,32'h06ca6351,32'h14292967,
    32'h27b70a85,32'h2e1b2138,32'h4d2c6dfc,32'h53380d13,32'h650a7354,32'h766a0abb,32'h81c2c92e,32'h92722c85,
    32'ha2bfe8a1,32'ha81a664b,32'hc24b8b70,32'hc76c51a3,32'hd192e819,32'hd6990624,32'hf40e3585,32'h106aa070,
    32'h19a4c116,32'h1e376c08,32'h2748774c,32'h34b0bcb5,32'h391c0cb3,32'h4ed8aa4a,32'h5b9cca4f,32'h682e6ff3,
    32'h748f82ee,32'h78a5636f,32'h84c87814,32'h8cc70208,32'h90befffa,32'ha4506ceb,32'hbef9a3f7,32'hc67178f2
    },
    parameter kSize =64

    //parameter [2047:0]k = 2048'h428a2f9871374491b5c0fbcfe9b5dba53956c25b59f111f1923f82a4ab1c5ed5hd807aa9812835b01243185be550c7dc372be5d7480deb1fe9bdc06a7c19bf174he49b69c1efbe47860fc19dc6240ca1cc2de92c6f4a7484aa5cb0a9dc76f988dah983e5152a831c66db00327c8bf597fc7c6e00bf3d5a7914706ca635114292967h27b70a852e1b21384d2c6dfc53380d13650a7354766a0abb81c2c92e92722c85ha2bfe8a1a81a664bc24b8b70c76c51a3d192e819d6990624f40e3585106aa070h19a4c1161e376c082748774c34b0bcb5391c0cb34ed8aa4a5b9cca4f682e6ff3h748f82ee78a5636f84c878148cc7020890befffaa4506cebbef9a3f7c67178f2

)
(
    input wire [255:0] flattenedInput,
    output wire [255:0] flattenedOutput
);

wire [31:0] k [63:0];
genvar kVar;
for(kVar=0; kVar<64; kVar=kVar+1) begin: kInitialLoop
    assign k[kVar]= ((kflat>>(32*(kSize-kVar-1)))&{32{1'b1}});
end


wire [31:0] messageSchedule[63:0];
wire [31:0] nonce[7:0];
genvar index1;
for (index1=0; index1 < 8; index1=index1+1) begin: unflatten
    assign nonce[index1] = flattenedInput[255-32*index1:224-32*index1];
end

//copy nonce
genvar copyVar;
for(copyVar=0; copyVar<8; copyVar=copyVar+1) begin: copy
    assign messageSchedule[copyVar]=nonce[copyVar];
end

//insert 1 and pad 0s
assign messageSchedule[8]=32'h80000000;
genvar zeroVar;
for(zeroVar=9; zeroVar<15; zeroVar=zeroVar+1) begin: zeros
    assign messageSchedule[zeroVar]=0;
end
//write in 32 as length into the last 32 bit word
assign messageSchedule[15]=32;

//rest of the schedule array is generated based on algorithm
genvar wVar;
for(wVar=16; wVar<64; wVar=wVar+1) begin: scheduleFill
    wire [31:0] outputwVar;

    MSUnit ms(
        messageSchedule[wVar-15], messageSchedule[wVar-2],
        messageSchedule[wVar-16], messageSchedule[wVar-7],

        outputwVar
    );

    assign messageSchedule[wVar]=outputwVar;
end

wire [255:0] partialHashes [64:0];
//initialize our hash array to our default values
wire [31:0] initialHash[7:0];
assign initialHash[0] = 32'h6a09e667;
assign initialHash[1] = 32'hbb67ae85;
assign initialHash[2] = 32'h3c6ef372;
assign initialHash[3] = 32'ha54ff53a;
assign initialHash[4] = 32'h510e527f;
assign initialHash[5] = 32'h9b05688c;
assign initialHash[6] = 32'h1f83d9ab;
assign initialHash[7] = 32'h5be0cd19;

wire [255:0] flattenedInitialHash;
genvar initialHashVar;
for(initialHashVar=0; initialHashVar<8; initialHashVar=initialHashVar+1) begin: initialHashLoop
    assign flattenedInitialHash[255-32*initialHashVar:224-32*initialHashVar] = initialHash[initialHashVar];
end

assign partialHashes[0]=flattenedInitialHash;

//apply hashing
genvar hashVar;
for(hashVar=0; hashVar<64; hashVar=hashVar+1) begin: hashLoop
    wire [31:0] hashOutput [7: 0];
    wire [255:0] flattenedHashOutput;
    wire [255:0] flattenedPartialInput=partialHashes[hashVar];

    hashRound hr(
        flattenedPartialInput,
        k[hashVar],
        messageSchedule[hashVar],

        flattenedHashOutput
    );

    assign partialHashes[hashVar+1]=flattenedHashOutput;
end


assign flattenedOutput=partialHashes[64];

endmodule