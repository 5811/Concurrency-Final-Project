module MSUnit(
    input wire [0:31] w15,
    input wire [0:31] w2,
    input wire [0:31] w16,
    input wire [0:31] w7,

    output wire [0:31] w0

);
    wire unsigned [0:31] S0 = {w15[25 : 31], [0 : 24]} ^ {w15[14 : 31], [0 : 13]} ^ {w15 >> 3};
    wire unsigned [0:31] S1 =  {w2[]15 : 31], [0 : 14]} ^ {w2[13 : 31], [0 : 12]} ^ {w2 >> 10};
    
    assign w0=w16+S0+w7+S1;

endModule

module hashRound
(
    input wire unsigned [0:31] workingVariables[0:7],
    input wire unsigned [0:31] k,
    input wire unsigned [0:31] w,

    output wire unsigned [0:31] nWorkingVariables[0:7]

);

    wire unsigned [0:31] e=workingVariables[4];
    wire unsigned [0:31] a=workingVariables[0]
    wire unsigned [0:31] S1 =  {e[26 : 31], [0 : 25]} ^ {e[21 : 31], [0 : 20]} ^ {e[7 : 31], [0 : 6]};
    wire unsigned [0:31] ch = (workingVariables[4] & workingVariables[5]) ^ ((~ workingVariables[4]) & workingVariables[6]);
    wire unsigned [0:31] temp1 = workingVariables[7] + S1 + ch + k + w;
    wire unsigned [0:31] S0 = {a[30 : 31], [0 : 29]} ^ {a[19 : 31], [0 : 18]} ^ {a[10 : 31], [0 : 9]};
    wire unsigned [0:31] maj = (workingVariables[0] & workingVariables[1]) ^ (workingVariables[0] & workingVariables[2]) ^ (workingVariables[1] & workingVariables[2]);
    wire unsigned [0:31]t temp2 = S0 + maj;

    assign nWorkingVariables[7] = workingVariables[6];
    assign nworkingVariables[5] = workingVariables[4];
    assign nworkingVariables[4] = workingVariables[3]+ temp1;
    assign nworkingVariables[6] = workingVariables[5];
    assign nworkingVariables[3] = workingVariables[2];
    assign nworkingVariables[2] = workingVariables[1];
    assign nworkingVariables[1] = workingVariables[0];
    assign nworkingVariables[0] = temp1 + temp2;


endModule

//hashes a nonce that is a 32 byte value
module hashModule(

    localparam [31:0] k[0:63] =
    {
    32'h428a2f98,32'h71374491,32'hb5c0fbcf,32'he9b5dba5,32'h3956c25b,32'h59f111f1,32'h923f82a4,32'hab1c5ed5,
    32'hd807aa98,32'h12835b01,32'h243185be,32'h550c7dc3,32'h72be5d74,32'h80deb1fe,32'h9bdc06a7,32'hc19bf174,
    32'he49b69c1,32'hefbe4786,32'h0fc19dc6,32'h240ca1cc,32'h2de92c6f,32'h4a7484aa,32'h5cb0a9dc,32'h76f988da,
    32'h983e5152,32'ha831c66d,32'hb00327c8,32'hbf597fc7,32'hc6e00bf3,32'hd5a79147,32'h06ca6351,32'h14292967,
    32'h27b70a85,32'h2e1b2138,32'h4d2c6dfc,32'h53380d13,32'h650a7354,32'h766a0abb,32'h81c2c92e,32'h92722c85,
    32'ha2bfe8a1,32'ha81a664b,32'hc24b8b70,32'hc76c51a3,32'hd192e819,32'hd6990624,32'hf40e3585,32'h106aa070,
    32'h19a4c116,32'h1e376c08,32'h2748774c,32'h34b0bcb5,32'h391c0cb3,32'h4ed8aa4a,32'h5b9cca4f,32'h682e6ff3,
    32'h748f82ee,32'h78a5636f,32'h84c87814,32'h8cc70208,32'h90befffa,32'ha4506ceb,32'hbef9a3f7,32'hc67178f2
    }

)
(
    input wire unsigned [0: 31] nonce[0:7],
    output wire unsigned[0: 31] hash[0:7]
);

wire unsigned [0:31] messageSchedule[0:63];

//copy nonce
genvar copyVar;
for(copyVar=0; copyVar<8; copyVar=copyVar+1) begin: copy
    messageSchedule[copyVar]=nonce[copyVar];

end

//insert 1 and pad 0s
messageSchedule[8]=32'80000000;
genvar zeroVar;
for(zeroVar=9; zeroVar<14; zeroVar=zeroVar+1) begin: zeros
    messageSchedule[zeroVar]=nonce[zeroVar];

//write in 32 as length into the last 32 bit word
messageSchedule[15]=32;

//rest of the schedule array is generated based on algorithm
genvar wVar;
for(wVar=16; wVar<64; wVar=wVar+1) begin: scheduleFill
    wire unsigned [0:31] outputwVar;

    MSUnit ms(
        messageSchedule[wVar-15], messageSchedule[wVar-2],
        messageSchedule[wVar-16], messageSchedule[wVar-7],

        outputwVar
    );

    messageSchedule[wVar]=outputwVar;
end

wire unsigned [0:31] partialHashes [0:7][0:64];
//initialize our hash array to our default values
wire unsigned [0:31] initialHash[0:7]={
    32'6a09e667, 
    32'bb67ae85,
    32'3c6ef372,
    32'a54ff53a,
    32'510e527f,
    32'9b05688c,
    32'1f83d9ab,
    32'5be0cd19,
};

assign partialHashes[0]=initialHash;


//apply hashing
genvar hashVar;
for(hashVar=0; hashVar<63; hashVar=hashVar+1) begin: hashLoop
    wire unsigned [0:31] hashOutput [0: 7];
    hashRound hr(
        partialHashes[hashVar],
        k[hashVar],
        messageSchedule[hashVar],

        hashOutput


    );

    assign partialHashes[hashVar+1]=hashOutput;
end

assign hash=partialHashes[64];

endModule