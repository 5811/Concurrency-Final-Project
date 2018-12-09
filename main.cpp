#include <stdlib.h>
#include <string>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <getopt.h>
#include <stdint.h>
#include <atomic>
#include <iomanip>
#include <time.h>
#include <pthread.h>
#include <thread>
#include <time.h>

#include <cuda.cpp>

static struct option long_options[]=
                {
                    {"target", required_argument, nullptr, 't'},
                    {"workers", required_argument, nullptr, 'w'},
                    {"step", required_argument, nullptr, 's'}
                };

static std::string target="";
static int workers=1;
static int step=1;

struct timespec start, end;

void startTimer(){
    clock_gettime(CLOCK_MONOTONIC, &start);
}
double endTimer(){
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time= end.tv_sec-start.tv_sec;
    time=time+(end.tv_nsec-start.tv_nsec)/1000000000.0;
    return time;
}

//function for printing hashes as a hexstring, assumes the hash is an array of uint32_t of length 8
void printHash(uint32_t* hash){
    for(int i=0; i<8; i++){
        std::cout<<std::hex<<std::setfill('0')<<std::setw(8)<<unsigned(hash[i]);
    }
    std::cout<<std::endl;
}
void printNonce(unsigned char* charPointer){;
    for(uint32_t i=0; i<8*sizeof(uint32_t); i++){
        //std::cout<< "Byte[" << i << "]" << std::hex<<std::setfill('0')<<std::setw(2)<< int(charPointer[i])  << std::endl;
        std::cout<<std::hex<<std::setfill('0')<<std::setw(2)<<int(charPointer[i]);
    }
    std::cout<<std::endl;
}
//rotation functions based off this blog post https://blog.regehr.org/archives/1063 
uint32_t rightRotate (uint32_t value, uint32_t offset)
{
  return (value>>offset) | (value<<(-offset&31));
}


const uint32_t roundConstants[64] = 
{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

//hashes a 512 bit/64 byte chunk
void hashChunk(char* chunk, uint32_t* currentHash){
    uint32_t scheduleArray[64];

	/*for (int i = 0; i < 64; i++) {
        std::cout << "Chunk[" << i << "] :" << int(chunk[i]) << std::endl;
    }*/

	#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
	std::memcpy(scheduleArray, chunk, 64);
	#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
	char * charScheduleArray = (char*)scheduleArray;
	for (int i = 0; i < 64; i+=4) {
		charScheduleArray[i] = chunk[i+3];
		charScheduleArray[i+1] = chunk[i+2];
		charScheduleArray[i+2] = chunk[i+1];
		charScheduleArray[i+3] = chunk[i];
    }
	/*for (int i = 0; i < 64; i++) {
        std::cout << "Char[" << i << "] :" << int(charScheduleArray[i]) << std::endl;
    }*/
	#endif
	
    for(int i= 16; i<64; i++){
        uint32_t s0 = rightRotate(scheduleArray[i-15], 7) ^ rightRotate(scheduleArray[i-15], 18) ^ (scheduleArray[i-15] >> 3);
        uint32_t s1 = rightRotate(scheduleArray[i-2], 17) ^ rightRotate(scheduleArray[i-2] , 19) ^ (scheduleArray[i-2] >> 10);
        scheduleArray[i] = scheduleArray[i-16] + s0 + scheduleArray[i-7] + s1;
    }

	/*for (int i = 0; i < 64; i++) {
    	std::cout << "Array " << i << " : " << scheduleArray[i] << std::endl;
    }*/

    uint32_t workingVariables[8];
    std::memcpy(workingVariables, currentHash, 32);

    for (int i=0; i<64; i++){
        uint32_t S1 = rightRotate(workingVariables[4], 6) ^ rightRotate(workingVariables[4], 11) ^ rightRotate(workingVariables[4], 25);
        uint32_t ch = (workingVariables[4] & workingVariables[5]) ^ ((~ workingVariables[4]) & workingVariables[6]);
        uint32_t temp1 = workingVariables[7] + S1 + ch + roundConstants[i] + scheduleArray[i];
        uint32_t S0 = rightRotate(workingVariables[0] , 2) ^ rightRotate(workingVariables[0] , 13) ^ rightRotate(workingVariables[0] , 22);
        uint32_t maj = (workingVariables[0] & workingVariables[1]) ^ (workingVariables[0] & workingVariables[2]) ^ (workingVariables[1] & workingVariables[2]);
        uint32_t temp2 = S0 + maj;
 
        workingVariables[7] = workingVariables[6];
        workingVariables[6] = workingVariables[5];
        workingVariables[5] = workingVariables[4];
        workingVariables[4] = workingVariables[3]+ temp1;
        workingVariables[3] = workingVariables[2];
        workingVariables[2] = workingVariables[1];
        workingVariables[1] = workingVariables[0];
        workingVariables[0] = temp1 + temp2;
    }

    for(int i=0; i<8; i++){
        currentHash[i]+=workingVariables[i];
    }

}

inline uint32_t ceilingIntDivision(const uint32_t val, const uint32_t mod) { //assert mod != 0, val != 0
    return (val + mod - 1) / mod;
}

inline uint32_t generateProcessSize(const uint32_t inputSize) {
    uint32_t inputSizeInBits = inputSize * 8;
    uint32_t paddedInBits = inputSizeInBits + 64 + 1;
    return ceilingIntDivision(paddedInBits, 512) * 512 / 8;
}

inline void writeIntToBufferAsBigEndian(char* start, const uint64_t value) {
    *((uint64_t*) start) = value;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    for (unsigned i = 0; i < sizeof(uint64_t) / 2; ++i) {
        char temp = start[i];
        start[i] = start[sizeof(uint64_t) - 1 - i];
        start[sizeof(uint64_t) - 1 - i] = temp; 
    }
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    // Do nothing
#elif __BYTE_ORDER__ == __ORDER_PDP_ENDIAN__
    #error "PDP byte ordering not handled."
#endif
}

void hash(const char* input, const uint64_t size, uint32_t* result){
    //initialize hash values
    uint32_t hash[]={
    0x6a09e667, 
    0xbb67ae85,
    0x3c6ef372,
    0xa54ff53a,
    0x510e527f,
    0x9b05688c,
    0x1f83d9ab,
    0x5be0cd19,
    };

    //preprocessing

    //processedSize is size of padded array in bytes
    uint32_t processedSize=generateProcessSize(size);

    char* buffer=(char*) malloc(processedSize);
    //copy over input
    std::memcpy(buffer, input, size);
    //append 1 bit and 0 bits
    buffer[size]=0x80;
    std::memset(buffer+size+1, 0, processedSize-size-1-sizeof(uint64_t));
    //append size
    writeIntToBufferAsBigEndian(buffer+processedSize-sizeof(uint64_t), size*8);
    
	/*for (int i = 0; i < 64; i++) {
		std::cout << "Buffer[" << i << "] :" << int(buffer[i]) << std::endl;
	}*/
    //hash chunks in a loop
    for(uint32_t i=0; i<processedSize/64; i++){
        char* chunkAddress=buffer+(i*64);
        hashChunk(chunkAddress, hash);
    }

    //free memory we allocated earlier
    free(buffer);
    std::memcpy(result, hash, 8*sizeof(uint32_t));
    return;
}
static std::atomic_flag lock = ATOMIC_FLAG_INIT;
static bool done=false;

void incrementNonce(uint32_t* value, uint32_t increment){
     int i=0;
        while(true){
            if(i==8){
                std::cout<<"Checked all possible 512 bit values! Still no nonce found!"<<std::endl;
            }
            uint32_t temp=i==0? value[i]+increment : value[i]+1;
            if(temp<value[i]){
                //overflow, increment i to handle the carry
                value[i]=temp;
                i++;
            }
            else{
                value[i]=temp;
                break;
            }
        }
}
void searchForNonceThread(uint16_t leadingByte, uint32_t startingValue, uint32_t increment, uint32_t* result){
    //allocate 32 bytes for the nonce we are going to hash
    //since we are using SHA 256, the number of possible inputs with 32 bytes is the same as the number of possible hashes
    //there may be some overlap, but we are not looking for an exact hash so it should be reasonable to expect us to find a working nonce
    //before exhausting all possible nonces
    uint32_t nonce[8]={0};
    nonce[0]=startingValue;

    uint32_t tempHash[8];

    while(true){
        //generate hash
        hash((char*)nonce, 32, tempHash);

        //get done lock and check if we have found a working nonce or if another thread is done 
        while (lock.test_and_set(std::memory_order_acquire)); 
        if(done==true){
            lock.clear(std::memory_order_release);
            return;
        }
        else{
                if(((uint16_t*)tempHash)[1]==leadingByte){
                    done=true;
                    /*
                    std::cout<<"Found something that hashes to: ";
                    printHash(tempHash);
                    std::cout<<"leading bytes: ";
                    std::cout<<std::hash<<(tempHash)[0];
                    std::cout<<std::endl;
                    */
                    lock.clear(std::memory_order_release);
                    std::memcpy(result, nonce, 8*sizeof(uint32_t));
                    return;
                }
        }
        lock.clear(std::memory_order_release);

        //increment
       incrementNonce(nonce, increment);
    }




}

void searchForNonceGPU(uint16_t leadingByte, uint32_t blocks, uint32_t* result){


    uint32_t* gpuResult;

    cudaMalloc((void**)&gpuResult, 8*sizeof(uint32_t));

	int *deviceDone;

	cudaMalloc((void**)&deviceDone, sizeof(int));
	cudaMemcpy(deviceDone,0,sizeof(int),cudaMemcpyHostToDevice); 

    searchForNonce<<<blocks, 32>>>(leadingByte, gpuResult, deviceDone);

	cudaFree(deviceDone);

    cudaMemcpy(result, gpuResult, 8*sizeof(uint32_t), cudaMemcpyDeviceToHost);

}


int main(int argc, char** argv){

     //parse options
    while(true)
    {
            const auto opt= getopt_long(argc, argv, "t:w:s:", long_options, nullptr);

            if(opt==-1)
                    break;

            switch(opt)
            {
                    case 't':
                            target=std::string(optarg);
                            std::cout<<"Target leading characters of hash: "<< target<<std::endl;
                            break;

                    case 'w':
                            workers=std::stoi(optarg);
                            std::cout<<"workers: "<< workers<<std::endl;
                            break;
                    case 's':
                            step=std::stoi(optarg);
                            std::cout<<"Step: "<< step<<std::endl;
                            break;
                    default:
                            std::cout<<"Undefined option found"<<std::endl;
            }
    }

    uint32_t hashResult[8];
    unsigned char nonce[32];
    cudaEvent_t gpuStart;


    //array of worker threads if we need them
    std::thread workerThreads[workers];


    double timeTaken;
    switch(step){
        case 1:


            startTimer();
            searchForNonceThread(0, 0, 1, (uint32_t*)nonce);
            timeTaken=endTimer();
            
            break;
        case 2:
            startTimer();
            for(int i=0; i<workers; i++){
                workerThreads[i]=std::thread(searchForNonceThread, 0, i,workers,(uint32_t*)nonce);

            }
            for(int i=0; i<workers; i++){
                workerThreads[i].join();

            }
            timeTaken=endTimer();

            break;
        case 3:

            gpuStart=getGpuTime();
            searchForNonceGPU(0, workers, (uint32_t*) nonce);
            timeTaken=(double)getElapsedGpuTime(gpuStart);
            break;

        default:
            std::cout<<"Undefined step"<<std::endl;
    }


    


    printNonce(nonce);
    std::cout<<"hashes to"<<std::endl;
    hash((char*)nonce,32, hashResult);
    printHash(hashResult);

    std::cout<<"Time taken: "<<timeTaken<<" seconds"<<std::endl;

}
