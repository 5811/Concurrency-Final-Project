#include <stdlib.h>
#include <string>
#include <iostream>
#include <unistd.h>
#include <getopt.h>





static struct option long_options[]=
                {
                    {"target", required_argument, nullptr, 't'},
                    {"workers", required_argument, nullptr, 'w'},
                    {"step", required_argument, nullptr, 's'}
                };

static std::string target="";
static int workers=1;
static int step=1;

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

}