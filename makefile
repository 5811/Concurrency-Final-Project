
objects = main.o


FLAGS= -std=c++11 -O3 -Xcompiler="-pthread -Wall -Wextra -Werror"

main: $(objects)
		nvcc -arch=sm_60 $(objects) -o main $(FLAGS)


%.o: %.cpp
		nvcc -x cu -arch=sm_60 -I. -dc $< -o $@ $(FLAGS)

clean:
		rm -f *.o main
