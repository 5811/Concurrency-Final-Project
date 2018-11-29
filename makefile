CC = g++
FLAGS= -std=c++11 -pthread -Wall -Wextra -Werror

%: %.cpp
		$(CC) $(FLAGS) $^ -o $@
