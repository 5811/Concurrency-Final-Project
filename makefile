CC = g++
FLAGS= -std=c++11 -pthread -Wall -Wextra -Werror -g

%: %.cpp
		$(CC) $(FLAGS) $^ -o $@
