CC = gcc -Wall 

all:
	$(CC) -I/home/piotrek/gls/include -c c_multiplication.c
	gcc -L/home/piotrek/gls/lib c_multiplication.o -lgsl -lgslcblas -lm

clean:
	rm -f c_multiplication.o a.out