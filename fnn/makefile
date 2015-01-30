CC=g++
CFLAGS=-c -Wall
LDFLAGS=-lfann
SOURCES= main.cpp

all: NN

NN: main.o 
	$(CC) $(LDFLAGS) main.o -o NN

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp

clean:
	rm -rf *o

cleanAll:
	rm -rf *o NN