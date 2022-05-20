CC=g++
CFLAGS=-std=c++17 -c  -O3 -march=native -I/opt/local/include  
LDFLAGS=
SOURCES=Wannier_U_MC.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE= Wannier_U_MC.x

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(LDFLAGS) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o Wannier_U_MC.x