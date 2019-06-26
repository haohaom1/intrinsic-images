# set the C compiler, for C++ use g++
# use this setting on most sane machines
CC = gcc

# set the path to the include directory
INCDIR =.

# set the flags for the C and C++ compiler to give lots of warnings
CFLAGS = -I$(INCDIR) -O2 -Wall -Wstrict-prototypes -Wnested-externs -Wmissing-prototypes -Wmissing-declarations
CPPFLAGS = $(CFLAGS)

# path to the object file directory
ODIR = obj

# path to the lib directory
LIBDIR =../lib

# path to the bin directory
BINDIR =../bin

# libraries to include
LIBS =-lm
LFLAGS =

# patterns for compiling source code
# $< is the file that caused the action to occur
$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o: %.C $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)


# put the specific executable rules here

makepair: $(ODIR)/makePair.o $(ODIR)/ppmIO.o $(ODIR)/makeMaterialMap.o $(ODIR)/gaussDist.o $(ODIR)/util.o
	$(CC) -o $@ $^ $(LFLAGS) $(LIBS)


.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core 
