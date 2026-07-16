ROOTCFLAGS := $(shell root-config --cflags)
ROOTLIBS   := -L$(shell root-config --libdir) -lCore -lRIO -lTree -lThread -Wl,-rpath,$(shell root-config --libdir)

extract_hits: extract_hits.cpp
	g++ -O2 -march=native $(ROOTCFLAGS) -o $@ $< $(ROOTLIBS)

clean:
	rm -f extract_hits
