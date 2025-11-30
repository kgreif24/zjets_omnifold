
RFLAGS = `root-config --cflags --libs` -lz
FJFLAGS = `fastjet-config --cxxflags --libs --plugins`
CFLAGS = -g -O3 -Wall -Wextra -Wpedantic -fopenmp -I./indicators/include -I./cnpy

# Object directory
OBJDIR = obj

# Possible source code extensions
EXTS = .c .C .cc .cxx .cpp

# Source files and object files for plots
SRCS = doHisto.cc MakeOmni.C jetHelpers.cxx cnpy/cnpy.cpp HistoGroup.cxx
# Handle object file paths, including subdirectory paths
OBJS = $(patsubst %.c,obj/%.o,$(patsubst %.C,obj/%.o,$(patsubst %.cc,obj/%.o,$(patsubst %.cxx,obj/%.o,$(patsubst %.cpp,obj/%.o,$(SRCS))))))
# Fix path for cnpy submodule (cnpy/cnpy.cpp -> obj/cnpy/cnpy.o)
OBJS := $(patsubst obj/cnpy/cnpy.o,obj/cnpy_cnpy.o,$(OBJS))

# Target files
PLOT = doHisto.out

# All target
all: $(PLOT) $(ADDLJP)

# Rule to create object directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Compilation rule for .c, .C, .cc, .cxx, and .cpp files into .o files
$(OBJDIR)/%.o: %.c | $(OBJDIR)
	$(CXX) $(CFLAGS) $(RFLAGS) $(FJFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.C | $(OBJDIR)
	$(CXX) $(CFLAGS) $(RFLAGS) $(FJFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.cc | $(OBJDIR)
	$(CXX) $(CFLAGS) $(RFLAGS) $(FJFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.cxx | $(OBJDIR)
	$(CXX) $(CFLAGS) $(RFLAGS) $(FJFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	$(CXX) $(CFLAGS) $(RFLAGS) $(FJFLAGS) -c $< -o $@

# Special rule for cnpy submodule
$(OBJDIR)/cnpy_cnpy.o: cnpy/cnpy.cpp | $(OBJDIR)
	$(CXX) $(CFLAGS) $(RFLAGS) $(FJFLAGS) -c $< -o $@

# Linking for plot
$(PLOT): $(OBJDIR) $(OBJS)
	$(CXX) $(CFLAGS) $(RFLAGS) $(FJFLAGS) -o $(PLOT) $(OBJS)

# Clean rule
clean:
	rm -rf $(OBJDIR) $(PLOT)

hello:
	echo "Hello world!"

test:
	echo $(CXX)