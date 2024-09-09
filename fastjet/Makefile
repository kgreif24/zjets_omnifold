
RFLAGS = `root-config --cflags --libs`
FJFLAGS = `fastjet-config --cxxflags --libs --plugins`
CFLAGS = -g -O3 -Wall -Wextra -Wpedantic -fopenmp

# Object directory
OBJDIR = obj

# Possible source code extensions
EXTS = .c .C .cc .cxx

# Source files and object files for plots
SRCS = doHisto.cc MakeOmni.C MakeTruth.C DoPlots.cxx analysisHelpers.cxx jetHelpers.cxx AtlasLabels.C AtlasStyle.C AtlasUtils.C
OBJS = $(patsubst %.c,obj/%.o,$(patsubst %.C,obj/%.o,$(patsubst %.cc,obj/%.o,$(patsubst %.cxx,obj/%.o,$(SRCS)))))

# Source files and object files for add LJP
LSRCS = addLJP.cc
LOBJS = $(patsubst %.c,obj/%.o,$(patsubst %.C,obj/%.o,$(patsubst %.cc,obj/%.o,$(patsubst %.cxx,obj/%.o,$(LSRCS)))))

# Target files
PLOT = doHisto.out
ADDLJP = addLJP.out

# All target
all: $(PLOT) $(ADDLJP)

# Rule to create object directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Compilation rule for .c, .C, .cc, and .cxx files into .o files
$(OBJDIR)/%.o: %.c | $(OBJDIR)
	$(CXX) $(CFLAGS) $(RFLAGS) $(FJFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.C | $(OBJDIR)
	$(CXX) $(CFLAGS) $(RFLAGS) $(FJFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.cc | $(OBJDIR)
	$(CXX) $(CFLAGS) $(RFLAGS) $(FJFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.cxx | $(OBJDIR)
	$(CXX) $(CFLAGS) $(RFLAGS) $(FJFLAGS) -c $< -o $@

# Linking for plot
$(PLOT): $(OBJDIR) $(OBJS)
	$(CXX) $(CFLAGS) $(RFLAGS) $(FJFLAGS) -o $(PLOT) $(OBJS)

# Linking for add LJP
$(ADDLJP): $(OBJDIR) $(LOBJS)
	$(CXX) $(CFLAGS) $(RFLAGS) $(FJFLAGS) -o $(ADDLJP) $(LOBJS)

# Clean rule
clean:
	rm -rf $(OBJDIR) $(PLOT)

hello:
	echo "Hello world!"

test:
	echo $(CXX)