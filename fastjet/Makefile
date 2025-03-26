
RFLAGS = `root-config --cflags --libs` -lz
FJFLAGS = `fastjet-config --cxxflags --libs --plugins`
CFLAGS = -g -O3 -Wall -Wextra -Wpedantic -fopenmp

# Object directory
OBJDIR = obj

# Possible source code extensions
EXTS = .c .C .cc .cxx

# Source files and object files for plots
SRCS = doHisto.cc MakeOmni.C jetHelpers.cxx cnpy.cxx
OBJS = $(patsubst %.c,obj/%.o,$(patsubst %.C,obj/%.o,$(patsubst %.cc,obj/%.o,$(patsubst %.cxx,obj/%.o,$(SRCS)))))

# Target files
PLOT = doHisto.out

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

# Clean rule
clean:
	rm -rf $(OBJDIR) $(PLOT)

hello:
	echo "Hello world!"

test:
	echo $(CXX)