
This is a copy of Jad's omniphysics package: https://github.com/jmsardain/omniphysics. It can be used to plot arbitrary observables defined on the Z+jets final state. To compile the code, it is easiest to use an LCG release obtained from `cvmfs`:

```
setupATLAS
lsetup "views LCG_105 x86_64-el9-gcc13-opt"
```

I am currently trying to get this to work on Perlmutter via shifter containers, but haven't figured it out yet. Will update when it is figured out.

# Calculate observables

You first need to append omniweights to the MC file

```
python addWeightToTree.py
```

Then you need to install fastjet. I have fastjet-3.4.1 

```
## please check https://fastjet.fr/quickstart.html
curl -O https://fastjet.fr/repo/fastjet-3.4.1.tar.gz
tar zxvf fastjet-3.4.1.tar.gz
cd fastjet-3.4.1/

./configure --prefix=$PWD/../fastjet-install
make
make check
make install
cd ..
```

To compile: 
```

setupATLAS 
lsetup "root 6.30.02-x86_64-el9-gcc13-opt" 

COMPILER=$(root-config --cxx)
FLAGS=$(root-config --cflags --libs)
echo $COMPILER $FLAGS
$COMPILER $FLAGS -g -O3 -Wall -Wextra -Wpedantic -fopenmp ./doHisto.cc ./MakeOmni.C ./MakeTruth.C  ./DoPlots.cxx ./analysisHelpers.cxx ./jetHelpers.cxx ./AtlasLabels.C ./AtlasStyle.C ./AtlasUtils.C `fastjet-install/bin/fastjet-config --cxxflags --libs --plugins` -I. -o doHisto
```

To run, it's a 2 step approach (not really needed but well..). First run to get an output root file 
```
./doHisto reco 
```

then run to get the plots in ATLAS style 
```
./doHisto plots
```

# Add Lund Jet Plane information