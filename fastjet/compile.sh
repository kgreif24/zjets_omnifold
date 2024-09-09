#!/bin/bash

setupATLAS 
lsetup "views LCG_105 x86_64-el9-gcc13-opt" 

echo Cleaning object files and compiling from scratch
make clean
make
