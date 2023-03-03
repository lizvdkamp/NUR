#!/bin/bash

echo "Run handin Liz"


echo "Download txt file for in report..."
if [ ! -e Vandermonde.txt ]; then
  wget home.strw.leidenuniv.nl/~daalen/Handin_files/Vandermonde.txt
fi

# Script that returns a txt file
echo "Run the first script ..."
python3 NURHW1LizQ1.py 

# Script that pipes output to multiple files
echo "Run the second script ..."
python3 NURHW1LizQ2.py


echo "Generating the pdf"

pdflatex Solutions.tex
bibtex Solutions.aux
pdflatex Solutions.tex
pdflatex Solutions.tex
