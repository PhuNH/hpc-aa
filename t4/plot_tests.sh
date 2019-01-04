# Sam(oa)² - SFCs and Adaptive Meshes for Oceanic And Other Applications
# Copyright (C) 2010 Oliver Meister, Kaveh Rahnema
# This program is licensed under the GPL, for details see the file LICENSE


#!/bin/bash
echo -n "Plotting results... "

echo "#Time by kernel" > "results.plt"
echo "#kernel size time[s]" >> "results.plt"

for file in *.log; do
	kernel=$(echo $file | grep -oE "heq_[a-zA-Z0-9]+" | tail -c+5)
	size=$(echo $file | grep -oE "_s[0-9]+" | tail -c+3)

	echo -n $kernel $size" " >> "results.plt"
	grep -E "Total time" $file | grep -oE "[0-9\.]+" | tr "\n" " " | cat >> "results.plt"
	echo "" >> "results.plt"
done

sort -t" " -n -k 2,2 results.plt -o results.plt
sort -t" " -s -k 1,1 results.plt -o results.plt

join -1 1 -2 2 -o 1.1 1.2 1.3 2.3 <(join -j 2 -o 1.2 1.3 2.3 <(cat results.plt | grep ellpack) <(cat results.plt | grep cusparse)) <(cat results.plt | grep band) > results_joined.plt

mv results_joined.plt results.plt

#gnuplot &> /dev/null << EOT
gnuplot << EOT

set terminal postscript enhanced color font ',20'
set xlabel "Matrix size n"
set ylabel "time [s]"
set key below font ",16"

set style line 1 lt 2 lw 8 lc rgb "black"
set style line 2 lt 1 lc rgb "red" lw 4
set style line 3 lt 1 lc rgb "blue" lw 4
set style line 4 lt 1 lc rgb "green" lw 4

set title "Performance by Heat Equation kernel"
set output '| ps2pdf - results.pdf'

set xtic 128

plot "results.plt" u (\$1):2 ls 2 w linespoints t "ellpack", \
    "results.plt" u (\$1):3 ls 3 w linespoints t "cusparse", \
    "results.plt" u (\$1):4 ls 4 w linespoints t "band"

set xtic 2
set log x 2
set ylabel "time difference [s]"

set output '| ps2pdf - results_diff.pdf'

plot "results.plt" u (\$1):(\$2-\$2) ls 2 w linespoints t "ellpack", \
    "results.plt" u (\$1):(\$3-\$2) ls 3 w linespoints t "cusparse", \
    "results.plt" u (\$1):(\$4-\$2) ls 4 w linespoints t "band"

EOT

echo "done."
