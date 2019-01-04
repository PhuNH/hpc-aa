#!/bin/bash

echo "Running scenarios..."

. ../.env

make

for size in 8 16 32 64 128 256 512 1024
do
    ./sparse -H $size > 'heq_ellpack_s'$size'.log'
    ./sparse -H -C $size > 'heq_cusparse_s'$size'.log'
    ./sparse -H -B $size > 'heq_band_s'$size'.log'
done

