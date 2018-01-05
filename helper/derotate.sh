#!/bin/sh
if test $# -ne 2
then
   echo "Invalid command line input"
   echo "I expect two command line inputs"
fi

input=$1
output=$2

fits op=xyin in=$input out=p
reorder in=p out=q mode=231
fits op=xyout in=q out=$output
rm -r p q
