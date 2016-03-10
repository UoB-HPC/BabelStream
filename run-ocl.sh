#! /bin/bash

# Author : Minh Quan HO
# This script runs GPU-STREAM benchmark using OpenCL in varying array size, 
# then prints out results in column for R-friendly people. 
# Command : $ ./run-ocl.sh

BEGIN=204800
END=102400000
STEP=409600

PRECISION="double"   # double float
NUMTIMES=10          # Run the test NUM times (NUM >= 2, default = 10)

DEVICE_ID=0          # default device on which run benchmark
DEVICE_NAME=$(./gpu-stream-ocl --list | grep "$DEVICE_ID:" | cut -d':' -f2)

if [[ $PRECISION == "double" ]]; then
	ELEM_SIZE=8 # in bytes
	OPTS=""
else
	ELEM_SIZE=4 # in bytes
	OPTS="--float"
fi

echo    "# Benchmark GPU-STREAM running on $DEVICE_NAME"
echo    "# Precision: $PRECISION. Range: [$BEGIN .. $END] step $STEP"
echo    "# For more details see https://github.com/UoB-HPC/GPU-STREAM"
echo -n "#  ArrayElements    ArraySize(MB)      "
echo    "Copy(MBytes/s)    Mul(MBytes/s)      Add(MBytes/s)      Triad(MBytes/s)"
for nb_elem in $(seq $BEGIN $STEP $END)
do 
	array_size=$(echo "scale=2; $nb_elem * $ELEM_SIZE / 1048576" | bc -l) # total array size in MB

	echo -n "    $nb_elem            $array_size             "

	# Run benchmark and get results on Copy, Add, Mul and Triad
	all=$(./gpu-stream-ocl $OPTS -s $nb_elem -n $NUMTIMES --device $DEVICE_ID | grep -e Copy -e Mul -e Add -e Triad)
	
	# Loop test and print in column
	echo "$all" |
	while read line
	do 
		if   [[ $line == "Copy"* ]]; then
			copy=$(echo $line | awk '{print $2}')
			echo -n "$copy          "
		elif [[ $line == "Mul"* ]]; then
			mul=$(echo $line | awk '{print $2}')
			echo -n "$mul          "
		elif [[ $line == "Add"* ]]; then
			add=$(echo $line | awk '{print $2}')
			echo -n "$add          "
		elif [[ $line == "Triad"* ]]; then
			triad=$(echo $line | awk '{print $2}')
			echo -n "$triad          "
		fi
	done
	echo ""
done

exit 0
