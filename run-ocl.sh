#! /bin/bash

# Author : Minh Quan HO
# This script runs GPU-STREAM benchmark using OpenCL in varying array size, 
# then prints out results in column for R-friendly people. 
# Command : $ ./run-ocl.sh --help

BEGIN=204800
END=20848640
STEP=81920

PRECISION="double"   # double float
NUMTIMES=10          # Run the test NUM times (NUM >= 2, default = 10)

BINARY_NAME="gpu-stream-ocl"
TIMEOUT=20

DEVICE_ID=0          # default device on which run benchmark

if [[ $PRECISION == "double" ]]; then
	ELEM_SIZE=8 # in bytes
	OPTS=""
else
	ELEM_SIZE=4 # in bytes
	OPTS="--float"
fi

usage()
{
cat << EOF

Usage: $0 [ OPTIONS ]

OPTIONS : 
   -b, --begin       SIZE          Begin range of nb_elements (default=$BEGIN)
   -e, --end         SIZE          End range of nb_elements (default=$END)
   -s, --step        SIZE          Step range of nb_elements (default=$STEP)
   -d, --device-id  INDEX          Device ID to use (default=$DEVICE_ID)
   -n, --numtimes    NUM           Run the test NUM times (NUM >= 2)
       --timeout     NUM           Timeout of each test launch (default=$TIMEOUT)
   -t, --exec     BINARY_NAME      Executable to run (default=$BINARY_NAME)
   -p, --precision {double|float}  Precision (default=$PRECISION)
   -h, --help                      Print this help and quit
   
EOF
}

#####################################################################
# Process arguments
#####################################################################
if [[ $# -le 0 ]];then
	usage
	exit 1
fi

ARGS=$(getopt -o b:e:s:d:n:t:p:h \
	-l "begin:,end:,step:,device-id:numtimes:,timeout:,exec:,precision:" \
	-n "$0" -- "$@");

if [ $? -ne 0 ];then
	usage
 	exit 1
fi

eval set -- "$ARGS";
while true; do
  	case "$1" in
		-h|--help)
			usage
			exit 0
			;;
		-b|--begin)
			shift
			if [ -n "$1" ]; then
				BEGIN=$1
				shift
			fi
			;;
		-e|--end)
			shift
			if [ -n "$1" ]; then
				END=$1
				shift
			fi
			;;
		-s|--step)
			shift
			if [ -n "$1" ]; then
				STEP=$1
				shift
			fi
			;;
		-d|--device-id)
			shift
			if [ -n "$1" ]; then
				DEVICE_ID=$1
				shift
			fi
			;;
		-n|--numtimes)
			shift
			if [ -n "$1" ]; then
				NUMTIMES=$1
				shift
			fi
			;;
		--timeout)
			shift
			if [ -n "$1" ]; then
				TIMEOUT=$1
				shift
			fi
			;;
		-t|--exec)
			shift
			if [ -n "$1" ]; then
				BINARY_NAME=$1
				shift
			fi
			;;
		-p|--precision)
			shift
			if [ -n "$1" ]; then
				PRECISION=$1
				shift
			fi
			;;
		*)
			break
			;;
  	esac
done

DEVICE_NAME=$(./$BINARY_NAME --list | grep "$DEVICE_ID:" | cut -d':' -f2)

echo    "# Benchmark GPU-STREAM running on $DEVICE_NAME"
echo    "# Precision: $PRECISION. Range: [$BEGIN .. $END] step $STEP"
echo    "# For more details see https://github.com/UoB-HPC/GPU-STREAM"
echo -n "#  ArrayElements    ArraySize(MB)      "
echo    "Copy(MBytes/s)    Mul(MBytes/s)      Add(MBytes/s)      Triad(MBytes/s)"
for nb_elem in $(seq $BEGIN $STEP $END)
do 
	error="ok"
	array_size=$(echo "scale=2; $nb_elem * $ELEM_SIZE / 1048576" | bc -l) # total array size in MB

	echo -n "    $nb_elem            $array_size             "

	tmp="/tmp/"$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)

	# Run benchmark and get results on Copy, Add, Mul and Triad
	# I observed a kernel bug of Intel driver, freeze sometimes on Xeon Phi, 
	# so command is wrapped by timeout and repeated until it passed
	timeout $TIMEOUT bash -c "./$BINARY_NAME $OPTS -s $nb_elem -n $NUMTIMES --device $DEVICE_ID | grep -E \"|Copy|Mul|Add|Triad|error\" > $tmp"
	while [ $? -ne 0 ]
	do
		timeout $TIMEOUT bash -c "./$BINARY_NAME $OPTS -s $nb_elem -n $NUMTIMES --device $DEVICE_ID | grep -E \"|Copy|Mul|Add|Triad|error\" > $tmp"
	done

	# Loop on results and print in column
	while read line
	do 
		if   [[ $line == *"Copy"* ]]; then
			copy=$(echo $line | awk '{print $2}')
			echo -n "$copy          "
		elif [[ $line == *"Mul"* ]]; then
			mul=$(echo $line | awk '{print $2}')
			echo -n "$mul          "
		elif [[ $line == *"Add"* ]]; then
			add=$(echo $line | awk '{print $2}')
			echo -n "$add          "
		elif [[ $line == *"Triad"* ]]; then
			triad=$(echo $line | awk '{print $2}')
			echo -n "$triad          "
		elif [[ $line == *"error"* ]]; then
			# Test failed on false results of a, b, c
			error="failed"
		fi
	done < $tmp
	echo "     $error" # print if test is failed
	rm -f $tmp

	# sleep some seconds to cool down device
	sleep 2
done

exit 0
