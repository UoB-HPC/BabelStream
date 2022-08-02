# WARNINGS='-DCMAKE_C_FLAGS=-Wall -Wextra -DCMAKE_CXX_FLAGS=-Wall -Wextra'
WARNINGFLAGS='-Wall -Wextra -fsanitize=address'
WARNINGFLAGS='-Wall -Wextra'
#OFFLOAD=ON    # Beware, "ON" needs to be SHOUTED to make it work!
OFFLOAD=OFF    # Beware, "ON" needs to be SHOUTED to make it work!
COMMAND=cmake
# COMMAND=$HOME/tmp/a.out

#      -DCMAKE_VERBOSE_MAKEFILE=on \

ARGS="-DCMAKE_VERBOSE_MAKEFILE=on \
     -DCMAKE_C_COMPILER=`which clang` \
     -DCMAKE_CXX_COMPILER=`which clang++` \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_C_FLAGS=\"$WARNINGFLAGS\"\
     -DCMAKE_CXX_FLAGS=\"$WARNINGFLAGS\"\
     -DRELEASE_FLAGS=\"-O3 -march=armv8.1-a\"\
     -DMODEL=omp\
     -DOFFLOAD=$OFFLOAD\
     .. "

echo $COMMAND $ARGS
bash -c "$COMMAND $ARGS"
