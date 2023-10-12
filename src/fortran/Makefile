ifeq ($(COMPILER),nvhpc)
    include make.inc.nvhpc
else ifeq ($(COMPILER),oneapi)
    include make.inc.oneapi
else ifeq ($(COMPILER),gcc)
    include make.inc.gcc
else ifeq ($(COMPILER),amd)
    include make.inc.amd
else ifeq ($(COMPILER),arm)
    include make.inc.arm
else ifeq ($(COMPILER),cray)
    include make.inc.cray
else ifeq ($(COMPILER),fj)
    include make.inc.fj
else
    $(info Set COMPILER={nvhpc,oneapi,amd,arm,cray,fj,gcc}. Default is gcc.)
    include make.inc.gcc
    COMPILER=gcc
endif

FCFLAGS += -DVERSION_STRING="5.0"
#FCFLAGS += -DUSE_INT32

ifeq ($(IMPLEMENTATION),DoConcurrent)
    FCFLAGS += -DUSE_DOCONCURRENT $(DOCONCURRENT_FLAG)
    IMPLEMENTATION_OBJECT = DoConcurrentStream.o

else ifeq ($(IMPLEMENTATION),Array)
    FCFLAGS += -DUSE_ARRAY $(ARRAY_FLAG)
    IMPLEMENTATION_OBJECT = ArrayStream.o

else ifeq ($(IMPLEMENTATION),OpenMP)
    FCFLAGS += -DUSE_OPENMP $(OPENMP_FLAG)
    IMPLEMENTATION_OBJECT = OpenMPStream.o

else ifeq ($(IMPLEMENTATION),OpenMPWorkshare)
    FCFLAGS += -DUSE_OPENMPWORKSHARE $(OPENMP_FLAG)
    IMPLEMENTATION_OBJECT = OpenMPWorkshareStream.o

else ifeq ($(IMPLEMENTATION),OpenMPTarget)
    FCFLAGS += -DUSE_OPENMPTARGET $(OPENMP_FLAG)
    IMPLEMENTATION_OBJECT = OpenMPTargetStream.o

else ifeq ($(IMPLEMENTATION),OpenMPTargetLoop)
    FCFLAGS += -DUSE_OPENMPTARGETLOOP $(OPENMP_FLAG)
    IMPLEMENTATION_OBJECT = OpenMPTargetLoopStream.o

else ifeq ($(IMPLEMENTATION),OpenMPTaskloop)
    FCFLAGS += -DUSE_OPENMPTASKLOOP $(OPENMP_FLAG)
    IMPLEMENTATION_OBJECT = OpenMPTaskloopStream.o

else ifeq ($(IMPLEMENTATION),OpenACC)
    FCFLAGS += -DUSE_OPENACC $(OPENACC_FLAG)
    IMPLEMENTATION_OBJECT = OpenACCStream.o

else ifeq ($(IMPLEMENTATION),OpenACCArray)
    FCFLAGS += -DUSE_OPENACCARRAY $(OPENACC_FLAG)
    IMPLEMENTATION_OBJECT = OpenACCArrayStream.o

else ifeq ($(IMPLEMENTATION),CUDA)
    FCFLAGS += -DUSE_CUDA $(CUDA_FLAG)
    IMPLEMENTATION_OBJECT = CUDAStream.o

else ifeq ($(IMPLEMENTATION),CUDAKernel)
    FCFLAGS += -DUSE_CUDAKERNEL $(CUDA_FLAG)
    IMPLEMENTATION_OBJECT = CUDAKernelStream.o

else ifeq ($(IMPLEMENTATION),Sequential)
    FCFLAGS += -DUSE_SEQUENTIAL $(SEQUENTIAL_FLAG)
    IMPLEMENTATION_OBJECT = SequentialStream.o

else
    $(info Set IMPLEMENTATION={DoConcurrent,Array,OpenMP,OpenMPWorkshare,OpenMPTarget,OpenMPTargetLoop,OpenMPTaskloop,OpenACC,OpenACCArray,CUDA,CUDAKernel}.)
    FCFLAGS += -DUSE_SEQUENTIAL $(SEQUENTIAL_FLAG)
    IMPLEMENTATION=Sequential
    IMPLEMENTATION_OBJECT = SequentialStream.o

endif

all: BabelStream.$(COMPILER).$(IMPLEMENTATION)

BabelStream.$(COMPILER).$(IMPLEMENTATION): main.F90 $(IMPLEMENTATION_OBJECT)
	$(FC) $(FCFLAGS) $^ BabelStreamTypes.o -o $@

BabelStreamTypes.o BabelStreamTypes.mod: BabelStreamTypes.F90
	$(FC) $(FCFLAGS) -c $<

%.o: %.F90 BabelStreamTypes.mod
	$(FC) $(FCFLAGS) -c $<

clean:
	-rm -f main.o BabelStreamUtil.mod babelstreamutil.mod
	-rm -f BabelStreamTypes.o BabelStreamTypes.mod babelstreamtypes.mod
	-rm -f DoConcurrentStream.o DoConcurrentStream.mod doconcurrentstream.mod
	-rm -f ArrayStream.o ArrayStream.mod arraystream.mod
	-rm -f SequentialStream.o SequentialStream.mod sequentialstream.mod
	-rm -f OpenMPStream.o OpenMPStream.mod openmpstream.mod
	-rm -f OpenMPWorkshareStream.o OpenMPWorkshareStream.mod openmpworksharestream.mod
	-rm -f OpenMPTaskloopStream.o OpenMPTaskloopStream.mod openmptaskloopstream.mod
	-rm -f OpenMPTargetStream.o OpenMPTargetStream.mod openmptargetstream.mod
	-rm -f OpenMPTargetLoopStream.o OpenMPTargetLoopStream.mod openmptargetloopstream.mod
	-rm -f OpenACCStream.o OpenACCStream.mod openaccstream.mod
	-rm -f OpenACCArrayStream.o OpenACCArrayStream.mod openaccarraystream.mod
	-rm -f CUDAStream.o CUDAStream.mod cudastream.mod CUDAFortranKernels.mod cudafortrankernels.mod
	-rm -f CUDAKernelStream.o CUDAKernelStream.mod cudakernelstream.mod
	-rm -f *.modmic *.mod *.o *.cub *.ptx

realclean: clean
	-rm -f BabelStream.*
