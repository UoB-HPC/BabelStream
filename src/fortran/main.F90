module BabelStreamUtil
    use, intrinsic :: ISO_Fortran_env, only: REAL64,INT64
    use BabelStreamTypes

    implicit none

    integer(kind=StreamIntKind) :: array_size = 33554432
    integer(kind=StreamIntKind) :: num_times  = 100
    logical                     :: mibibytes  = .false.
    logical                     :: use_gigs   = .false.
    logical                     :: csv        = .false.
    character(len=1), parameter :: csv_sep    = ","

    ! 1 = All
    ! 2 = Triad
    ! 3 = Nstream
    integer                     :: selection  = 1

    real(kind=REAL64), parameter :: startA      = real(0.1d0,kind=REAL64)
    real(kind=REAL64), parameter :: startB      = real(0.2d0,kind=REAL64)
    real(kind=REAL64), parameter :: startC      = real(0.0d0,kind=REAL64)
    real(kind=REAL64), parameter :: startScalar = real(0.4d0,kind=REAL64)

    contains

        function get_wtime() result(t)
#if defined(USE_OMP_GET_WTIME)
          use omp_lib
          implicit none
          real(kind=REAL64) :: t
          t = omp_get_wtime()
#elif  defined(USE_CPU_TIME)
          implicit none
          real(kind=REAL64) :: t
          real :: r
          call cpu_time(r)
          t = r
#else
          implicit none
          real(kind=REAL64) :: t
          integer(kind=INT64) :: c, r
          call system_clock(count = c, count_rate = r)
          t = real(c,REAL64) / real(r,REAL64)
#endif
        end function get_wtime

        subroutine parseArguments()
            use, intrinsic :: ISO_Fortran_env, only: compiler_version, compiler_options
#if defined(USE_DOCONCURRENT)
            use DoConcurrentStream, only: list_devices, set_device
#elif defined(USE_ARRAY)
            use ArrayStream, only: list_devices, set_device
#elif defined(USE_OPENMP)
            use OpenMPStream, only: list_devices, set_device
#elif defined(USE_OPENMPWORKSHARE)
            use OpenMPWorkshareStream, only: list_devices, set_device
#elif defined(USE_OPENMPTARGET)
            use OpenMPTargetStream, only: list_devices, set_device
#elif defined(USE_OPENMPTARGETLOOP)
            use OpenMPTargetLoopStream, only: list_devices, set_device
#elif defined(USE_OPENMPTASKLOOP)
            use OpenMPTaskloopStream, only: list_devices, set_device
#elif defined(USE_OPENACC)
            use OpenACCStream, only: list_devices, set_device
#elif defined(USE_OPENACCARRAY)
            use OpenACCArrayStream, only: list_devices, set_device
#elif defined(USE_CUDA)
            use CUDAStream, only: list_devices, set_device
#elif defined(USE_CUDAKERNEL)
            use CUDAKernelStream, only: list_devices, set_device
#elif defined(USE_SEQUENTIAL)
            use SequentialStream, only: list_devices, set_device
#endif
            implicit none
            integer :: i, argc
            integer :: arglen,err,pos(2)
            character(len=64) :: argtmp
            argc = command_argument_count()
            do i=1,argc
                call get_command_argument(i,argtmp,arglen,err)
                if (err.eq.0) then
                    !
                    ! list devices
                    !
                    pos(1) = index(argtmp,"--list")
                    if (pos(1).eq.1) then
                        call list_devices()
                        stop
                    endif
                    !
                    ! set device number
                    !
                    pos(1) = index(argtmp,"--device")
                    if (pos(1).eq.1) then
                        if (i+1.gt.argc) then
                            print*,'You failed to provide a value for ',argtmp
                            stop
                        else
                            call get_command_argument(i+1,argtmp,arglen,err)
                            block
                                integer :: dev
                                read(argtmp,'(i15)') dev
                                call set_device(dev)
                            end block
                        endif
                        cycle
                    endif
                    !
                    ! array size
                    !
                    pos(1) = index(argtmp,"--arraysize")
                    pos(2) = index(argtmp,"-s")
                    if (any(pos(:).eq.1) ) then
                        if (i+1.gt.argc) then
                            print*,'You failed to provide a value for ',argtmp
                        else
                            call get_command_argument(i+1,argtmp,arglen,err)
                            block
                              integer(kind=INT64) :: big_size
                              read(argtmp,'(i15)') big_size
                              if (big_size .gt. HUGE(array_size)) then
                                print*,'Array size does not fit into integer:'
                                print*,big_size,'>',HUGE(array_size)
                                print*,'Stop using USE_INT32'
                                stop
                              else
                                array_size = INT(big_size,kind=StreamIntKind)
                              endif
                            end block
                        endif
                        cycle
                    endif
                    !
                    ! number of iterations
                    !
                    pos(1) = index(argtmp,"--numtimes")
                    pos(2) = index(argtmp,"-n")
                    if (any(pos(:).eq.1) ) then
                        if (i+1.gt.argc) then
                            print*,'You failed to provide a value for ',argtmp
                        else
                            call get_command_argument(i+1,argtmp,arglen,err)
                            read(argtmp,'(i15)') num_times
                            if (num_times.lt.2) then
                                write(*,'(a)') "Number of times must be 2 or more"
                                stop
                            end if
                        endif
                        cycle
                    endif
                    !
                    ! precision
                    !
                    pos(1) = index(argtmp,"--float")
                    if (pos(1).eq.1) then
                        write(*,'(a46,a39)') "Sorry, you have to recompile with -DUSE_FLOAT ", &
                                             "to run BabelStream in single precision."
                        stop
                    endif
                    !
                    ! selection (All, Triad, Nstream)
                    !
                    pos(1) = index(argtmp,"--triad-only")
                    if (pos(1).eq.1) then
                        selection = 2
                        cycle
                    endif
                    pos(1) = index(argtmp,"--nstream-only")
                    if (pos(1).eq.1) then
                        selection = 3
                        cycle
                    endif
                    !
                    ! CSV
                    !
                    pos(1) = index(argtmp,"--csv")
                    if (pos(1).eq.1) then
                        csv = .true.
                        !write(*,'(a39)') "Sorry, CSV support isn't available yet."
                        !stop
                    endif
                    !
                    ! units
                    !
                    pos(1) = index(argtmp,"--mibibytes")
                    if (pos(1).eq.1) then
                        mibibytes = .true.
                        cycle
                    endif
                    !
                    ! giga/gibi instead of mega/mebi
                    !
                    pos(1) = index(argtmp,"--gigs")
                    if (pos(1).eq.1) then
                        use_gigs = .true.
                        cycle
                    endif
                    !
                    !
                    !
                    pos(1) = index(argtmp,"--compiler-info")
                    if (pos(1).eq.1) then
                        write(*,'(a)') 'Compiler version: ',compiler_version()
                        write(*,'(a)') 'Compiler options: ',compiler_options()
                        stop
                    endif
                    !
                    ! help
                    !
                    pos(1) = index(argtmp,"--help")
                    pos(2) = index(argtmp,"-h")
                    if (any(pos(:).eq.1) ) then
                        call get_command_argument(0,argtmp,arglen,err)
                        write(*,'(a7,a,a10)') "Usage: ", trim(argtmp), " [OPTIONS]"
                        write(*,'(a)') "Options:"
                        write(*,'(a)') "  -h  --help               Print the message"
                        write(*,'(a)') "      --list               List available devices"
                        write(*,'(a)') "      --device     INDEX   Select device at INDEX"
                        write(*,'(a)') "  -s  --arraysize  SIZE    Use SIZE elements in the array"
                        write(*,'(a)') "  -n  --numtimes   NUM     Run the test NUM times (NUM >= 2)"
                        !write(*,'(a)') "      --float              Use floats (rather than doubles)"
                        write(*,'(a)') "      --triad-only         Only run triad"
                        write(*,'(a)') "      --nstream-only       Only run nstream"
                        write(*,'(a)') "      --csv                Output as csv table"
                        write(*,'(a)') "      --mibibytes          Use MiB=2^20 for bandwidth calculation (default MB=10^6)"
                        write(*,'(a)') "      --gigs               Use GiB=2^30 or GB=10^9 instead of MiB/MB"
                        write(*,'(a)') "      --compiler-info      Print information about compiler and flags, then exit."
                        stop
                    endif
                end if
            end do
        end subroutine parseArguments

        subroutine run_all(timings, summ)
#if defined(USE_DOCONCURRENT)
            use DoConcurrentStream
#elif defined(USE_ARRAY)
            use ArrayStream
#elif defined(USE_OPENMP)
            use OpenMPStream
#elif defined(USE_OPENMPWORKSHARE)
            use OpenMPWorkshareStream
#elif defined(USE_OPENMPTARGET)
            use OpenMPTargetStream
#elif defined(USE_OPENMPTARGETLOOP)
            use OpenMPTargetLoopStream
#elif defined(USE_OPENMPTASKLOOP)
            use OpenMPTaskloopStream
#elif defined(USE_OPENACC)
            use OpenACCStream
#elif defined(USE_OPENACCARRAY)
            use OpenACCArrayStream
#elif defined(USE_CUDA)
            use CUDAStream
#elif defined(USE_CUDAKERNEL)
            use CUDAKernelStream
#elif defined(USE_SEQUENTIAL)
            use SequentialStream
#endif
            implicit none
            real(kind=REAL64), intent(inout) :: timings(:,:)
            real(kind=REAL64), intent(out) :: summ
            real(kind=REAL64) :: t1, t2
            integer(kind=StreamIntKind) :: i

            do i=1,num_times

                t1 = get_wtime()
                call copy()
                t2 = get_wtime()
                timings(1,i) = t2-t1

                t1 = get_wtime()
                call mul(startScalar)
                t2 = get_wtime()
                timings(2,i) = t2-t1

                t1 = get_wtime()
                call add()
                t2 = get_wtime()
                timings(3,i) = t2-t1

                t1 = get_wtime()
                call triad(startScalar)
                t2 = get_wtime()
                timings(4,i) = t2-t1

                t1 = get_wtime()
                summ = dot()
                t2 = get_wtime()
                timings(5,i) = t2-t1

            end do

        end subroutine run_all

        subroutine run_triad(timings)
#if defined(USE_DOCONCURRENT)
            use DoConcurrentStream
#elif defined(USE_ARRAY)
            use ArrayStream
#elif defined(USE_OPENMP)
            use OpenMPStream
#elif defined(USE_OPENMPWORKSHARE)
            use OpenMPWorkshareStream
#elif defined(USE_OPENMPTARGET)
            use OpenMPTargetStream
#elif defined(USE_OPENMPTARGETLOOP)
            use OpenMPTargetLoopStream
#elif defined(USE_OPENMPTASKLOOP)
            use OpenMPTaskloopStream
#elif defined(USE_OPENACC)
            use OpenACCStream
#elif defined(USE_OPENACCARRAY)
            use OpenACCArrayStream
#elif defined(USE_CUDA)
            use CUDAStream
#elif defined(USE_CUDAKERNEL)
            use CUDAKernelStream
#elif defined(USE_SEQUENTIAL)
            use SequentialStream
#endif
            implicit none
            real(kind=REAL64), intent(inout) :: timings(:,:)
            real(kind=REAL64) :: t1, t2
            integer(kind=StreamIntKind) :: i

            do i=1,num_times

                t1 = get_wtime()
                call triad(startScalar)
                t2 = get_wtime()
                timings(1,i) = t2-t1

            end do

        end subroutine run_triad

        subroutine run_nstream(timings)
#if defined(USE_DOCONCURRENT)
            use DoConcurrentStream
#elif defined(USE_ARRAY)
            use ArrayStream
#elif defined(USE_OPENMP)
            use OpenMPStream
#elif defined(USE_OPENMPWORKSHARE)
            use OpenMPWorkshareStream
#elif defined(USE_OPENMPTARGET)
            use OpenMPTargetStream
#elif defined(USE_OPENMPTARGETLOOP)
            use OpenMPTargetLoopStream
#elif defined(USE_OPENMPTASKLOOP)
            use OpenMPTaskloopStream
#elif defined(USE_OPENACC)
            use OpenACCStream
#elif defined(USE_OPENACCARRAY)
            use OpenACCArrayStream
#elif defined(USE_CUDA)
            use CUDAStream
#elif defined(USE_CUDAKERNEL)
            use CUDAKernelStream
#elif defined(USE_SEQUENTIAL)
            use SequentialStream
#endif
            implicit none
            real(kind=REAL64), intent(inout) :: timings(:,:)
            real(kind=REAL64) :: t1, t2
            integer(kind=StreamIntKind) :: i

            do i=1,num_times

                t1 = get_wtime()
                call nstream(startScalar)
                t2 = get_wtime()
                timings(1,i) = t2-t1

            end do

        end subroutine run_nstream

        subroutine check_solution(A, B, C, summ)
            use, intrinsic :: IEEE_Arithmetic, only: IEEE_Is_Normal
            implicit none
            real(kind=REAL64), intent(in) :: A(:), B(:), C(:)
            real(kind=REAL64), intent(in) :: summ

            integer(kind=StreamIntKind) :: i
            real(kind=REAL64) :: goldA, goldB, goldC, goldSum
            real(kind=REAL64) :: scalar

            ! always use double because of accumulation error
            real(kind=REAL64) :: errA, errB, errC, errSum, epsi
            logical :: cleanA, cleanB, cleanC, cleanSum

            goldA = startA
            goldB = startB
            goldC = startC
            goldSum = 0.0d0

            scalar = startScalar

            do i=1,num_times

                if (selection.eq.1) then
                    goldC = goldA
                    goldB = scalar * goldC
                    goldC = goldA + goldB
                    goldA = goldB + scalar * goldC
                else if (selection.eq.2) then
                    goldA = goldB + scalar * goldC
                else if (selection.eq.3) then
                    goldA = goldA + goldB + scalar * goldC;
                endif

            end do

            goldSum = goldA * goldB * array_size

            cleanA = ALL(IEEE_Is_Normal(A))
            cleanB = ALL(IEEE_Is_Normal(B))
            cleanC = ALL(IEEE_Is_Normal(C))
            cleanSum = IEEE_Is_Normal(summ)

            if (.not. cleanA) then
                write(*,'(a51)') "Validation failed on A. Contains NaA/Inf/Subnormal."
            end if
            if (.not. cleanB) then
                write(*,'(a51)') "Validation failed on B. Contains NaA/Inf/Subnormal."
            end if
            if (.not. cleanC) then
                write(*,'(a51)') "Validation failed on C. Contains NaA/Inf/Subnormal."
            end if
            if (.not. cleanSum) then
                write(*,'(a54,e20.12)') "Validation failed on Sum. Contains NaA/Inf/Subnormal: ",summ
            end if

            errA = SUM( ABS( A - goldA ) ) / array_size
            errB = SUM( ABS( B - goldB ) ) / array_size
            errC = SUM( ABS( C - goldC ) ) / array_size
            errSum = ABS( (summ - goldSum) /  goldSum)

            epsi = epsilon(real(0,kind=StreamRealKind)) * 100.0d0

            if (errA .gt. epsi) then
                write(*,'(a38,e20.12)') "Validation failed on A. Average error ", errA
            end if
            if (errB .gt. epsi) then
                write(*,'(a38,e20.12)') "Validation failed on B. Average error ", errB
            end if
            if (errC .gt. epsi) then
                write(*,'(a38,e20.12)') "Validation failed on C. Average error ", errC
            end if

            if (selection.eq.1) then
                if (errSum .gt. 1.0e-8) then
                    write(*,'(a38,e20.12)') "Validation failed on Sum. Error ", errSum
                    write(*,'(a8,e20.12,a15,e20.12)') "Sum was ",summ, " but should be ", errSum
                end if
            endif

        end subroutine check_solution

end module BabelStreamUtil

program BabelStream
    use BabelStreamUtil
#if defined(USE_DOCONCURRENT)
    use DoConcurrentStream
#elif defined(USE_ARRAY)
    use ArrayStream
#elif defined(USE_OPENMP)
    use OpenMPStream
#elif defined(USE_OPENMPWORKSHARE)
    use OpenMPWorkshareStream
#elif defined(USE_OPENMPTARGET)
    use OpenMPTargetStream
#elif defined(USE_OPENMPTARGETLOOP)
    use OpenMPTargetLoopStream
#elif defined(USE_OPENMPTASKLOOP)
    use OpenMPTaskloopStream
#elif defined(USE_OPENACC)
    use OpenACCStream
#elif defined(USE_OPENACCARRAY)
    use OpenACCArrayStream
#elif defined(USE_CUDA)
    use CUDAStream
#elif defined(USE_CUDAKERNEL)
    use CUDAKernelStream
#elif defined(USE_SEQUENTIAL)
    use SequentialStream
#endif
    implicit none
    integer :: element_size, err
    real(kind=REAL64) :: scaling
    character(len=3) :: label
    real(kind=REAL64), allocatable :: timings(:,:)
    real(kind=REAL64), allocatable :: h_A(:), h_B(:), h_C(:)
    real(kind=REAL64) :: summ

    call parseArguments()

    element_size = storage_size(real(0,kind=StreamRealKind)) / 8

    if (mibibytes) then
      if (use_gigs) then
        scaling = 2.0d0**(-30)
        label   = "GiB"
      else
        scaling = 2.0d0**(-20)
        label   = "MiB"
      endif
    else
      if (use_gigs) then
        scaling = 1.0d-9
        label   = "GB"
      else
        scaling = 1.0d-6
        label   = "MB"
      endif
    endif

    if (.not.csv) then

      write(*,'(a)')        "BabelStream Fortran"
      write(*,'(a9,f4.1)')  "Version: ", VERSION_STRING
      write(*,'(a16,a)')    "Implementation: ", implementation_name

      block
        character(len=32) :: printout
        write(printout,'(i9,1x,a5)') num_times,'times'
        write(*,'(a16,a)') 'Running kernels ',ADJUSTL(printout)
      end block
      write(*,'(a11,a6)') 'Precision: ',ADJUSTL(StreamRealName)

      write(*,'(a12,f9.1,a3)') 'Array size: ',1.0d0 * element_size * (array_size * scaling), label
      write(*,'(a12,f9.1,a3)') 'Total size: ',3.0d0 * element_size * (array_size * scaling), label

    endif ! csv

    allocate( timings(5,num_times) )

    call alloc(array_size)

    call init_arrays(startA, startB, startC)
    summ = 0.0d0

    timings = -1.0d0
    if (selection.eq.1) then
        call run_all(timings, summ)
    else if (selection.eq.2) then
        call run_triad(timings)
    else if (selection.eq.3) then
        call run_nstream(timings)
    endif

    allocate( h_A(1:array_size), h_B(1:array_size), h_C(1:array_size), stat=err)
    if (err .ne. 0) then
      write(*,'(a20,i3)') 'allocate returned ',err
      stop 1
    endif

    call read_arrays(h_A, h_B, h_C)
    call check_solution(h_A, h_B, h_C, summ)

    block
      character(len=20) :: printout(8)
      real(kind=REAL64) :: tmin,tmax,tavg,nbytes
      
      if (csv) then
        write(*,'(a,a1)',advance='no')  'function',  csv_sep
        write(*,'(a,a1)',advance='no')  'num_times', csv_sep
        write(*,'(a,a1)',advance='no')  'n_elements',csv_sep
        write(*,'(a,a1)',advance='no')  'sizeof',    csv_sep
        if (mibibytes) then
          write(*,'(a,a1)',advance='no')  'max_mibytes_per_sec',csv_sep
        else
          write(*,'(a,a1)',advance='no')  'max_mbytes_per_sec', csv_sep
        endif
        write(*,'(a,a1)',advance='no')  'min_runtime',csv_sep
        write(*,'(a,a1)',advance='no')  'max_runtime',csv_sep
        write(*,'(a,a1)',advance='yes') 'avg_runtime'
      else
        write(printout(1),'(a8)')   'Function'
        write(printout(2),'(a3,a8)') TRIM(label),'ytes/sec'
        write(printout(3),'(a9)')   'Min (sec)'
        write(printout(4),'(a3)')   'Max'
        write(printout(5),'(a7)')   'Average'
        write(*,'(5a12)') ADJUSTL(printout(1:5))
      endif ! csv
    
      if (selection.eq.1) then
        block
          integer, parameter :: sizes(5) = [2,2,3,3,2]
          character(len=5), parameter :: labels(5) = ["Copy ", "Mul  ", "Add  ", "Triad", "Dot  "]
          integer :: i
          do i=1,5
            tmin = MINVAL(timings(i,2:num_times))
            tmax = MAXVAL(timings(i,2:num_times))
            tavg = SUM(timings(i,2:num_times)) / (num_times-1)
            nbytes = element_size * REAL(array_size,kind=REAL64) * sizes(i)
            write(printout(1),'(a)')     labels(i)
            if (csv) then
              write(printout(2),'(i20)')   num_times
              write(printout(3),'(i20)')   array_size
              write(printout(4),'(i20)')   element_size
              write(printout(5),'(i20)')   INT(scaling*nbytes/tmin)
              write(printout(6),'(f20.8)') tmin
              write(printout(7),'(f20.8)') tmax
              write(printout(8),'(f20.8)') tavg
              write(*,'(a,a1)',advance='no')  TRIM(ADJUSTL(printout(1))),csv_sep
              write(*,'(a,a1)',advance='no')  TRIM(ADJUSTL(printout(2))),csv_sep
              write(*,'(a,a1)',advance='no')  TRIM(ADJUSTL(printout(3))),csv_sep
              write(*,'(a,a1)',advance='no')  TRIM(ADJUSTL(printout(4))),csv_sep
              write(*,'(a,a1)',advance='no')  TRIM(ADJUSTL(printout(5))),csv_sep
              write(*,'(a,a1)',advance='no')  TRIM(ADJUSTL(printout(6))),csv_sep
              write(*,'(a,a1)',advance='no')  TRIM(ADJUSTL(printout(7))),csv_sep
              write(*,'(a,a1)',advance='yes') TRIM(ADJUSTL(printout(8)))
            else
              write(printout(2),'(f12.3)') scaling*nbytes/tmin
              write(printout(3),'(f12.5)') tmin
              write(printout(4),'(f12.5)') tmax
              write(printout(5),'(f12.5)') tavg
              write(*,'(5a12)') ADJUSTL(printout(1:5))
            endif
          enddo
        end block
      else if ((selection.eq.2).or.(selection.eq.3)) then
        tmin = MINVAL(timings(1,2:num_times))
        tmax = MAXVAL(timings(1,2:num_times))
        tavg = SUM(timings(1,2:num_times)) / (num_times-1)
        if (selection.eq.2) then
          nbytes = element_size * REAL(array_size,kind=REAL64) * 3
          write(printout(1),'(a12)')   "Triad"
        else if (selection.eq.3) then
          nbytes = element_size * REAL(array_size,kind=REAL64) * 4
          write(printout(1),'(a12)')   "Nstream"
        endif
        if (csv) then
          write(printout(2),'(i20)')   num_times
          write(printout(3),'(i20)')   array_size
          write(printout(4),'(i20)')   element_size
          write(printout(5),'(i20)')   INT(scaling*nbytes/tmin)
          write(printout(6),'(f20.8)') tmin
          write(printout(7),'(f20.8)') tmax
          write(printout(8),'(f20.8)') tavg
          write(*,'(a,a1)',advance='no')  TRIM(ADJUSTL(printout(1))),csv_sep
          write(*,'(a,a1)',advance='no')  TRIM(ADJUSTL(printout(2))),csv_sep
          write(*,'(a,a1)',advance='no')  TRIM(ADJUSTL(printout(3))),csv_sep
          write(*,'(a,a1)',advance='no')  TRIM(ADJUSTL(printout(4))),csv_sep
          write(*,'(a,a1)',advance='no')  TRIM(ADJUSTL(printout(5))),csv_sep
          write(*,'(a,a1)',advance='no')  TRIM(ADJUSTL(printout(6))),csv_sep
          write(*,'(a,a1)',advance='no')  TRIM(ADJUSTL(printout(7))),csv_sep
          write(*,'(a,a1)',advance='yes') TRIM(ADJUSTL(printout(8)))
        else
          write(printout(2),'(f12.3)') scaling*nbytes/tmin
          write(printout(3),'(f12.5)') tmin
          write(printout(4),'(f12.5)') tmax
          write(printout(5),'(f12.5)') tavg
          write(*,'(5a12)') ADJUSTL(printout(1:5))
        endif
      endif
    end block

    call dealloc()

end program BabelStream
