module CUDAFortranKernels
    use, intrinsic :: ISO_Fortran_env
    use BabelStreamTypes

    contains

        attributes(global) subroutine do_copy(n,A,C)
            implicit none
            integer(kind=StreamIntKind), intent(in), value :: n
            real(kind=REAL64), intent(in)  :: A(n)
            real(kind=REAL64), intent(out) :: C(n)
            integer(kind=StreamIntKind) :: i
            i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
            if (i <= N) then
               C(i) = A(i)
            endif
        end subroutine do_copy

        attributes(global) subroutine do_add(n,A,B,C)
            implicit none
            integer(kind=StreamIntKind), intent(in), value :: n
            real(kind=REAL64), intent(in)  :: A(n), B(n)
            real(kind=REAL64), intent(out) :: C(n)
            integer(kind=StreamIntKind) :: i
            i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
            if (i <= N) then
               C(i) = A(i) + B(i)
            endif
        end subroutine do_add

        attributes(global) subroutine do_mul(n,scalar,B,C)
            implicit none
            integer(kind=StreamIntKind), intent(in), value :: n
            real(kind=REAL64), intent(in), value :: scalar
            real(kind=REAL64), intent(out) :: B(n)
            real(kind=REAL64), intent(in)  :: C(n)
            integer(kind=StreamIntKind) :: i
            i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
            if (i <= N) then
               B(i) = scalar * C(i)
            endif
        end subroutine do_mul

        attributes(global) subroutine do_triad(n,scalar,A,B,C)
            implicit none
            integer(kind=StreamIntKind), intent(in), value :: n
            real(kind=REAL64), intent(in), value :: scalar
            real(kind=REAL64), intent(out) :: A(n)
            real(kind=REAL64), intent(in)  :: B(n), C(n)
            integer(kind=StreamIntKind) :: i
            i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
            if (i <= N) then
               A(i) = B(i) + scalar * C(i)
            endif
        end subroutine do_triad

        attributes(global) subroutine do_nstream(n,scalar,A,B,C)
            implicit none
            integer(kind=StreamIntKind), intent(in), value :: n
            real(kind=REAL64), intent(in), value :: scalar
            real(kind=REAL64), intent(inout) :: A(n)
            real(kind=REAL64), intent(in)    :: B(n), C(n)
            integer(kind=StreamIntKind) :: i
            i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
            if (i <= N) then
               A(i) = A(i) + B(i) + scalar * C(i)
            endif
        end subroutine do_nstream

#if 0
        attributes(global) subroutine do_dot(n,A,B,r)
            implicit none
            integer(kind=StreamIntKind), intent(in), value :: n
            real(kind=REAL64), intent(in) :: A(n), B(n) 
            real(kind=REAL64), intent(out) :: r
            integer(kind=StreamIntKind) :: i
            r = real(0,kind=REAL64)
            !$cuf kernel do <<< *, * >>>
            do i=1,N
               r = r + A(i) * B(i)
            end do
        end subroutine do_dot
#endif

end module CUDAFortranKernels

module CUDAStream
    use, intrinsic :: ISO_Fortran_env
    use BabelStreamTypes
    use cudafor, only: dim3

    implicit none

    character(len=4), parameter :: implementation_name = "CUDA"

    integer(kind=StreamIntKind) :: N

#ifdef USE_MANAGED
    real(kind=REAL64), allocatable, managed :: A(:), B(:), C(:)
#else
    real(kind=REAL64), allocatable, device :: A(:), B(:), C(:)
#endif

    type(dim3) :: grid, tblock

    contains

        subroutine list_devices()
            use cudafor
            implicit none
            integer :: num, err
            err = cudaGetDeviceCount(num)
            if (err.ne.0) then
              write(*,'(a)') "cudaGetDeviceCount failed"
              write(*,'(a)') cudaGetErrorString(err)
              stop
            else if (num.eq.0) then
              write(*,'(a17)') "No devices found."
            else
              write(*,'(a10,i1,a8)') "There are ",num," devices."
            end if
        end subroutine list_devices

        subroutine set_device(dev)
            use cudafor
            implicit none
            integer, intent(in) :: dev
            integer :: num, err
            err = cudaGetDeviceCount(num)
            if (err.ne.0) then
              write(*,'(a)') "cudaGetDeviceCount failed"
              write(*,'(a)') cudaGetErrorString(err)
              stop
            else if (num.eq.0) then
              write(*,'(a17)') "No devices found."
              stop
            else if (dev.ge.num) then
              write(*,'(a21)') "Invalid device index."
              stop
            else
              err = cudaSetDevice(dev)
              if (err.ne.0) then
                write(*,'(a)') "cudaSetDevice failed"
                write(*,'(a)') cudaGetErrorString(err)
                stop
              end if
            end if
        end subroutine set_device

        subroutine alloc(array_size)
            implicit none
            integer(kind=StreamIntKind) :: array_size
            integer :: err
            N = array_size
            allocate( A(1:N), B(1:N), C(1:N), stat=err)
            if (err .ne. 0) then
              write(*,'(a20,i3)') 'allocate returned ',err
              stop 1
            endif
            ! move to separate subroutine later
            tblock = dim3(128,1,1)
            grid = dim3(ceiling(real(N)/tblock%x),1,1)
        end subroutine alloc

        subroutine dealloc()
            implicit none
            integer :: err
            deallocate( A, B, C, stat=err)
            if (err .ne. 0) then
              write(*,'(a20,i3)') 'deallocate returned ',err
              stop 1
            endif
        end subroutine dealloc

        subroutine init_arrays(initA, initB, initC)
            use cudafor, only: cudaDeviceSynchronize, cudaGetErrorString
            implicit none
            real(kind=REAL64), intent(in) :: initA, initB, initC
            integer(kind=StreamIntKind) :: i
            integer :: err
            A = initA
            B = initB
            C = initC
            err = cudaDeviceSynchronize()
            if (err.ne.0) then
              write(*,'(a)') "cudaDeviceSynchronize failed"
              write(*,'(a)') cudaGetErrorString(err)
              stop
            endif
        end subroutine init_arrays

        subroutine read_arrays(h_A, h_B, h_C)
            use cudafor, only: cudaDeviceSynchronize, cudaGetErrorString
            implicit none
            real(kind=REAL64), intent(inout) :: h_A(:), h_B(:), h_C(:)
            integer(kind=StreamIntKind) :: i
            integer :: err
            h_A = A
            h_B = B
            h_C = C
            err = cudaDeviceSynchronize()
            if (err.ne.0) then
              write(*,'(a)') "cudaDeviceSynchronize failed"
              write(*,'(a)') cudaGetErrorString(err)
              stop
            endif
        end subroutine read_arrays

        subroutine copy()
            use CUDAFortranKernels, only: do_copy
            use cudafor, only: cudaDeviceSynchronize, cudaGetErrorString
            implicit none
            integer :: err
            call do_copy<<<grid, tblock>>>(N, A, C)
            err = cudaDeviceSynchronize()
            if (err.ne.0) then
              write(*,'(a)') "cudaDeviceSynchronize failed"
              write(*,'(a)') cudaGetErrorString(err)
              stop
            endif
        end subroutine copy

        subroutine add()
            use CUDAFortranKernels, only: do_add
            use cudafor, only: cudaDeviceSynchronize, cudaGetErrorString
            implicit none
            integer :: err
            call do_add<<<grid, tblock>>>(N, A, B, C)
            err = cudaDeviceSynchronize()
            if (err.ne.0) then
              write(*,'(a)') "cudaDeviceSynchronize failed"
              write(*,'(a)') cudaGetErrorString(err)
              stop
            endif
        end subroutine add

        subroutine mul(startScalar)
            use CUDAFortranKernels, only: do_mul
            use cudafor, only: cudaDeviceSynchronize, cudaGetErrorString
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            integer :: err
            scalar = startScalar
            call do_mul<<<grid, tblock>>>(N, scalar, B, C)
            err = cudaDeviceSynchronize()
            if (err.ne.0) then
              write(*,'(a)') "cudaDeviceSynchronize failed"
              write(*,'(a)') cudaGetErrorString(err)
              stop
            endif
        end subroutine mul

        subroutine triad(startScalar)
            use CUDAFortranKernels, only: do_triad
            use cudafor, only: cudaDeviceSynchronize, cudaGetErrorString
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            integer :: err
            scalar = startScalar
            call do_triad<<<grid, tblock>>>(N, scalar, A, B, C)
            err = cudaDeviceSynchronize()
            if (err.ne.0) then
              write(*,'(a)') "cudaDeviceSynchronize failed"
              write(*,'(a)') cudaGetErrorString(err)
              stop
            endif
        end subroutine triad

        subroutine nstream(startScalar)
            use CUDAFortranKernels, only: do_nstream
            use cudafor, only: cudaDeviceSynchronize, cudaGetErrorString
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            integer :: err
            scalar = startScalar
            call do_nstream<<<grid, tblock>>>(N, scalar, A, B, C)
            err = cudaDeviceSynchronize()
            if (err.ne.0) then
              write(*,'(a)') "cudaDeviceSynchronize failed"
              write(*,'(a)') cudaGetErrorString(err)
              stop
            endif
        end subroutine nstream

        function dot() result(r)
            !use CUDAFortranKernels, only: do_dot
            use cudafor, only: cudaDeviceSynchronize, cudaGetErrorString
            implicit none
            real(kind=REAL64) :: r
            integer :: err
            integer(kind=StreamIntKind) :: i
            !call do_dot<<<grid, tblock>>>(N, B, C, r)
            r = real(0,kind=REAL64)
            !$cuf kernel do <<< *, * >>>
            do i=1,N
               r = r + A(i) * B(i)
            end do
            err = cudaDeviceSynchronize()
            if (err.ne.0) then
              write(*,'(a)') "cudaDeviceSynchronize failed"
              write(*,'(a)') cudaGetErrorString(err)
              stop
            endif
        end function dot

end module CUDAStream
