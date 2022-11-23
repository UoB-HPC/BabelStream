module CUDAKernelStream
    use, intrinsic :: ISO_Fortran_env
    use BabelStreamTypes

    implicit none

    character(len=10), parameter :: implementation_name = "CUDAKernel"

    integer(kind=StreamIntKind) :: N

#ifdef USE_MANAGED
    real(kind=REAL64), allocatable, managed :: A(:), B(:), C(:)
#else
    real(kind=REAL64), allocatable, device :: A(:), B(:), C(:)
#endif

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
            use cudafor, only: cudaDeviceSynchronize, cudaGetErrorString
            implicit none
            integer(kind=StreamIntKind) :: i
            integer :: err
            !$cuf kernel do <<< *, * >>>
            do i=1,N
               C(i) = A(i)
            end do
            err = cudaDeviceSynchronize()
            if (err.ne.0) then
              write(*,'(a)') "cudaDeviceSynchronize failed"
              write(*,'(a)') cudaGetErrorString(err)
              stop
            endif
        end subroutine copy

        subroutine add()
            use cudafor, only: cudaDeviceSynchronize, cudaGetErrorString
            implicit none
            integer(kind=StreamIntKind) :: i
            integer :: err
            !$cuf kernel do <<< *, * >>>
            do i=1,N
               C(i) = A(i) + B(i)
            end do
            err = cudaDeviceSynchronize()
            if (err.ne.0) then
              write(*,'(a)') "cudaDeviceSynchronize failed"
              write(*,'(a)') cudaGetErrorString(err)
              stop
            endif
        end subroutine add

        subroutine mul(startScalar)
            use cudafor, only: cudaDeviceSynchronize, cudaGetErrorString
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            integer(kind=StreamIntKind) :: i
            integer :: err
            scalar = startScalar
            !$cuf kernel do <<< *, * >>>
            do i=1,N
               B(i) = scalar * C(i)
            end do
            err = cudaDeviceSynchronize()
            if (err.ne.0) then
              write(*,'(a)') "cudaDeviceSynchronize failed"
              write(*,'(a)') cudaGetErrorString(err)
              stop
            endif
        end subroutine mul

        subroutine triad(startScalar)
            use cudafor, only: cudaDeviceSynchronize, cudaGetErrorString
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            integer(kind=StreamIntKind) :: i
            integer :: err
            scalar = startScalar
            !$cuf kernel do <<< *, * >>>
            do i=1,N
               A(i) = B(i) + scalar * C(i)
            end do
            err = cudaDeviceSynchronize()
            if (err.ne.0) then
              write(*,'(a)') "cudaDeviceSynchronize failed"
              write(*,'(a)') cudaGetErrorString(err)
              stop
            endif
        end subroutine triad

        subroutine nstream(startScalar)
            use cudafor, only: cudaDeviceSynchronize, cudaGetErrorString
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            integer(kind=StreamIntKind) :: i
            integer :: err
            scalar = startScalar
            !$cuf kernel do <<< *, * >>>
            do i=1,N
               A(i) = A(i) + B(i) + scalar * C(i)
            end do
            err = cudaDeviceSynchronize()
            if (err.ne.0) then
              write(*,'(a)') "cudaDeviceSynchronize failed"
              write(*,'(a)') cudaGetErrorString(err)
              stop
            endif
        end subroutine nstream

        function dot() result(r)
            use cudafor, only: cudaDeviceSynchronize, cudaGetErrorString
            implicit none
            real(kind=REAL64) :: r
            integer(kind=StreamIntKind) :: i
            integer :: err
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

end module CUDAKernelStream
