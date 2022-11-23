module OpenMPStream
    use, intrinsic :: ISO_Fortran_env
    use BabelStreamTypes

    implicit none

    character(len=6), parameter :: implementation_name = "OpenMP"

    integer(kind=StreamIntKind) :: N

    real(kind=REAL64), allocatable :: A(:), B(:), C(:)

    contains

        subroutine list_devices()
            implicit none
            write(*,'(a36,a12)') "Listing devices is not supported by ", implementation_name
        end subroutine list_devices

        subroutine set_device(dev)
            implicit none
            integer, intent(in) :: dev
            write(*,'(a32,a12)') "Device != 0 is not supported by ", implementation_name
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
            implicit none
            real(kind=REAL64), intent(in) :: initA, initB, initC
            integer(kind=StreamIntKind) :: i
            !$omp parallel do simd
            do i=1,N
               A(i) = initA
               B(i) = initB
               C(i) = initC
            end do
        end subroutine init_arrays

        subroutine read_arrays(h_A, h_B, h_C)
            implicit none
            real(kind=REAL64), intent(inout) :: h_A(:), h_B(:), h_C(:)
            integer(kind=StreamIntKind) :: i
            !$omp parallel do simd
            do i=1,N
               h_A(i) = A(i)
               h_B(i) = B(i)
               h_C(i) = C(i)
            end do
        end subroutine read_arrays

        subroutine copy()
            implicit none
            integer(kind=StreamIntKind) :: i
            !$omp parallel do simd
            do i=1,N
               C(i) = A(i)
            end do
        end subroutine copy

        subroutine add()
            implicit none
            integer(kind=StreamIntKind) :: i
            !$omp parallel do simd
            do i=1,N
               C(i) = A(i) + B(i)
            end do
        end subroutine add

        subroutine mul(startScalar)
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            integer(kind=StreamIntKind) :: i
            scalar = startScalar
            !$omp parallel do simd
            do i=1,N
               B(i) = scalar * C(i)
            end do
        end subroutine mul

        subroutine triad(startScalar)
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            integer(kind=StreamIntKind) :: i
            scalar = startScalar
            !$omp parallel do simd
            do i=1,N
               A(i) = B(i) + scalar * C(i)
            end do
        end subroutine triad

        subroutine nstream(startScalar)
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            integer(kind=StreamIntKind) :: i
            scalar = startScalar
            !$omp parallel do simd
            do i=1,N
               A(i) = A(i) + B(i) + scalar * C(i)
            end do
        end subroutine nstream

        function dot() result(s)
            implicit none
            real(kind=REAL64) :: s
            integer(kind=StreamIntKind) :: i
            s = real(0,kind=REAL64)
            !$omp parallel do simd reduction(+:s)
            do i=1,N
               s = s + A(i) * B(i)
            end do
        end function dot

end module OpenMPStream
