module OpenMPTargetLoopStream
    use, intrinsic :: ISO_Fortran_env
    use BabelStreamTypes

    implicit none

    character(len=16), parameter :: implementation_name = "OpenMPTargetLoop"

    integer(kind=StreamIntKind) :: N

    real(kind=REAL64), allocatable :: A(:), B(:), C(:)

    contains

        subroutine list_devices()
            use omp_lib
            implicit none
            integer :: num
            num = omp_get_num_devices()
            if (num.eq.0) then
              write(*,'(a17)') "No devices found."
            else
              write(*,'(a10,i1,a8)') "There are ",num," devices."
            end if
        end subroutine list_devices

        subroutine set_device(dev)
            use omp_lib
            implicit none
            integer, intent(in) :: dev
            integer :: num
            num = omp_get_num_devices()
            if (num.eq.0) then
              write(*,'(a17)') "No devices found."
              stop
            else if (dev.gt.num) then
              write(*,'(a21)') "Invalid device index."
              stop
            else
              call omp_set_default_device(dev)
            end if
        end subroutine set_device

        subroutine alloc(array_size)
            implicit none
            integer(kind=StreamIntKind) :: array_size
            integer :: err
            N = array_size
            allocate( A(1:N), B(1:N), C(1:N), stat=err)
#ifndef USE_MANAGED
            !$omp target enter data map(alloc: A,B,C)
#endif
            if (err .ne. 0) then
              write(*,'(a20,i3)') 'allocate returned ',err
              stop 1
            endif
        end subroutine alloc

        subroutine dealloc()
            implicit none
            integer :: err
#ifndef USE_MANAGED
            !$omp target exit data map(delete: A,B,C)
#endif
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
            !$omp target teams loop
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
            ! this might need to use a copy API instead...
            !$omp target teams loop
            do i=1,N
               h_A(i) = A(i)
               h_B(i) = B(i)
               h_C(i) = C(i)
            end do
        end subroutine read_arrays

        subroutine copy()
            implicit none
            integer(kind=StreamIntKind) :: i
            !$omp target teams loop
            do i=1,N
               C(i) = A(i)
            end do
        end subroutine copy

        subroutine add()
            implicit none
            integer(kind=StreamIntKind) :: i
            !$omp target teams loop
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
            !$omp target teams loop
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
            !$omp target teams loop
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
            !$omp target teams loop
            do i=1,N
               A(i) = A(i) + B(i) + scalar * C(i)
            end do
        end subroutine nstream

        function dot() result(s)
            implicit none
            real(kind=REAL64) :: s
            integer(kind=StreamIntKind) :: i
            s = real(0,kind=REAL64)
            !$omp target teams loop reduction(+:s)
            do i=1,N
               s = s + A(i) * B(i)
            end do
        end function dot

end module OpenMPTargetLoopStream
