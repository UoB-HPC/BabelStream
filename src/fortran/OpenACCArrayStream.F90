module OpenACCArrayStream
    use, intrinsic :: ISO_Fortran_env
    use BabelStreamTypes

    implicit none

    character(len=12), parameter :: implementation_name = "OpenACCArray"

    integer(kind=StreamIntKind) :: N

    real(kind=REAL64), allocatable :: A(:), B(:), C(:)

    contains

        subroutine list_devices()
            use openacc
            implicit none
            integer :: num
            num = acc_get_num_devices(acc_get_device_type())
            if (num.eq.0) then
              write(*,'(a17)') "No devices found."
            else
              write(*,'(a10,i1,a8)') "There are ",num," devices."
            end if
        end subroutine list_devices

        subroutine set_device(dev)
            use openacc
            implicit none
            integer, intent(in) :: dev
            integer :: num
            num = acc_get_num_devices(acc_get_device_type())
            if (num.eq.0) then
              write(*,'(a17)') "No devices found."
              stop
            else if (dev.gt.num) then
              write(*,'(a21)') "Invalid device index."
              stop
            else
              call acc_set_device_num(dev, acc_get_device_type())
            end if
        end subroutine set_device

        subroutine alloc(array_size)
            implicit none
            integer(kind=StreamIntKind) :: array_size
            integer :: err
            N = array_size
            allocate( A(1:N), B(1:N), C(1:N), stat=err)
#ifndef USE_MANAGED
            !$acc enter data create(A,B,C)
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
            !$acc exit data delete(A,B,C)
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
            !$acc kernels
            A = initA
            B = initB
            C = initC
            !$acc end kernels
        end subroutine init_arrays

        subroutine read_arrays(h_A, h_B, h_C)
            implicit none
            real(kind=REAL64), intent(inout) :: h_A(:), h_B(:), h_C(:)
            !$acc kernels
            h_A = A
            h_B = B
            h_C = C
            !$acc end kernels
        end subroutine read_arrays

        subroutine copy()
            implicit none
            !$acc kernels
            C = A
            !$acc end kernels
        end subroutine copy

        subroutine add()
            implicit none
            !$acc kernels
            C = A + B
            !$acc end kernels
        end subroutine add

        subroutine mul(startScalar)
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            scalar = startScalar
            !$acc kernels
            B = scalar * C
            !$acc end kernels
        end subroutine mul

        subroutine triad(startScalar)
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            scalar = startScalar
            !$acc kernels
            A = B + scalar * C
            !$acc end kernels
        end subroutine triad

        subroutine nstream(startScalar)
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            scalar = startScalar
            !$acc kernels
            A = A + B + scalar * C
            !$acc end kernels
        end subroutine nstream

        function dot() result(s)
            implicit none
            real(kind=REAL64) :: s
            !$acc kernels
            s = dot_product(A,B)
            !$acc end kernels
        end function dot

end module OpenACCArrayStream
