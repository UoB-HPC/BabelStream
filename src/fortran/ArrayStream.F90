module ArrayStream
    use, intrinsic :: ISO_Fortran_env
    use BabelStreamTypes

    implicit none

    character(len=5), parameter :: implementation_name = "Array"

    integer(kind=StreamIntKind) :: N

    real(kind=REAL64), allocatable :: A(:), B(:), C(:)

    contains

        subroutine list_devices()
            implicit none
            integer :: num
            write(*,'(a36,a5)') "Listing devices is not supported by ", implementation_name
        end subroutine list_devices

        subroutine set_device(dev)
            implicit none
            integer, intent(in) :: dev
            write(*,'(a32,a5)') "Device != 0 is not supported by ", implementation_name
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
            A = initA
            B = initB
            C = initC
        end subroutine init_arrays

        subroutine read_arrays(h_A, h_B, h_C)
            implicit none
            real(kind=REAL64), intent(inout) :: h_A(:), h_B(:), h_C(:)
            h_A = A
            h_B = B
            h_C = C
        end subroutine read_arrays

        subroutine copy()
            implicit none
            C = A
        end subroutine copy

        subroutine add()
            implicit none
            C = A + B
        end subroutine add

        subroutine mul(startScalar)
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            scalar = startScalar
            B = scalar * C
        end subroutine mul

        subroutine triad(startScalar)
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            scalar = startScalar
            A = B + scalar * C
        end subroutine triad

        subroutine nstream(startScalar)
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            scalar = startScalar
            A = A + B + scalar * C
        end subroutine nstream

        function dot() result(s)
            implicit none
            real(kind=REAL64) :: s
            s = dot_product(A,B)
        end function dot

end module ArrayStream
