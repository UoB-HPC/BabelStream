module OpenMPWorkshareStream
    use, intrinsic :: ISO_Fortran_env
    use BabelStreamTypes

    implicit none

    character(len=15), parameter :: implementation_name = "OpenMPWorkshare"

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
            !$omp parallel workshare
            A = initA
            B = initB
            C = initC
            !$omp end parallel workshare
        end subroutine init_arrays

        subroutine read_arrays(h_A, h_B, h_C)
            implicit none
            real(kind=REAL64), intent(inout) :: h_A(:), h_B(:), h_C(:)
            !$omp parallel workshare
            h_A = A
            h_B = B
            h_C = C
            !$omp end parallel workshare
        end subroutine read_arrays

        subroutine copy()
            implicit none
            !$omp parallel workshare
            C = A
            !$omp end parallel workshare
        end subroutine copy

        subroutine add()
            implicit none
            !$omp parallel workshare
            C = A + B
            !$omp end parallel workshare
        end subroutine add

        subroutine mul(startScalar)
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            scalar = startScalar
            !$omp parallel workshare
            B = scalar * C
            !$omp end parallel workshare
        end subroutine mul

        subroutine triad(startScalar)
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            scalar = startScalar
            !$omp parallel workshare
            A = B + scalar * C
            !$omp end parallel workshare
        end subroutine triad

        subroutine nstream(startScalar)
            implicit none
            real(kind=REAL64), intent(in) :: startScalar
            real(kind=REAL64) :: scalar
            scalar = startScalar
            !$omp parallel workshare
            A = A + B + scalar * C
            !$omp end parallel workshare
        end subroutine nstream

        function dot() result(s)
            implicit none
            real(kind=REAL64) :: s
            !$omp parallel workshare
            s = dot_product(A,B)
            !$omp end parallel workshare
        end function dot

end module OpenMPWorkshareStream
