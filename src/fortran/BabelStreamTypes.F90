module BabelStreamTypes
    use, intrinsic :: ISO_Fortran_env, only: REAL64,REAL32,INT64,INT32

    implicit none

#ifdef USE_FLOAT
    integer, parameter :: StreamRealKind = REAL32
    character(len=6)   :: StreamRealName = "REAL32"
#else
    integer, parameter :: StreamRealKind = REAL64
    character(len=6)   :: StreamRealName = "REAL64"
#endif

#ifdef USE_INT32
#warning There is no checking for overflowing INT32, so be careful.
    integer, parameter :: StreamIntKind  = INT32
#else
    integer, parameter :: StreamIntKind  = INT64
#endif

end module BabelStreamTypes
