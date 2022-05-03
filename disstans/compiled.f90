! file compiled.f90
! contains subroutines that speed up some tasks if they are precompiled

! This subroutine is taken from midas.f, downloaded from
! http://geodesy.unr.edu/MIDAS_release.tar on 2021-09-13,
! converted to F90, slightly modified to work better
! with Python/NumPy, and with hardcoded maxn.
! License of the original file:
! Author: Geoff Blewitt.  Copyright (C) 2015.
SUBROUTINE selectpair(t, tstep, tol, m, nstep, n, ip)
    !
    ! Given a time tag array t(m), select pairs ip(2,n)
    !
    ! Moves forward in time: for each time tag, pair it with only
    ! one future time tag.
    ! First attempt to form a pair within tolerance tol of 1 year.
    ! If this fails, then find next unused partner.
    ! If this fails, cycle through all possible future partners again.
    !
    ! MIDAS calls this twice -- firstly forward in time, and
    ! secondly backward in time with negative tags and data.
    ! This ensures a time symmetric solution.
    
    ! 2010-10-12: now allow for apriori list of step epochs
    ! - do not select pairs that span or include the step epoch

    IMPLICIT NONE

    ! constant
    INTEGER, PARAMETER :: maxn = 19999
    
    ! input
    INTEGER, INTENT(IN) :: m, nstep
    REAL, INTENT(IN) :: t(m), tstep(nstep+1), tol
    
    ! output
    INTEGER, INTENT(OUT) :: n, ip(2, maxn)
    
    ! local
    INTEGER :: i, j, k, i2, istep
    REAL :: dt, fdt
     
    k = 0
    n = 0
    istep = 1
    DO i = 1, m
        IF (n >= maxn) EXIT
        IF (t(i) > (t(m) + tol - 1.0)) EXIT
     
        ! scroll through steps until next step time is later than epoch 1
        DO
            IF (istep > nstep) EXIT
            IF (t(i) < tstep(istep) + tol) EXIT
            istep = istep + 1
        ENDDO
        IF (istep <= nstep) THEN
            IF (t(i) > (tstep(istep) + tol - 1.0)) CYCLE
        ENDIF 
 
        DO j = i + 1, m
            IF (k < j) k = j
            IF (istep <= nstep) THEN
                IF (t(j) > (tstep(istep) - tol)) EXIT
            ENDIF
           
            dt = t(j) - t(i)
 
            ! time difference from 1 year
            fdt = (dt - 1.0)
            
            ! keep searching IF pair less than one year
            IF (fdt < -tol) CYCLE

            ! try to find a matching pair within tolerance of 1 year
            IF (fdt < tol) THEN
                i2 = j
            ELSE
            ! otherwise, if greater than 1 year, cycle through remaining data
                i2 = k
                dt = t(i2) - t(i)
                IF (istep <= nstep) THEN
                    IF (t(i2) > (tstep(istep) - tol)) THEN
                        k = 0
                        CYCLE
                    ENDIF
                ENDIF
                IF (k == m) k = 0
                k = k + 1
            ENDIF

            ! data pair has been found
            n = n + 1
            ip(1, n) = i
            ip(2, n) = i2
            EXIT
        ENDDO
    ENDDO

END SUBROUTINE selectpair
