! file compiled.f90
! contains subroutines that speed up some tasks if they are precompiled

SUBROUTINE maskedmedfilt2d(array, mask_in, m, n, kernel, medians)
    ! filters a 2-d input array columns wise given the kernel size
    ! ignores masked-out values and returns

    IMPLICIT NONE

    ! inputs
    INTEGER, INTENT(IN) :: m, n, kernel
    REAL, INTENT(IN) :: array(m, n)
    LOGICAL, INTENT(IN) :: mask_in(m, n)

    ! outputs
    REAL, INTENT(OUT) :: medians(m, n)

    ! internal variables
    INTEGER :: masksum, i, j, halfwindow
    REAL :: subvec(kernel)

    ! check for odd kernel size
    IF ( MOD(kernel, 2).eq.0 ) STOP "even"

    ! process columns individually
    DO j = 1, n

        ! beginning of the column, need to start with small kernel window
        halfwindow = 0
        DO i = 1, (kernel - 1) / 2
            masksum = COUNT(mask_in(i-halfwindow:i+halfwindow, j))
            IF ( masksum .gt. 0 ) THEN
                ! can proceed with median calculation
                subvec(1:masksum) = PACK(array(i-halfwindow:i+halfwindow, j), &
                                         mask_in(i-halfwindow:i+halfwindow, j))
                CALL median(subvec(1:masksum), masksum, medians(i, j))
            END IF
            halfwindow = halfwindow + 1
        END DO

        ! bug check
        IF ( halfwindow .ne. (kernel - 1) / 2 ) STOP "bug"
        ! center of the column, can use full kernel
        DO i = halfwindow + 1, m - halfwindow
            masksum = COUNT(mask_in(i-halfwindow:i+halfwindow, j))
            IF ( masksum .gt. 0 ) THEN
                ! can proceed with median calculation
                subvec(1:masksum) = PACK(array(i-halfwindow:i+halfwindow, j), &
                                         mask_in(i-halfwindow:i+halfwindow, j))
                CALL median(subvec(1:masksum), masksum, medians(i, j))
            END IF
        END DO

        ! end of column, need to shrink kernel window successively
        DO i = m - halfwindow + 1, m
            halfwindow = halfwindow - 1
            masksum = COUNT(mask_in(i-halfwindow:i+halfwindow, j))
            IF ( masksum .gt. 0 ) THEN
                ! can proceed with median calculation
                subvec(1:masksum) = PACK(array(i-halfwindow:i+halfwindow, j), &
                                         mask_in(i-halfwindow:i+halfwindow, j))
                CALL median(subvec(1:masksum), masksum, medians(i, j))
            END IF
        END DO

    END DO

    CONTAINS

        SUBROUTINE median(x, n, xmed)
            ! from https://jblevins.org/mirror/amiller/median.f90
            ! Find the median of X(1), ... , X(N), using as much of the quicksort
            ! algorithm as is needed to isolate it.
            ! N.B. On exit, the array X is partially ordered.
        
            !     Latest revision - 26 November 1996
            IMPLICIT NONE
        
            INTEGER, INTENT(IN)                :: n
            REAL, INTENT(IN OUT), DIMENSION(:) :: x
            REAL, INTENT(OUT)                  :: xmed
        
            ! Local variables
        
            REAL    :: temp, xhi, xlo, xmax, xmin
            LOGICAL :: odd
            INTEGER :: hi, lo, nby2, nby2p1, mid, i, j, k
        
            nby2 = n / 2
            nby2p1 = nby2 + 1
            odd = .true.
        
            !     HI & LO are position limits encompassing the median.
        
            IF (n == 2 * nby2) odd = .false.
            lo = 1
            hi = n
            IF (n < 3) THEN
                IF (n < 1) THEN
                xmed = 0.0
                RETURN
                END IF
                xmed = x(1)
                IF (n == 1) RETURN
                xmed = 0.5*(xmed + x(2))
                RETURN
            END IF
        
            !     Find median of 1st, middle & last values.
        
            10 mid = (lo + hi)/2
            xmed = x(mid)
            xlo = x(lo)
            xhi = x(hi)
            IF (xhi < xlo) THEN          ! Swap xhi & xlo
                temp = xhi
                xhi = xlo
                xlo = temp
            END IF
            IF (xmed > xhi) THEN
                xmed = xhi
            ELSE IF (xmed < xlo) THEN
                xmed = xlo
            END IF
        
            ! The basic quicksort algorithm to move all values <= the sort key (XMED)
            ! to the left-hand end, and all higher values to the other end.
        
            i = lo
            j = hi
            50 DO
                IF (x(i) >= xmed) EXIT
                i = i + 1
            END DO
            DO
                IF (x(j) <= xmed) EXIT
                j = j - 1
            END DO
            IF (i < j) THEN
                temp = x(i)
                x(i) = x(j)
                x(j) = temp
                i = i + 1
                j = j - 1
        
            !     Decide which half the median is in.
        
                IF (i <= j) GO TO 50
            END IF
        
            IF (.NOT. odd) THEN
                IF (j == nby2 .AND. i == nby2p1) GO TO 130
                IF (j < nby2) lo = i
                IF (i > nby2p1) hi = j
                IF (i /= j) GO TO 100
                IF (i == nby2) lo = nby2
                IF (j == nby2p1) hi = nby2p1
            ELSE
                IF (j < nby2p1) lo = i
                IF (i > nby2p1) hi = j
                IF (i /= j) GO TO 100
        
            ! Test whether median has been isolated.
        
                IF (i == nby2p1) RETURN
            END IF
            100 IF (lo < hi - 1) GO TO 10
        
            IF (.NOT. odd) THEN
                xmed = 0.5*(x(nby2) + x(nby2p1))
                RETURN
            END IF
            temp = x(lo)
            IF (temp > x(hi)) THEN
                x(lo) = x(hi)
                x(hi) = temp
            END IF
            xmed = x(nby2p1)
            RETURN
        
            ! Special case, N even, J = N/2 & I = J + 1, so the median is
            ! between the two halves of the series.   Find max. of the first
            ! half & min. of the second half, then average.
        
            130 xmax = x(1)
            DO k = lo, j
                xmax = MAX(xmax, x(k))
            END DO
            xmin = x(n)
            DO k = i, hi
                xmin = MIN(xmin, x(k))
            END DO
            xmed = 0.5*(xmin + xmax)
        
            RETURN
        END SUBROUTINE median

END SUBROUTINE maskedmedfilt2d

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
