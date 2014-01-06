	REAL DATA(PYTHON_NUMBER_OF_STARS,100)
        REAL ARRAY1(100,100), ARRAY2(100,100), VECT1(100), VECT2(100)
        real vect3(PYTHON_NUMBER_OF_STARS), vect4(PYTHON_NUMBER_OF_OBJECTS)
C
	OPEN(UNIT=21,STATUS='OLD',FILE='pcat_input.txt')
C
	N = PYTHON_NUMBER_OF_STARS
	M = 100
	DO I = 1, N
           read(21,*)(data(i,j),j=1,m)
c          READ(21,100)(DATA(I,J),J=1,M)
  100	   FORMAT(8F7.1)
 	ENDDO
C	CALL OUTMAT(N,M,DATA)
C
	METHOD = 3
	IPRINT = 3
	CALL PCA(N,M,DATA,METHOD,IPRINT,ARRAY1,VECT1,VECT2,
     X              vect3,vect4,ARRAY2,IERR)
             
	IF (IERR.NE.0) GOTO 9000
C
	GOTO 9900
 9000	WRITE (6,*) ' ABNORMAL END: IERR =', IERR
 9900	CONTINUE
	END

C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C                                  
C  Carry out a PRINCIPAL COMPONENTS ANALYSIS
C              (KARHUNEN-LOEVE EXPANSION).  
C                                           
C  To call: CALL PCA(N,M,DATA,METHOD,IPRINT,A1,W1,W2,W3,W4,A2,IERR) 
C           where
C                                
C                                
C  N, M  : integer dimensions of ...
C  DATA  : input data.              
C          On output, DATA contains in first 7 columns the 
C          projections of the row-points on the first 7 
C          principal components. 
C  METHOD: analysis option.                 
C          = 1: on sums of squares & cross products matrix. 
C               (I.e. no transforming of original data.)
C          = 2: on covariance matrix.    
C               (I.e. original data centered to zero mean.)
C          = 3: on correlation matrix.   
C               (I.e. original data centered, and reduced to unit std.dev.)
C          = 4: on covariance matrix of range-normalized data.
C               (I.e. original data normalized by div. by ranges; then cent'd.)
C          = 5: on Kenkall rank-order correlation matrix.
C          = 6: on Spearman rank-order correlation matrix.
C  IPRINT: print options.                
C          = 0: no printed output- arrays/vectors, only, contain 
C               items calculated.                 
C          = 1: eigenvalues, only, output. 
C          = 2: printed output, in addition, of correlation (or 
C               other) matrix, eigenvalues and eigenvectors.
C          = 3: full printing of items calculated.   
C  A1    : correlation, covariance or sums of squares & 
C          cross-products matrix, dimensions M * M.            
C          On output, A1 contains in the first 7 columns the 
C          projections of the column-points on the first 7 
C          principal components. 
C  W1,W2 : real vectors of dimension M (see called routines for 
C          use).
C          On output, W1 contains the cumulative percentage 
C          variances associated with the principal components.        
C          Correction Jan. '90: on output, W1 contains the eigenvalues
C          (read with index M-J+1, where running index J is J=1,7)
C  W3,W4 : real vectors of dimension N (used only by Spearman corr. routine)
C  A2    : real array of dimensions M * M (see routines for use).   
C  IERR  : error indicator (normally zero).            
C                                                      
C                                                      
C  Inputs here are N, M, DATA, METHOD, IPRINT (and IERR).
C  Output information is contained in DATA, A1, and W1.  
C  All printed outputs are carried out in easily recognizable sub- 
C  routines called from the first subroutine following.    
C                                                          
C  If IERR > 0, then its value indicates the eigenvalue for which
C  no convergence was obtained.      
C                                 
C  F. Murtagh, ST-ECF/ESA/ESO, Garching-bei-Muenchen, January 1986. 
C
C  HISTORY
C
C  Initial coding, testing                             [F.M., Jan. 1986]
C  Bug fix - col. projs. overwritten by cum. % var.    [F.M., Jan. 1990]
C  Subr. names altered thoughout, to begin with P and to be <= 6 chars. 
C    long, to avoid conflicts with other similarly named subr. in other
C    prgs. such as corresp. anal. or mult. disc. anal. [F.M., Aug. 1990] 
C  Addition of OPTION=4, and subr. PRANCV              [F.M., May 1991]   
C  Addition of OPTIONs 5 and 6, and assoc. subrs.      [F.M., May 1991]
C                             
C-------------------------------------------------------------------------
        SUBROUTINE PCA(N,M,DATA,METHOD,IPRINT,A,W,FV1,W3,W4,Z,IERR)
        REAL    DATA(N,M), A(M,M), W(M), FV1(M), Z(M,M), W3(N), W4(N)
C
        IF (METHOD.EQ.1) GOTO 100
        IF (METHOD.EQ.2) GOTO 200
        IF (METHOD.EQ.4) GOTO 400
        IF (METHOD.EQ.5) GOTO 500
        IF (METHOD.EQ.6) GOTO 600
C       If method.eq.3 or otherwise ...
        GOTO 300
C
C---    Form sums of squares and cross-products matrix.
  100   CONTINUE
        CALL PSCPCL(N,M,DATA,A)
        IF (IPRINT.GT.1) CALL POUTHM(METHOD,M,A)
C       Now do the PCA.
        GOTO 1000
C
C---    Form covariance matrix.
  200   CONTINUE
        CALL PCOVCL(N,M,DATA,W,A)
        IF (IPRINT.GT.1) CALL POUTHM(METHOD,M,A)
C       Now do the PCA.
        GOTO 1000
C
C---    Construct correlation matrix.
  300   CONTINUE
        CALL PCORCL(N,M,DATA,W,FV1,A)
        IF (IPRINT.GT.1) CALL POUTHM(METHOD,M,A)
C       Now do the PCA.
        GOTO 1000
C
C---    Normalize by dividing by ranges; then use covariances.
  400   CONTINUE
        CALL PRANCV(N,M,DATA,W,FV1,A) 
        IF (IPRINT.GT.1) CALL POUTHM(METHOD,M,A)
C       Now do the PCA.
        GOTO 1000
C
C---    Construct Kendall rank-order correlation matrix.
  500   CONTINUE
        CALL PKEND(N,M,DATA,A)
        IF (IPRINT.GT.1) CALL POUTHM(METHOD,M,A)
C       Now do the PCA.
        GOTO 1000
C
C---    Construct Spearman rank-order correlation matrix.
  600   CONTINUE
        CALL PSPEAR(N,M,DATA,W3,W4,A)
        IF (IPRINT.GT.1) CALL POUTHM(METHOD,M,A)
C       Now do the PCA.
        GOTO 1000
C
C---    Carry out eigenreduction.
 1000   M2 = M
        CALL PTRED2(M,M2,A,W,FV1,Z)
        CALL PTQL2(M,M2,W,FV1,Z,IERR)
        IF (IERR.NE.0) GOTO 9000
C
C---    Output eigenvalues and eigenvectors.
        IF (IPRINT.GT.0) CALL POUTEV(N,M,W)
        IF (IPRINT.GT.1) CALL POUTVC(N,M,Z)
C
C---    Determine projections and output them.
        CALL PPROJX(N,M,DATA,Z,FV1)
        IF (IPRINT.EQ.3) CALL POUTPX(N,M,DATA)
        CALL PPROJY(M,W,A,Z,FV1)
        IF (IPRINT.EQ.3) CALL POUTPY(M,A)
C
 9000   RETURN  
        END
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C                           
C  Determine correlations of columns.
C  First determine the means of columns, storing in WORK1.
C                                         
C--------------------------------------------------------
        SUBROUTINE PCORCL(N,M,DATA,WORK1,WORK2,OUT)
        DIMENSION       DATA(N,M), OUT(M,M), WORK1(M), WORK2(M)
        DATA            EPS/1.E-10/
C
        DO 30 J = 1, M
           WORK1(J) = 0.0
           DO 20 I = 1, N
              WORK1(J) = WORK1(J) + DATA(I,J)
   20      CONTINUE
           WORK1(J) = WORK1(J)/FLOAT(N)
   30   CONTINUE
C
C          Next det. the std. devns. of cols., storing in WORK2.
C
        DO 50 J = 1, M
           WORK2(J) = 0.0
           DO 40 I = 1, N
              WORK2(J) = WORK2(J) + (DATA(I,J)
     X                   -WORK1(J))*(DATA(I,J)-WORK1(J))
   40      CONTINUE
           WORK2(J) = WORK2(J)/FLOAT(N)
           WORK2(J) = SQRT(WORK2(J))
           IF (WORK2(J).LE.EPS) WORK2(J) = 1.0
   50   CONTINUE

C
C          Now centre and reduce the column points.
C
        DO 70 I = 1, N
           DO 60 J = 1, M
              DATA(I,J) = (DATA(I,J)
     X                    -WORK1(J))/(SQRT(FLOAT(N))*WORK2(J))
   60      CONTINUE
   70   CONTINUE
C
C          Finally calc. the cross product of the data matrix.
C
        DO 100 J1 = 1, M-1
           OUT(J1,J1) = 1.0
           DO 90 J2 = J1+1, M
              OUT(J1,J2) = 0.0
              DO 80 I = 1, N
                 OUT(J1,J2) = OUT(J1,J2) + DATA(I,J1)*DATA(I,J2)
   80         CONTINUE
              OUT(J2,J1) = OUT(J1,J2)
   90      CONTINUE
  100   CONTINUE
        OUT(M,M) = 1.0
C
        RETURN
        END
C++++++++++++++++++++++++++++++++++++++++++++++++++++++
C                       
C  Determine covariances of columns. 
C  First determine the means of columns, storing in WORK.
C                                
C------------------------------------------------------
        SUBROUTINE PCOVCL(N,M,DATA,WORK,OUT)
        DIMENSION       DATA(N,M), OUT(M,M), WORK(M)
C
        DO 30 J = 1, M
           WORK(J) = 0.0
           DO 20 I = 1, N
              WORK(J) = WORK(J) + DATA(I,J)
   20      CONTINUE
           WORK(J) = WORK(J)/FLOAT(N)
   30   CONTINUE
C
C          Now centre the column points.
C
        DO 50 I = 1, N
           DO 40 J = 1, M
              DATA(I,J) = DATA(I,J)-WORK(J)
   40      CONTINUE
   50   CONTINUE
C
C          Finally calculate the cross product matrix of the 
C          redefined data matrix.
C
        DO 80 J1 = 1, M
           DO 70 J2 = J1, M
              OUT(J1,J2) = 0.0
              DO 60 I = 1, N
                 OUT(J1,J2) = OUT(J1,J2) + DATA(I,J1)*DATA(I,J2)
   60         CONTINUE
              OUT(J2,J1) = OUT(J1,J2)
   70      CONTINUE
   80   CONTINUE
C
        RETURN
        END
C++++++++++++++++++++++++++++++++++++++++++++++++++++++
C                       
C  Determine covariances of columns, having first 
C  normalized by dividing by ranges. 
C  First determine max and min of columns, storing in 
C  WORK1 and WORK2.
C                                
C------------------------------------------------------
        SUBROUTINE PRANCV(N,M,DATA,WORK1,WORK2,OUT)
        DIMENSION       DATA(N,M), OUT(M,M), WORK1(M), WORK2(M)
C
        DO 30 J = 1, M
           WORK1(J) = -10000.0
           WORK2(J) =  10000.0
           DO 20 I = 1, N
              IF (DATA(I,J).GT.WORK1(J)) WORK1(J) = DATA(I,J)
              IF (DATA(I,J).LT.WORK2(J)) WORK2(J) = DATA(I,J)
   20      CONTINUE
C          Range:
           WORK1(J) = WORK1(J)-WORK2(J)
   30   CONTINUE
C
C          Now normalize the column points.
C
        DO 50 J = 1, M
           WORK2(J) = 0.0
           DO 40 I = 1, N
              DATA(I,J) = DATA(I,J)/WORK1(J)
              WORK2(J) = WORK2(J) + DATA(I,J)
   40      CONTINUE
           WORK2(J) = WORK2(J)/FLOAT(N)
   50   CONTINUE
C
        DO 58 I = 1, N
           DO 56 J = 1, M
              DATA(I,J) = DATA(I,J)-WORK2(J)
 56        CONTINUE
 58     CONTINUE
C
C          Finally calculate the cross product matrix of the 
C          redefined data matrix.
C
        DO 80 J1 = 1, M
           DO 70 J2 = J1, M
              OUT(J1,J2) = 0.0
              DO 60 I = 1, N
                 OUT(J1,J2) = OUT(J1,J2) + DATA(I,J1)*DATA(I,J2)
   60         CONTINUE
              OUT(J2,J1) = OUT(J1,J2)
   70      CONTINUE
   80   CONTINUE
C
        RETURN
        END
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C                                                         
C  Determine sums of squares and cross-products of columns.
C                                            
C---------------------------------------------------------
        SUBROUTINE PSCPCL(N,M,DATA,OUT)
        DIMENSION       DATA(N,M), OUT(M,M)
C
        DO 30 J1 = 1, M
           DO 20 J2 = J1, M
              OUT(J1,J2) = 0.0
              DO 10 I = 1, N
                 OUT(J1,J2) = OUT(J1,J2) + DATA(I,J1)*DATA(I,J2)
   10         CONTINUE
              OUT(J2,J1) = OUT(J1,J2)
   20      CONTINUE
   30   CONTINUE
C
        RETURN
        END
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C                                   
C Reduce a real, symmetric matrix to a symmetric, tridiagonal
C matrix.                         
C                                 
C To call:    CALL PTRED2(NM,N,A,D,E,Z)    where
C                                     
C NM = row dimension of A and Z;      
C N = order of matrix A (will always be <= NM);
C A = symmetric matrix of order N to be reduced to tridiag. form;
C D = vector of dim. N containing, on output, diagonal elts. of
C     tridiagonal matrix.               
C E = working vector of dim. at least N-1 to contain subdiagonal
C     elements.                          
C Z = matrix of dims. NM by N containing, on output, orthogonal
C     transformation matrix producing the reduction. 
C                                            
C Normally a call to PTQL2 will follow the call to PTRED2 in order to
C produce all eigenvectors and eigenvalues of matrix A.
C                                        
C Algorithm used: Martin et al., Num. Math. 11, 181-195, 1968. 
C                                 
C Reference: Smith et al., Matrix Eigensystem Routines - EISPACK
C Guide, Lecture Notes in Computer Science 6, Springer-Verlag, 
C 1976, pp. 489-494.                     
C                                        
C----------------------------------------------------------------
        SUBROUTINE PTRED2(NM,N,A,D,E,Z)
        REAL A(NM,N),D(N),E(N),Z(NM,N)
C
        DO 100 I = 1, N
           DO 100 J = 1, I
              Z(I,J) = A(I,J)
  100   CONTINUE
        IF (N.EQ.1) GOTO 320
        DO 300 II = 2, N
           I = N + 2 - II
           L = I - 1
           H = 0.0
           SCALE = 0.0
           IF (L.LT.2) GOTO 130
           DO 120 K = 1, L
              SCALE = SCALE + ABS(Z(I,K))
  120      CONTINUE
           IF (SCALE.NE.0.0) GOTO 140
  130      E(I) = Z(I,L)
           GOTO 290
  140      DO 150 K = 1, L
              Z(I,K) = Z(I,K)/SCALE
              H = H + Z(I,K)*Z(I,K)
  150      CONTINUE
C
           F = Z(I,L)
           G = -SIGN(SQRT(H),F)
           E(I) = SCALE * G
           H = H - F * G
           Z(I,L) = F - G
           F = 0.0
C
           DO 240 J = 1, L
              Z(J,I) = Z(I,J)/H
              G = 0.0
C             Form element of A*U.
              DO 180 K = 1, J
                 G = G + Z(J,K)*Z(I,K)
  180         CONTINUE
              JP1 = J + 1
              IF (L.LT.JP1) GOTO 220
              DO 200 K = JP1, L
                 G = G + Z(K,J)*Z(I,K)
  200         CONTINUE
C             Form element of P where P = I - U U' / H .
  220         E(J) = G/H
              F = F + E(J) * Z(I,J)
  240      CONTINUE
           HH = F/(H + H)
C          Form reduced A.
           DO 260 J = 1, L
              F = Z(I,J)
              G = E(J) - HH * F
              E(J) = G
              DO 250 K = 1, J
                 Z(J,K) = Z(J,K) - F*E(K) - G*Z(I,K)
  250         CONTINUE
  260      CONTINUE
  290      D(I) = H
  300   CONTINUE
  320   D(1) = 0.0
        E(1) = 0.0
C       Accumulation of transformation matrices.
        DO 500 I = 1, N
           L = I - 1
           IF (D(I).EQ.0.0) GOTO 380
           DO 360 J = 1, L
              G = 0.0
              DO 340 K = 1, L
                 G = G + Z(I,K) * Z(K,J)
  340         CONTINUE
              DO 350 K = 1, L
                 Z(K,J) = Z(K,J) - G * Z(K,I)
  350         CONTINUE
  360      CONTINUE
  380      D(I) = Z(I,I)
           Z(I,I) = 1.0
           IF (L.LT.1) GOTO 500
           DO 400 J = 1, L
              Z(I,J) = 0.0
              Z(J,I) = 0.0
  400      CONTINUE
  500   CONTINUE
C
        RETURN
        END
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C              
C Determine eigenvalues and eigenvectors of a symmetric,
C tridiagonal matrix.     
C                         
C To call:    CALL PTQL2(NM,N,D,E,Z,IERR)    where
C                             
C NM = row dimension of Z;    
C N = order of matrix Z;      
C D = vector of dim. N containing, on output, eigenvalues;
C E = working vector of dim. at least N-1;      
C Z = matrix of dims. NM by N containing, on output, eigenvectors;
C IERR = error, normally 0, but 1 if no convergence.    
C                      
C Normally the call to PTQL2 will be preceded by a call to PTRED2 in 
C order to set up the tridiagonal matrix.   
C                     
C Algorithm used: QL method of Bowdler et al., Num. Math. 11,
C 293-306, 1968.                   
C                                 
C Reference: Smith et al., Matrix Eigensystem Routines - EISPACK 
C Guide, Lecture Notes in Computer Science 6, Springer-Verlag,
C 1976, pp. 468-474.                 
C                                    
C--------------------------------------------------------------
        SUBROUTINE PTQL2(NM,N,D,E,Z,IERR)
        REAL    D(N), E(N), Z(NM,N)
        DATA    EPS/1.E-12/
C
        IERR = 0
        IF (N.EQ.1) GOTO 1001
        DO 100 I = 2, N
           E(I-1) = E(I)
  100   CONTINUE
        F = 0.0
        B = 0.0
        E(N) = 0.0
C
        DO 240 L = 1, N
           J = 0
           H = EPS * (ABS(D(L)) + ABS(E(L)))
           IF (B.LT.H) B = H
C          Look for small sub-diagonal element.
           DO 110 M = L, N
              IF (ABS(E(M)).LE.B) GOTO 120
C             E(N) is always 0, so there is no exit through
C             the bottom of the loop.
  110      CONTINUE
  120      IF (M.EQ.L) GOTO 220
  130      IF (J.EQ.30) GOTO 1000
           J = J + 1
C          Form shift.
           L1 = L + 1
           G = D(L)
           P = (D(L1)-G)/(2.0*E(L))
           R = SQRT(P*P+1.0)
           D(L) = E(L)/(P+SIGN(R,P))
           H = G-D(L)
C
           DO 140 I = L1, N
              D(I) = D(I) - H
  140      CONTINUE
C
           F = F + H
C          QL transformation.
           P = D(M)
           C = 1.0
           S = 0.0
           MML = M - L
C
           DO 200 II = 1, MML
              I = M - II
              G = C * E(I)
              H = C * P
              IF (ABS(P).LT.ABS(E(I))) GOTO 150
              C = E(I)/P
              R = SQRT(C*C+1.0)
              E(I+1) = S * P * R
              S = C/R
              C = 1.0/R
              GOTO 160
  150         C = P/E(I)
              R = SQRT(C*C+1.0)
              E(I+1) = S * E(I) * R
              S = 1.0/R
              C = C * S
  160         P = C * D(I) - S * G
              D(I+1) = H + S * (C * G + S * D(I))
C             Form vector.
              DO 180 K = 1, N
                 H = Z(K,I+1)
                 Z(K,I+1) = S * Z(K,I) + C * H
                 Z(K,I) = C * Z(K,I) - S * H
  180         CONTINUE
  200      CONTINUE
           E(L) = S * P
           D(L) = C * P
           IF (ABS(E(L)).GT.B) GOTO 130
  220      D(L) = D(L) + F
  240   CONTINUE
C
C       Order eigenvectors and eigenvalues.
        DO 300 II = 2, N
           I = II - 1
           K = I
           P = D(I)
           DO 260 J = II, N
              IF (D(J).GE.P) GOTO 260
              K = J
              P = D(J)
  260      CONTINUE
           IF (K.EQ.I) GOTO 300
           D(K) = D(I)
           D(I) = P
           DO 280 J = 1, N
              P = Z(J,I)
              Z(J,I) = Z(J,K)
              Z(J,K) = P
  280      CONTINUE
  300   CONTINUE
C
        GOTO 1001
C       Set error - no convergence after 30 iterns.
 1000   IERR = L
 1001   RETURN
        END  
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C               
C  Output array.
C               
C---------------------------------------------------------
        SUBROUTINE POUTMT(N,M,ARRAY)
        DIMENSION ARRAY(N,M)
C
        DO 100 K1 = 1, N
           WRITE (6,1000) (ARRAY(K1,K2),K2=1,M)
  100   CONTINUE
C
 1000   FORMAT(10(2X,F8.4))
        RETURN
        END
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C                                   
C  Output half of (symmetric) array.
C                                   
C----------------------------------------------------------
        SUBROUTINE POUTHM(ITYPE,NDIM,ARRAY)
        DIMENSION ARRAY(NDIM,NDIM)
C
        IF (ITYPE.EQ.1) WRITE (6,1000)
        IF (ITYPE.EQ.2) WRITE (6,2000)
        IF (ITYPE.EQ.3) WRITE (6,3000)
        IF (ITYPE.EQ.4) WRITE (6,4000)
        IF (ITYPE.EQ.5) WRITE (6,5000)
        IF (ITYPE.EQ.6) WRITE (6,6000)
C
        DO 100 K1 = 1, NDIM
           WRITE (6,9000) (ARRAY(K1,K2),K2=1,K1)
  100   CONTINUE
C
 1000   FORMAT
     X  (1H0,'SUMS OF SQUARES & CROSS-PRODUCTS MATRIX SECTION.',/)
 2000   FORMAT(1H ,'COVARIANCE MATRIX SECTION.',/)
 3000   FORMAT(1H ,'CORRELATION MATRIX SECTION.',/)
 4000   FORMAT(1H ,'COV. MATRIX OF RANGE-NORMALIZED DATA SECTION.',/)
 5000   FORMAT(1H ,'SPEARMAN CORRELATION MATRIX SECTION.',/)
 6000   FORMAT(1H ,'KENDALL CORRELATION MATRIX SECTION.',/)
 9000   FORMAT(8(2X,F8.4))
        RETURN
        END
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C          
C  Output eigenvalues in order of decreasing value.
C          
C-----------------------------------------------------------
        SUBROUTINE POUTEV(N,NVALS,VALS)
        DIMENSION       VALS(NVALS)
C
        TOT = 0.0
        DO 100 K = 1, NVALS
           TOT = TOT + VALS(K)
  100   CONTINUE
C
        WRITE (6,1000)
        CUM = 0.0
        K = NVALS + 1
C
        M = NVALS
C       (We only want Min(nrows,ncols) eigenvalues output:)
        M = MIN0(N,NVALS)
C
C        WRITE (6,1010)
C        WRITE (6,1020)
  200   CONTINUE
        K = K - 1
        CUM = CUM + VALS(K)
        VPC = VALS(K) * 100.0 / TOT
        VCPC = CUM * 100.0 / TOT
        WRITE (6,1030) VALS(K),VPC,VCPC
C       Correction, Jan. '90: replacing the evals. with the cum. var. has
C       an invalid effect on determining col. projns. later.
C        VALS(K) = VCPC                  
        IF (K.GT.NVALS-M+1) GOTO 200
C
        RETURN
 1000   FORMAT('EIGENVALUE SECTION.')
C 1010   FORMAT
C     X(' Eigenvalues        As Percentages    Cumul. Percentages')
C 1020   FORMAT
C     X(' -----------        --------------    ------------------')
 1030   FORMAT(F13.4,7X,F10.4,10X,F10.4)
        END
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C                
C         Output FIRST SEVEN eigenvectors associated with
C         eigenvalues in descending order.    
C                
C------------------------------------------------------------
        SUBROUTINE POUTVC(N,NDIM,VECS)
        DIMENSION       VECS(NDIM,NDIM)
C
        NUM = MIN0(N,NDIM,7)
C
        WRITE (6,1000)
C        WRITE (6,1010)
C        WRITE (6,1020)
        DO 100 K1 = 1, NDIM
        WRITE (6,1030) K1,(VECS(K1,NDIM-K2+1),K2=1,NUM)
  100   CONTINUE
C
        RETURN
 1000   FORMAT(1H0,'EIGENVECTOR SECTION.',/)
C 1010   FORMAT
C     X  ('  VBLE.   EV-1    EV-2    EV-3    EV-4    EV-5    EV-6 
C     X   EV-7')
C 1020   FORMAT 
C     X  (' ------  ------  ------  ------  ------  ------  ------  
C     X------')
 1030   FORMAT(I5,2X,7F8.4)
        END
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C                 
C  Output projections of row-points on first 7 pricipal components.
C                             
C-------------------------------------------------------------
        SUBROUTINE POUTPX(N,M,PRJN)
        REAL    PRJN(N,M)
C
        NUM = MIN0(M,7)
        WRITE (6,1000)
C        WRITE (6,1010)
C        WRITE (6,1020)
        DO 100 K = 1, N
           WRITE (6,1030) K,(PRJN(K,J),J=1,NUM)
  100   CONTINUE
C
 1000   FORMAT(1H0,'PRINCIPLE SCORE ROW SECTION.',/)
C 1010   FORMAT
C     X  (' OBJECT  PROJ-1  PROJ-2  PROJ-3  PROJ-4  PROJ-5  PROJ-6
C     X  PROJ-7')
C 1020   FORMAT
C     X  (' ------  ------  ------  ------  ------  ------  ------
C     X  ------')
 1030   FORMAT(I5,2X,7F8.4)
        RETURN
        END
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C                                 
C  Output projections of columns on first 7 principal components. 
C                             
C-------------------------------------------------------------
        SUBROUTINE POUTPY(M,PRJNS)
        REAL    PRJNS(M,M)
C
        NUM = MIN0(M,7)
        WRITE (6,1000)
C        WRITE (6,1010)
C        WRITE (6,1020)
        DO 100 K = 1, M
           WRITE (6,1030) K,(PRJNS(K,J),J=1,NUM)
  100   CONTINUE
C
 1000   FORMAT(1H0,'PRINCIPLE SCORE COLUMN SECTION.',/)
C 1010   FORMAT
C     X  ('  VBLE.  PROJ-1  PROJ-2  PROJ-3  PROJ-4  PROJ-5  PROJ-6
C     X  PROJ-7')
C 1020   FORMAT 
C     X  (' ------  ------  ------  ------  ------  ------  ------
C     X  ------')
 1030   FORMAT(I5,2X,7F8.4)
        RETURN
        END
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C              
C  Form projections of row-points on first 7 principal components.
C                     
C--------------------------------------------------------------
        SUBROUTINE PPROJX(N,M,DATA,EVEC,VEC)
        REAL    DATA(N,M), EVEC(M,M), VEC(M)
C
        NUM = MIN0(M,7)
        DO 300 K = 1, N
           DO 50 L = 1, M
              VEC(L) = DATA(K,L)
   50      CONTINUE
           DO 200 I = 1, NUM
              DATA(K,I) = 0.0
              DO 100 J = 1, M
                 DATA(K,I) = DATA(K,I) + VEC(J) *
     X                                   EVEC(J,M-I+1)
  100         CONTINUE
  200      CONTINUE
  300   CONTINUE
C
        RETURN
        END
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C                   
C  Determine projections of column-points on 7 prin. components.
C                           
C--------------------------------------------------------------
        SUBROUTINE PPROJY(M,EVALS,A,Z,VEC)
        REAL    EVALS(M), A(M,M), Z(M,M), VEC(M)
C
        NUM = MIN0(M,7)
        DO 300 J1 = 1, M
           DO 50 L = 1, M
              VEC(L) = A(J1,L)
   50      CONTINUE
           DO 200 J2 = 1, NUM
              A(J1,J2) = 0.0
              DO 100 J3 = 1, M
                 A(J1,J2) = A(J1,J2) + VEC(J3)*Z(J3,M-J2+1)
  100         CONTINUE
              IF (EVALS(M-J2+1).GT.0.00005) A(J1,J2) = 
     X                       A(J1,J2)/SQRT(EVALS(M-J2+1))
              IF (EVALS(M-J2+1).LE.0.00005) A(J1,J2) = 0.0 
  200      CONTINUE
  300   CONTINUE
C
        RETURN
        END
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C  
C   Determine Spearman (rank-order) correlations.
C   Adapted from: Numerical Recipes, Press et al.
C
C-----------------------------------------------------------
        SUBROUTINE PSPEAR(N,M,DATA,WKSP1,WKSP2,A)
        DIMENSION DATA(N,M), WKSP1(N), WKSP2(N), A(M,M)
        DO 900 J1 = 1, M-1
           A(J1,J1) = 1.0
           DO 800 J2 = J1+1, M
              DO 100 I=1,N
                 WKSP1(I)=DATA(I,J1)
                 WKSP2(I)=DATA(I,J2)
 100          CONTINUE
              CALL PSORT(N,WKSP1,WKSP2)
              CALL PRANK(N,WKSP1,SF)
              CALL PSORT(N,WKSP2,WKSP1)
              CALL PRANK(N,WKSP2,SG)
              D=0.
              DO 200 I=1,N
                 D=D+(WKSP1(I)-WKSP2(I))**2
 200          CONTINUE
              EN=N
              EN3N=EN**3-EN
C                AVED=EN3N/6.-(SF+SG)/12.
              FAC=(1.-SF/EN3N)*(1.-SG/EN3N)
C                VARD=((EN-1.)*EN**2*(EN+1.)**2/36.)*FAC
C                ZD=(D-AVED)/SQRT(VARD)
C                PROBD=ERFCC(ABS(ZD)/1.4142136)
              RS=(1.-(6./EN3N)*(D+0.5*(SF+SG)))/FAC
C                T=RS*SQRT((EN-2.)/((1.+RS)*(1.-RS)))
C                DF=EN-2.
C                PROBRS=BETAI(0.5*DF,0.5,DF/(DF+T**2))
              A(J1,J2) = RS
              A(J2,J1) = RS
 800       CONTINUE
 900    CONTINUE
        RETURN
        END
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        SUBROUTINE PSORT(N,RA,RB)
        DIMENSION RA(N),RB(N)
        L=N/2+1
        IR=N
 10     CONTINUE
              IF (L.GT.1) THEN
                        L=L-1
                        RRA=RA(L)
                        RRB=RB(L)
               ELSE
                        RRA=RA(IR)
                        RRB=RB(IR)
                        RA(IR)=RA(1)
                        RB(IR)=RB(1)
                        IR=IR-1
                        IF (IR.EQ.1) THEN
                                RA(1)=RRA
                                RB(1)=RRB
                                RETURN
                        ENDIF
                ENDIF
                I=L
                J=L+L
 20             IF (J.LE.IR) THEN
                        IF (J.LT.IR) THEN
                           IF (RA(J).LT.RA(J+1)) J=J+1
                        ENDIF
                        IF(RRA.LT.RA(J))THEN
                                RA(I)=RA(J)
                                RB(I)=RB(J)
                                I=J
                                J=J+J
                        ELSE
                                J=IR+1
                        ENDIF
                GO TO 20
                ENDIF
                RA(I)=RRA
                RB(I)=RRB
        GO TO 10
        END
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        SUBROUTINE PRANK(N,W,S)
        DIMENSION W(N)
        S=0.
        J=1
 1      IF (J.LT.N) THEN
           IF (W(J+1).NE.W(J)) THEN
                        W(J)=J
                        J=J+1
            ELSE
                        DO 100 JT=J+1,N
                               IF (W(JT).NE.W(J)) GO TO 2
 100                    CONTINUE
                        JT=N+1
 2                      RANK=0.5*(J+JT-1)
                        DO 200 JI=J,JT-1
                               W(JI)=RANK
 200                    CONTINUE
                        T=JT-J
                        S=S+T**3-T
                        J=JT
             ENDIF
        GO TO 1
        ENDIF
        IF(J.EQ.N)W(N)=N
        RETURN
        END
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
C Kendall (rank-order) correlations
C
C-----------------------------------------------------------
        SUBROUTINE PKEND(N,M,DATA,A)
        DIMENSION DATA(N,M), A(M,M)
        DO 900 J1 = 1, M-1
           A(J1,J1) = 1.0
           DO 800 J2 = J1+1, M
              N1=0
              N2=0
              IS=0
               DO 600 J=1,N-1
                  DO 500 K=J+1,N
                        A1=DATA(J,J1)-DATA(K,J1)
                        A2=DATA(J,J2)-DATA(K,J2)
                        AA=A1*A2
                        IF(AA.NE.0.)THEN
                                N1=N1+1
                                N2=N2+1
                                IF(AA.GT.0.)THEN
                                        IS=IS+1
                                ELSE
                                        IS=IS-1
                                ENDIF
                        ELSE
                                IF(A1.NE.0.)N1=N1+1
                                IF(A2.NE.0.)N2=N2+1
                        ENDIF
 500            CONTINUE
 600        CONTINUE
            TAU=FLOAT(IS)/SQRT(FLOAT(N1)*FLOAT(N2))
C           VAR=(4.*N+10.)/(9.*N*(N-1.))
C           Z=TAU/SQRT(VAR)
C           PROB=ERFCC(ABS(Z)/1.4142136)
            A(J1,J2) = TAU
            A(J2,J1) = TAU
 800      CONTINUE
 900    CONTINUE
        RETURN
        END



