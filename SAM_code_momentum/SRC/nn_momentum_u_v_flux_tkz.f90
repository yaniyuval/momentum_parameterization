
! neural_net convection emulator

module nn_momentum_u_v_flux_tkz_mod


!----------------------------------------------------------------------
use netcdf
use vars
use grid
use params , only: fac_cond, fac_fus, tprmin, a_pr, cp
implicit none
private


!---------------------------------------------------------------------
!  ---- public interfaces ----

   public  nn_momentum_u_v_flux_tkz, nn_momentum_u_v_flux_tkz_init, check

!-----------------------------------------------------------------------
!   ---- version number ----

 character(len=128) :: version = '$Id: nn_momentum_u_v_flux_tkz.f90,v 1 2017/08 fms Exp $'
 character(len=128) :: tag = '$Name: fez $'

!-----------------------------------------------------------------------
!   ---- local/private data ----

    logical :: do_init=.true.

    integer :: n_in ! Input dim features
    integer :: n_h1 ! hidden dim
    integer :: n_h2 ! hidden dim
    integer :: n_h3 ! hidden dim
    integer :: n_h4 ! hidden dim
    integer :: n_out ! outputs dim
    integer :: nrf  ! number of vertical levels the NN uses
    integer :: nrfq ! number of vertical levels the NN uses
    integer :: it,jt
    integer :: o_var_dim ! number of output variables  (different types)
    integer :: nrf2  ! number of vertical levels the NN uses
    integer :: nrf3 ! for tkz levels 

    real(4), allocatable, dimension(:,:)     :: r_w1
    real(4), allocatable, dimension(:,:)     :: r_w2
    real(4), allocatable, dimension(:,:)     :: r_w3
    real(4), allocatable, dimension(:,:)     :: r_w4
    real(4), allocatable, dimension(:,:)     :: r_w5
    real(4), allocatable, dimension(:)       :: r_b1
    real(4), allocatable, dimension(:)       :: r_b2
    real(4), allocatable, dimension(:)       :: r_b3
    real(4), allocatable, dimension(:)       :: r_b4
    real(4), allocatable, dimension(:)       :: r_b5
    real(4), allocatable, dimension(:)       :: xscale_mean
    real(4), allocatable, dimension(:)       :: xscale_stnd

    real(4), allocatable, dimension(:)       :: z1
    real(4), allocatable, dimension(:)       :: z2
    real(4), allocatable, dimension(:)       :: z3
    real(4), allocatable, dimension(:)       :: z4
    real(4), allocatable, dimension(:)       :: z5

    real(4), allocatable, dimension(:)       :: yscale_mean
    real(4), allocatable, dimension(:)       :: yscale_stnd

! Namelist:
! nn_filename       data for random forest
! no_ps             set scaled surface pressure to a constant (zero)
! n_neurons         number of neurons
! n_lev_nn          number of vertical levels used
! y_standard_scaler use standard scaling for outputs
! do_rh             use relative humidity instead of specific humidity


!-----------------------------------------------------------------------

contains

!#######################################################################

   subroutine nn_momentum_u_v_flux_tkz_init

!-----------------------------------------------------------------------
!
!        initialization for nn convection
!
!-----------------------------------------------------------------------
integer  unit,io,ierr

! This will be the netCDF ID for the file and data variable.
integer :: ncid
integer :: in_dimid, h1_dimid, out_dimid, single_dimid
integer :: h2_dimid, h3_dimid, h4_dimid
integer :: r_w1_varid, r_w2_varid, r_b1_varid, r_b2_varid
integer :: r_w3_varid, r_w4_varid, r_b3_varid, r_b4_varid
integer :: r_w5_varid, r_b5_varid


integer :: xscale_mean_varid, xscale_stnd_varid
integer :: yscale_mean_varid, yscale_stnd_varid

character(len=256) :: nn_filename
 
      call task_rank_to_index(rank,it,jt)
 

!-------------allocate arrays and read data-------------------------

! For x8
        nn_filename= '/glade/scratch/janniy/mldata_tmp/gcm_regressors/qobsFFTFFFFFF0FFTFTF48FFFFFFFTT815FFTTTTF3048FF_X01_u_v_flux_Ntr13856040_Nte729360_F_T_instead_qin_uug_vug_disteq_O_tkz_u_flux_v_flux_usf_vsf_NN_layers5in157out111_te14_tr15.nc'

! For x4:
!        nn_filename= '/glade/scratch/janniy/mldata_tmp/gcm_regressors/qobsFFTFFFFFF0FFTFTF48FFFFFFFTT415FFTTTTF3048FF_X01_u_v_flux_Ntr13856040_Nte729360_F_T_instead_qin_uug_vug_disteq_O_tkz_u_flux_v_flux_usf_vsf_NN_layers5in157out111_te10_tr10.nc'
 if(masterproc)  write(*,*) nn_filename


! Open the file. NF90_NOWRITE tells netCDF we want read-only access
! Get the varid or dimid for each variable or dimension based on its name.

      call check( nf90_open(     trim(nn_filename),NF90_NOWRITE,ncid ))

      call check( nf90_inq_dimid(ncid, 'N_in', in_dimid))
      call check( nf90_inquire_dimension(ncid, in_dimid, len=n_in))

      call check( nf90_inq_dimid(ncid, 'N_h1', h1_dimid))
      call check( nf90_inquire_dimension(ncid, h1_dimid, len=n_h1))
      call check( nf90_inq_dimid(ncid, 'N_h2', h2_dimid))
      call check( nf90_inquire_dimension(ncid, h2_dimid, len=n_h2))

      call check( nf90_inq_dimid(ncid, 'N_h3', h3_dimid))
      call check( nf90_inquire_dimension(ncid, h3_dimid, len=n_h3))

      call check( nf90_inq_dimid(ncid, 'N_h4', h4_dimid))
      call check( nf90_inquire_dimension(ncid, h4_dimid, len=n_h4)) 
      call check( nf90_inq_dimid(ncid, 'N_out', out_dimid))
      call check( nf90_inquire_dimension(ncid, out_dimid, len=n_out))

      call check( nf90_inq_dimid(ncid, 'N_out_dim', out_dimid))
      call check( nf90_inquire_dimension(ncid, out_dimid, len=o_var_dim)) 

     print *, 'size of features', n_in
     print *, 'size of outputs', n_out

     nrf = 48 ! Size in the vertical 
     nrfq = 47 !Size in the vertical  for advection
     nrf2 = 30
     nrf3 = 15

! Open the file. NF90_NOWRITE tells netCDF we want read-only access
! Get the varid of the data variable, based on its name.
! Read the data.

      call check( nf90_open(     trim(nn_filename),NF90_NOWRITE,ncid ))

      allocate(r_w1(n_in,n_h1))
      allocate(r_w2(n_h1,n_h2))
      allocate(r_w3(n_h2,n_h3))
      allocate(r_w4(n_h3,n_h4))
      allocate(r_w5(n_h4,n_out))

      allocate(r_b1(n_h1))
      allocate(r_b2(n_h2))
      allocate(r_b3(n_h3))
      allocate(r_b4(n_h4))
      allocate(r_b5(n_out))
      allocate(z1(n_h1))
      allocate(z2(n_h2))
      allocate(z3(n_h3))
      allocate(z4(n_h4))
      allocate(z5(n_out))

     
      allocate(xscale_mean(n_in))
      allocate(xscale_stnd(n_in))

      allocate(yscale_mean(o_var_dim))
      allocate(yscale_stnd(o_var_dim))

      call check( nf90_inq_varid(ncid, "w1", r_w1_varid))
      call check( nf90_get_var(ncid, r_w1_varid, r_w1))
      call check( nf90_inq_varid(ncid, "w2", r_w2_varid))
      call check( nf90_get_var(ncid, r_w2_varid, r_w2))

      call check( nf90_inq_varid(ncid, "w3", r_w3_varid))
      call check( nf90_get_var(ncid, r_w3_varid, r_w3))
      call check( nf90_inq_varid(ncid, "w4", r_w4_varid))
      call check( nf90_get_var(ncid, r_w4_varid, r_w4))

      call check( nf90_inq_varid(ncid, "w5", r_w5_varid))
      call check( nf90_get_var(ncid, r_w5_varid, r_w5))

      call check( nf90_inq_varid(ncid, "b1", r_b1_varid))
      call check( nf90_get_var(ncid, r_b1_varid, r_b1))
      call check( nf90_inq_varid(ncid, "b2", r_b2_varid))
      call check( nf90_get_var(ncid, r_b2_varid, r_b2))

      call check( nf90_inq_varid(ncid, "b3", r_b3_varid))
      call check( nf90_get_var(ncid, r_b3_varid, r_b3))
      call check( nf90_inq_varid(ncid, "b4", r_b4_varid))
      call check( nf90_get_var(ncid, r_b4_varid, r_b4))
      call check( nf90_inq_varid(ncid, "b5", r_b5_varid))
      call check( nf90_get_var(ncid, r_b5_varid, r_b5))

      call check( nf90_inq_varid(ncid,"fscale_mean",     xscale_mean_varid))
      call check( nf90_get_var(  ncid, xscale_mean_varid,xscale_mean      ))
      call check( nf90_inq_varid(ncid,"fscale_stnd",     xscale_stnd_varid))
      call check( nf90_get_var(  ncid, xscale_stnd_varid,xscale_stnd      ))

      call check( nf90_inq_varid(ncid,"oscale_mean",     yscale_mean_varid))
      call check( nf90_get_var(  ncid, yscale_mean_varid,yscale_mean      ))
      call check( nf90_inq_varid(ncid,"oscale_stnd",     yscale_stnd_varid))
      call check( nf90_get_var(  ncid, yscale_stnd_varid,yscale_stnd      ))

    
! Close the file
      call check( nf90_close(ncid))

      write(*, *) 'Finished reading NN regression file.'

      do_init=.false.
   end subroutine nn_momentum_u_v_flux_tkz_init


!#######################################################################

   subroutine nn_momentum_u_v_flux_tkz

!-----------------------------------------------------------------------
!
!  NN subgrid parameterization
!
!-----------------------------------------------------------------------
!
!-----------------------------------------------------------------------
!
!   input:  t                  energy var 
!           q                  total non-precipitating water
!           u
!           v
!           distance from equator
!   changes: u_tend              The zonal wind tendency
!            v_tend              The meridional wind tendency
!            diffusivity         vertical diffusivity for momentum variables
!
!-----------------------------------------------------------------------
!
!-----------------------------------------------------------------------



!-----------------------------------------------------------------------
!---------------------- local data -------------------------------------
   real,   dimension(nrf)             :: u_tendency_adv,v_tendency_adv
   real,   dimension(nrf+1)             :: u_flux_adv,v_flux_adv
   real,   dimension(nrf3)            :: tkz_preds
   real,   dimension(nzm)             :: qsat, irhoadz
   real(4), dimension(n_in)     :: features
   real(4), dimension(n_out)      :: outputs
   real    omn, lat_v
   integer  i, j, k,dd, dim_counter, out_dim_counter, out_var_control!, ib, jc

   if (do_init) call error_mesg('nn_momentum_u_v_flux_tkz_init has not been called.')
   if (.not. rf_uses_qp) then
    ! initialize precipitation 
    if(mod(nstep-1,nstatis).eq.0.and.icycle.eq.1) precsfc(:,:)=0.
   end if
 
   do k=1,nzm
           irhoadz(k) = 1.0/(rho(k)*adz(k)*dz) ! Useful factor
           !irhoadzdz(k) = irhoadz(k)/dz ! Note the time step
   end do
   do j=1,ny
    !jc = j + 1
    lat_v = real((dy*(j+jt-(ny_gl+YES3D-1)/2-0.5)))
    do i=1,nx
        !ib = i - 1
        ! Initialize variables
        features = 0.
        outputs = 0.
        u_tendency_adv = 0.
        v_tendency_adv = 0.
        u_flux_adv = 0.
        v_flux_adv = 0.
        tkz_preds = 0.
        z1 = 0.
        z2 = 0.
        z3 = 0.
        z4 = 0.
        z5 = 0.        
        dim_counter = 0
        ! Combine all features into one vector
        if (Tin_feature_rf) then
         features(dim_counter+1:dim_counter + nrf2) = real(t(i,j,1:nrf2),4) !on u grid
         dim_counter = dim_counter + nrf2
        endif
        if (qin_feature_rf) then
        if (rf_uses_rh) then
         ! generalized relative humidity is used as a feature
         do k=1,nzm
          omn = omegan(tabs(i,j,k))
          qsat(k) = omn*qsatw(tabs(i,j,k),pres(k))+(1.-omn)*qsati(tabs(i,j,k),pres(k))
         end do
         features(dim_counter+1:dim_counter+nrf) = real(q_i(i,j,1:nrf)/qsat(1:nrf),4)
         dim_counter = dim_counter + nrf
        else
         ! non-precipitating water is used as a feature
         features(dim_counter+1:dim_counter+nrf2) = real(q(i,j,1:nrf2),4) !on u grid
         dim_counter =  dim_counter + nrf2
        endif
        endif
        if (rf_uses_qp) then
         features(dim_counter+1:dim_counter+nrf) = real(qp_i(i,j,1:nrf),4)
         dim_counter = dim_counter + nrf
        endif

        !If I use the NN before ADAMS... If using after - see below 
        features(dim_counter+1:dim_counter+nrf) = real(u(i,j,1:nrf),4)
        dim_counter = dim_counter + nrf
        features(dim_counter+1:dim_counter+nrf) = real(v(i,j,1:nrf),4)
        dim_counter = dim_counter + nrf


        ! mod feature y
        if(do_yin_input) then
         features(dim_counter+1) = real(abs(dy*(j+jt-(ny_gl+YES3D-1)/2-0.5)))
         dim_counter = dim_counter+1
         
        endif


       
         
!Normalize features
       features = (features - xscale_mean) / xscale_stnd

! calculate predicted values using NN
! Apply trained regressor network to data using rectifier activation function
! forward prop to hiddelayer



        z1 = matmul( features, r_w1) + r_b1
        !print *, 'SHAPE of Z1', shape(z1)
! rectifier
        where (z1 .lt. 0.0)  z1 = 0.0

! forward prop to output layer

        
        z2 = matmul( z1,r_w2) + r_b2
        where (z2 .lt. 0.0)  z2 = 0.0

        z3 = matmul( z2,r_w3) + r_b3
        where (z3 .lt. 0.0)  z3 = 0.0

        z4 = matmul( z3,r_w4) + r_b4
        where (z4 .lt. 0.0)  z4 = 0.0

       outputs = matmul( z4,r_w5) + r_b5



        out_var_control =1
        tkz_preds(1:nrf3) = (outputs(1:nrf3) * yscale_stnd(out_var_control))  +  yscale_mean(out_var_control)
        do k=1,nrf3
         tkz_preds(k) = max(tkz_preds(k),0.)
        end do
        out_dim_counter = nrf3
        out_var_control =out_var_control + 1
        u_flux_adv(2:nrf) = (outputs(out_dim_counter+1:out_dim_counter+nrfq) * yscale_stnd(out_var_control))  +  yscale_mean(out_var_control)

!        advection surface flux is zero
        u_flux_adv(1) = 0.0 ! This is a verification as it should be automatic like that
        u_flux_adv(nrf+1) = 0.0 
        out_dim_counter = out_dim_counter + nrfq
        out_var_control = out_var_control + 1
        v_flux_adv(2:nrf) = (outputs(out_dim_counter+1:out_dim_counter+nrfq) * yscale_stnd(out_var_control))  +  yscale_mean(out_var_control)
        v_flux_adv(1) = 0.0 ! This is a verification as it should be automatic like that
        v_flux_adv(nrf+1) = 0.0
        out_dim_counter = out_dim_counter + nrfq
        out_var_control = out_var_control + 1          
        if (do_rf_q_surf_flux) then !temp flag if we include surface fluxes correction
         u_flux_adv(1) = (outputs(out_dim_counter+1)* yscale_stnd(out_var_control))  +  yscale_mean(out_var_control)
         u_flux_adv(1) = u_flux_adv(1) * dz ! I already divided by dz in the matlab code...
         out_dim_counter = out_dim_counter +1
         out_var_control = out_var_control + 1 
         v_flux_adv(1) = (outputs(out_dim_counter+1)* yscale_stnd(out_var_control))  +  yscale_mean(out_var_control)
         v_flux_adv(1) = v_flux_adv(1) * dz ! I already divided by dz in the matlab code...  
        end if
        do k=1,nrf 
           u_tendency_adv(k) = - (u_flux_adv(k+1) - u_flux_adv(k)) * irhoadz(k)
           v_tendency_adv(k) = - (v_flux_adv(k+1) - v_flux_adv(k)) * irhoadz(k)
        end do
        !Following Adams scheme:
        dudt(i,j,1:nrf,na) = dudt(i,j,1:nrf,na) + u_tendency_adv(1:nrf) 
        dvdt(i,j,1:nrf,na) = dvdt(i,j,1:nrf,na) + v_tendency_adv(1:nrf)
        tk_z(i,j,1:nrf3) = tkz_preds(1:nrf3) 
       end do
     end do
    



   end subroutine nn_momentum_u_v_flux_tkz





!#######################################################################


!##############################################################################
  subroutine check(status)

    ! checks error status after each netcdf, prints out text message each time
    !   an error code is returned. 

    integer, intent(in) :: status

    if(status /= nf90_noerr) then
       write(*, *) trim(nf90_strerror(status))
    end if
  end subroutine check

!#######################################################################
 subroutine error_mesg (message)
  character(len=*), intent(in) :: message

!  input:
!      message   message written to output   (character string)

    if(masterproc) print*, 'Neural network  module: ', message
    stop

 end subroutine error_mesg



!#######################################################################


end module nn_momentum_u_v_flux_tkz_mod

