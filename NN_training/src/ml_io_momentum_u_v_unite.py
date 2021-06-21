import src.ml_load as ml_load
from netCDF4 import Dataset
import netCDF4
import numpy as np
import pickle
import glob
import src.atmos_physics as atmos_physics
import numpy.matlib
import sys
from random import sample


def build_training_dataset(expt, start_time, end_time, interval, n_x_samp=5, train_size=0.9, do_shuffle=True,
                           flag_dict=dict(), is_cheyenne=False,
                           dx=12000 * 16,
                           dy=12000 * 16):
    """Builds training and testing datasets
    Args:
     expt (str): Experiment name
     interval (int): Number of timesteps between when each file is saved
     start_time (int): First timestep
     end_time (int): Last timestep
     n_x_samp (int): Number of random samples at each y and time step
     flag_dict (dict): including the specific configuration that we want to calculate the outputs for
    """
    if is_cheyenne == False:  # On aimsir/esker
        base_dir2 = '/net/aimsir/archive1/janniy/'
        base_dir = '/net/aimsir/archive1/janniy/ML_convection_data_cheyenne/'  # Newer data with more data

    elif flag_dict['resolution'] == 8 or flag_dict['resolution'] == 4 or flag_dict['resolution'] == 32 or flag_dict[
        'resolution'] == 16:
        base_dir = '/glade/scratch/janniy/'
        base_dir2 = '/glade/scratch/janniy/'
    else:
        base_dir = '/glade/work/janniy/'
        base_dir2 = base_dir

    input_dir = base_dir + 'ML_convection_data/' + expt + '/'  # Yani - where I save the files with the diffusion
    output_dir = base_dir2 + 'mldata_tmp/training_data/'

    # seed random number generation for reproducibility
    np.random.seed(123)


    filename_wildcard = input_dir + expt + 'km12x576_576x1440x48_ctl_288_' + str(start_time).zfill(
            10) + '_000*_diff_coarse_space_corrected_tkz' + str(flag_dict['resolution']) + '.nc4'  # New version of data

    print(filename_wildcard)
    filename = glob.glob(filename_wildcard)
    print(filename)

    f = Dataset(filename[0], mode='r')
    x = f.variables['x'][:]  # m
    y = f.variables['y'][:]  # m
    z = f.variables['z'][:]  # m
    p = f.variables['p'][:]  # hPa
    rho = f.variables['rho'][:]  # kg/m^3
    n_x = len(x)
    n_y = len(y)
    n_z = len(z)
    n_z_input = flag_dict['input_upper_lev']
    f.close()

    # Get ssty for calculation of the surface fluxes. (I need it because I had a wrong input for the x32 case}.
    if expt == 'qobs':
        file_in = open(input_dir + 'ssty_' + str(flag_dict['resolution']), 'r')
    elif expt == 'qobs4K':
        file_in = open(input_dir + 'ssty4K_' + str(flag_dict['resolution']), 'r')
    else:
        print('Chose an unknown experiment. Exiting exection.')
        sys.exit()

    int_count = 0
    ssty_coarse = []
    for line_val in file_in.read().split('\n'):
        if int_count < len(y):
            ssty_coarse.append(float(line_val))
        int_count = int_count + 1
    sstxy_coarse = np.matlib.repmat(ssty_coarse, len(x), 1).T

    # Initialize
    file_times = np.arange(start_time, end_time + interval, interval)
    n_files = np.size(file_times)

    Tin = np.zeros((n_z_input, n_y, n_x_samp, n_files))
    qin = np.zeros((n_z_input, n_y, n_x_samp, n_files))
    Tout = np.zeros((n_z, n_y, n_x_samp, n_files))
    qout = np.zeros((n_z, n_y, n_x_samp, n_files))
    if flag_dict['qn_resolved_as_var']:
        qnin = np.zeros((n_z_input, n_y, n_x_samp, n_files))

    # Yani added:
    if flag_dict['do_hor_wind_input']:
        uin = np.zeros((n_z_input, n_y, n_x_samp, n_files))
        vin = np.zeros((n_z_input, n_y, n_x_samp, n_files))
    if flag_dict['do_ver_wind_input']:
        win = np.zeros((n_z_input, n_y, n_x_samp, n_files))

    if flag_dict['do_surf_wind']:
        uAbsSurfin = np.zeros((n_y, n_x_samp, n_files))

    if flag_dict['dist_From_eq_in']:
        albedo_rad = np.zeros((n_y, n_x_samp, n_files))

    if flag_dict['do_q_surf_fluxes_out']:
        qSurfout = np.zeros((n_y, n_x_samp, n_files))  # Yani added
        tSurfout = np.zeros((n_y, n_x_samp, n_files))  # Yani added

    if flag_dict['output_precip']:
        precip_out = np.zeros((n_y, n_x_samp, n_files))  # Yani added

    if flag_dict['do_radiation_output']:
        Qradout = np.zeros((n_z, n_y, n_x_samp, n_files))  # Yani added

    if flag_dict['calc_tkz_z']:
        tkh_zout = np.zeros((flag_dict['tkz_levels'], n_y, n_x_samp, n_files))  # Yani added

    if flag_dict['do_uv_surf_fluxes_out'] or flag_dict['do_momentum_output']:
        u_tend_out = np.zeros((n_z_input, n_y, n_x_samp, n_files))
        v_tend_out = np.zeros((n_z_input, n_y, n_x_samp, n_files))
        w_tend_out = np.zeros((n_z_input, n_y, n_x_samp, n_files))

    if flag_dict['do_uv_surf_fluxes_out']:
        uSurfout = np.zeros((n_y, n_x_samp, n_files))
        vSurfout = np.zeros((n_y, n_x_samp, n_files))


    # Loop over files
    for ifile, file_time in enumerate(file_times):

        print(file_time)

        # Initialize
        zTin = np.zeros((n_z, n_y, n_x))
        zqin = np.zeros((n_z, n_y, n_x))
        if flag_dict['qn_resolved_as_var']:
            zqnin = np.zeros((n_z, n_y, n_x))
        zTout = np.zeros((n_z, n_y, n_x))
        zqout = np.zeros((n_z, n_y, n_x))
        tabs = np.zeros((n_z, n_y, n_x))
        t = np.zeros((n_z, n_y, n_x))
        qt = np.zeros((n_z, n_y, n_x))
        dqp = np.zeros((n_z, n_y, n_x))
        tflux_z = np.zeros((n_z, n_y, n_x))
        qtflux_z = np.zeros((n_z, n_y, n_x))
        qpflux_z = np.zeros((n_z, n_y, n_x))
        w = np.zeros((n_z, n_y, n_x))
        flux_down = np.zeros((n_y, n_x))
        flux_up = np.zeros((n_y, n_x))
        tfull_flux_diff_z = np.zeros((n_z, n_y, n_x))

        tflux_diff_z = np.zeros((n_z, n_y, n_x))
        qtflux_diff_z = np.zeros((n_z, n_y, n_x))
        tflux_diff_coarse_z = np.zeros((n_z, n_y, n_x))
        qtflux_diff_coarse_z = np.zeros((n_z, n_y, n_x))
        # qpflux_diff_z = np.zeros((n_z, n_y, n_x))
        qpflux_diff_coarse_z = np.zeros((n_z, n_y, n_x))

        # from sedimentation
        cloud_qt_tend = np.zeros(
            (n_z, n_y, n_x))  # found that there is no big difference between residuals and coarse - take coarse
        cloud_lat_heat = np.zeros(
            (n_z, n_y, n_x))  # found that there is no big difference between residuals and coarse - take coarse

        # fall tend
        dqp_fall = np.zeros((n_z, n_y, n_x))
        t_fall = np.zeros((n_z, n_y, n_x))
        # Marker to note distance from equator. Should be proxy for sst,albedo,radiation,etc.
        zalbedo_rad = np.zeros((n_y, n_x))

        if flag_dict['do_hor_wind_input'] or flag_dict['do_surf_wind']:
            u = np.zeros((n_z, n_y, n_x))
            v = np.zeros((n_z, n_y, n_x))

        if flag_dict['do_q_surf_fluxes_out']:
            zqSurfout = np.zeros((n_y, n_x))  # Yani added
            ztSurfout = np.zeros((n_y, n_x))  # Yani added

        if flag_dict['do_momentum_output']:
            zu_adv_out = np.zeros((n_z, n_y, n_x))
            zv_adv_out = np.zeros((n_z, n_y, n_x))
            zw_adv_out = np.zeros((n_z, n_y, n_x))

        if flag_dict['do_uv_surf_fluxes_out']:
            zuSurfout = np.zeros((n_y, n_x))  # Yani added
            zvSurfout = np.zeros((n_y, n_x))  # Yani added

        if flag_dict['do_uv_surf_fluxes_out'] or flag_dict['do_momentum_output']:
            zu_tend_out = np.zeros((n_z, n_y, n_x))
            zv_tend_out = np.zeros((n_z, n_y, n_x))
            zw_tend_out = np.zeros((n_z, n_y, n_x))


        # Variables to calculate the diffusivity.
        tkz_z = np.zeros((n_z, n_y, n_x))  # Yani added
        Pr1 = np.zeros((n_z, n_y, n_x))  # Yani added
        tkz_z_res = np.zeros((n_z, n_y, n_x))  # Yani added
        Pr1_res = np.zeros((n_z, n_y, n_x))  # Yani added
        # tkh_x = np.zeros((n_z, n_y, n_x))  # Yani added
        # tkh_y = np.zeros((n_z, n_y, n_x))  # Yani added
        tkh_z = np.zeros((n_z, n_y, n_x))  # Yani added

        if flag_dict['do_surf_flux_hemispheric_symmetric_correction']:  # Since I wanted to change how I calculate the surface scheme
            umin = 1.0
            cd = 1.1e-3
            wrk = (np.log(10 / 1.e-4) / np.log(z[0] / 1.e-4)) ** 2
            fluxbtfull = np.zeros((n_y, n_x))
            fluxbqt = np.zeros((n_y, n_x))
            Tfull_diff_z1 = np.zeros((n_z,n_y, n_x)) #Yani added
            Tfull_diff_z2 = np.zeros((n_z,n_y, n_x)) #Yani added
            Tfull = np.zeros((n_z,n_y, n_x)) #Yani added
            # s_wind= np.zeros((n_y, n_x))
            windspeed= np.zeros((n_y, n_x))
            deltfull= np.zeros((n_y, n_x))
            ssq= np.zeros((n_y, n_x))
            delqt= np.zeros((n_y, n_x))

        # Get filename


        filename_wildcard = input_dir + expt + 'km12x576_576x1440x48_ctl_288_' + str(file_time).zfill(
                10) + '_000*_diff_coarse_space_corrected_tkz' + str(flag_dict['resolution']) + '.nc4'  # New version of data

        filename = glob.glob(filename_wildcard)
        print(filename[0])

        # Open file and grab variables from it
        f = Dataset(filename[0], mode='r')
        # n_z x n_y x n_x
        if flag_dict['tabs_resolved_init']:
            tabs = f.variables['TABS_RESOLVED_INIT'][:]  # absolute temperature (K)
        else:
            tabs = f.variables['TABS'][:]  # absolute temperature (K)
        t = f.variables['T'][:]  # liquid static energy/cp (K)
        Qrad = f.variables['QRAD'][:] / 86400  # rad heating rate (K/s)
        # f.variables['QN'][:] - This is not good I think - it is the coarse grained value (in the end of the time step instead of the resolved at the beginning (?) of the time step

        if flag_dict['qn_coarse_init']:
            qt = (f.variables['Q'][:] + f.variables['QN_COARSE_INIT'][:]) / 1000.0  # total non-precip water (kg/kg)
        else:
            qt = (f.variables['Q'][:] + f.variables['QN'][:]) / 1000.0  # total non-precip water (kg/kg)

        if flag_dict['qn_resolved_as_var']:
            qn = f.variables['QN'][:] / 1000.0

        qp = f.variables['QP'][:] / 1000.0  # precipitating water (kg/kg)
        dqp = f.variables['DQP'][:] / 1000.0  # kg/kg/s - Taking coarse result since it is a smaller value to predict!
        dqp_t = f.variables['DQP_T_TEND_COARSE'][:]  # kg/kg/s - Taking coarse result since it is a smaller value to predict!
        tflux_z = f.variables['T_FLUX_Z'][:]  # SGS t flux K kg/m^2/s - new name in new version
        qtflux_z = f.variables['QT_FLUX_Z'][:] / 1000.0  # SGS qt flux kg/m^2/s
        qpflux_z = f.variables['QP_FLUX_Z'][:] / 1000.0  # SGS qp flux kg/m^2/s
        qpflux_z_coarse = f.variables['QP_FLUX_COARSE_Z'][
                          :] / 1000.0  # SGS qp flux kg/m^2/s #I need it to the calculation of the dL/dz term
        if sum(sum(sum(qpflux_z_coarse == 0))) == qpflux_z_coarse.shape[0] * qpflux_z_coarse.shape[1] * \
                qpflux_z_coarse.shape[2]:  # means that we didn't really wrote well qpflux_coarse
            raise Exception('Probably I did mess in the matlab files and didnt get correctly qpflux_z_coarse')

        if flag_dict['do_sedimentation']:
            # raise Exception('I didnt calculate yet in the matlab the sedimentation! - Need to do it...')
            # cloud_qt_tend = f.variables['QT_TEND_CLOUD_COARSE'][:]/1000.0 #found that there is no big difference between residuals and coarse - take coarse
            # cloud_lat_heat = f.variables['LAT_HEAT_CLOUD_COARSE'][:] #found that there is no big difference between residuals and coarse - take coarse
            # Decided to try a correction and see if in the online results it helps...
            cloud_qt_tend = f.variables['QT_TEND_CLOUD_RES'][:] / 1000.0
            cloud_lat_heat = f.variables['LAT_HEAT_CLOUD_RES'][:]

        if flag_dict['do_fall_tend']:
            raise Exception(
                'do_fall_tend - I think that this should be true only in the case that I do the whole 3D fields... ...')
            dqp_fall = f.variables['DQP_FALL_RES'][
                       :] / 1000.0  # Better to take the resolved (unless not doing the fall! - which is the case if not doing the full thing
            t_fall = f.variables['T_FALL_RES'][:]

        w = f.variables['W'][:]  # m/s
        precip = f.variables['PRECIP'][:]  # precipitation flux kg/m^2/s
        # Yani added
        if flag_dict['do_z_diffusion']:
            if flag_dict['Flux_for_TFULL']:
                tflux_diff_z = f.variables['TFULL_DIFF_FLUX_Z'][:] # SGS t flux K kg/m^2/s
            else:
                tflux_diff_z = f.variables['T_DIFF_FLUX_Z'][:]  # SGS t flux K kg/m^2/s #Was there a reason that I used tfull flux ??

            qtflux_diff_z = f.variables['QT_DIFF_FLUX_Z'][:] / 1000.0  # SGS qt flux kg/m^2/s

            tflux_diff_coarse_z = f.variables['T_DIFF_F_COARSE_Z'][
                                  :]  # SGS t flux K kg/m^2/s #Was there a reason that I used tfull flux ??
            qtflux_diff_coarse_z = f.variables['QT_DIFF_F_COARSE_Z'][:] / 1000.0  # SGS qt flux kg/m^2/s
            # qpflux_diff_z = f.variables['QP_DIFF_FLUX_COARSE_Z'][:]/1000.0 # SGS qp flux kg/m^2/s
        if flag_dict['do_qp_diff_corr_to_T']:
            qpflux_diff_coarse_z = f.variables['QP_DIFF_F_COARSE_Z'][
                                   :] / 1000.0  # SGS qp flux kg/m^2/s Note that I need this variable
            # in any case! because we use a different variable in the simple version...

        # Yani added wind variables for prediction:
        if flag_dict['do_hor_wind_input'] or flag_dict['do_surf_wind']:
            if 'no_c_grid' in flag_dict.keys():
                if flag_dict['no_c_grid']:
                    u = f.variables['U_NORM_GRID'][:]
                    v = f.variables['V_NORM_GRID'][:]
                else:
                    u = f.variables['U'][:]
                    v = f.variables['V'][:]
            else:
                u = f.variables['U'][:]
                v = f.variables['V'][:]

        # Variables to calculate the diffusivity.
        if flag_dict['calc_tkz_z']:
            if flag_dict['calc_tkz_z_correction']:
                # tkz_z_res = f.variables['TKZ_COARSE'][:] - f.variables['TKZ_RES'][:]  # diffusivity - m^2/s
                # Pr1_res = f.variables['PR1_COARSE'][:]-f.variables['PR1_RES'][:]  # no units
                tkz_z = f.variables['TKZ_COARSE'][:]  # diffusivity - m^2/s
                Pr1 = f.variables['PR1_COARSE'][:]  # no units
            else:
                tkz_z = f.variables['TKZ_COARSE'][:]  # diffusivity - m^2/s
                Pr1 = f.variables['PR1_COARSE'][:]  # no units

        if flag_dict['do_surf_flux_hemispheric_symmetric_correction'] and flag_dict['resolution'] == 32:  # Since I wanted to change how I calculate the surface scheme
            gamaz = atmos_physics.g / atmos_physics.cp * z[0]
            # t = f.variables['T'][:]
            if flag_dict['Flux_for_TFULL']:
                tflux_diff_z = f.variables['TFULL_DIFF_F_COARSE_Z'][:]
                # Tfull_diff_z2 = f.variables['TFULL_DIFF_FLUX_Z'][:]
                Tfull = f.variables['TFULL'][:]
            else:
                tflux_diff_z = f.variables['T_DIFF_F_COARSE_Z'][:]
                # Tfull_diff_z2 = f.variables['T_DIFF_FLUX_Z'][:]
                Tfull = f.variables['T'][:]

        if flag_dict['do_momentum_output']:
            if 'adv_u_v_grid' in flag_dict.keys(): # Calculated advection term without noise - from the local data of the grid itelf...
                if flag_dict['adv_u_v_grid']:
                    zu_adv_out = f.variables['U_ADV_U_GRID_RESOLVED'][:]
                    zv_adv_out = f.variables['V_ADV_V_GRID_RESOLVED'][:]
                    zw_adv_out = f.variables['W_ADV_RESOLVED'][:]
                else:
                    zu_adv_out = f.variables['U_ADV_RESOLVED'][:]
                    zv_adv_out = f.variables['V_ADV_RESOLVED'][:]
                    zw_adv_out = f.variables['W_ADV_RESOLVED'][:]
            elif 'no_c_grid' in flag_dict.keys():
                if flag_dict['no_c_grid']:
                    zu_adv_out = f.variables['U_ADV_NORM_GRID_RESOLVED'][:]
                    zv_adv_out = f.variables['V_ADV_NORM_GRID_RESOLVED'][:]
                    zw_adv_out = f.variables['W_ADV_RESOLVED'][:]
                else:
                    zu_adv_out = f.variables['U_ADV_RESOLVED'][:]
                    zv_adv_out = f.variables['V_ADV_RESOLVED'][:]
                    zw_adv_out = f.variables['W_ADV_RESOLVED'][:]
            else:
                zu_adv_out = f.variables['U_ADV_RESOLVED'][:]
                zv_adv_out = f.variables['V_ADV_RESOLVED'][:]
                zw_adv_out = f.variables['W_ADV_RESOLVED'][:]

        if flag_dict['do_uv_surf_fluxes_out']:
            if 'adv_u_v_grid' in flag_dict.keys(): # Calculated advection term without noise - from the local data of the grid itelf...
                if flag_dict['adv_u_v_grid']:
                    zuSurfout = f.variables['U_SURF_FLUX_U_GRID_RESOLVED'][:]
                    zvSurfout = f.variables['V_SURF_FLUX_V_GRID_RESOLVED'][:]
                else:
                    zuSurfout = f.variables['SFLUX_U_RESOLVED_SAM'][:]
                    zvSurfout = f.variables['SFLUX_V_RESOLVED_SAM'][:]
            elif 'no_c_grid' in flag_dict.keys():
                if flag_dict['no_c_grid']:
                    zuSurfout = f.variables['U_SURF_FLUX_NORM_GRID_SAM_RESOLVED'][:]
                    zvSurfout = f.variables['V_SURF_FLUX_NORM_GRID_SAM_RESOLVED'][:]
                else:
                    zuSurfout = f.variables['SFLUX_U_RESOLVED_SAM'][:]
                    zvSurfout = f.variables['SFLUX_V_RESOLVED_SAM'][:]
            else:
                zuSurfout = f.variables['SFLUX_U_RESOLVED_SAM'][:]
                zvSurfout = f.variables['SFLUX_V_RESOLVED_SAM'][:]
        f.close()


        if flag_dict['do_surf_flux_hemispheric_symmetric_correction'] and flag_dict['resolution'] == 32:  # Since I had a wrong input file to the ssty_32 ! I want to recalculate surface flux!
            windspeed[:, :] = np.sqrt(np.square(u[0, :, :]) + np.square(v[0, :, :]) + np.square(umin))
            deltfull[:, :] = Tfull[0, :, :] - gamaz - sstxy_coarse[:, :]
            ssq[:, :] = atmos_physics.sam_qsatw(sstxy_coarse[:, :], p[0])
            delqt[:, :] = qt[0, :, :] - ssq
            rhow = 1.16
            fluxbtfull[:, :] = -cd * windspeed[:, :] * deltfull[:, :] * wrk * rhow  # I multiply by rho to be consistant with the matlab definitions.
            fluxbqt[:, :] = -cd * windspeed[:, :] * delqt[:, :] * wrk * rhow
            # del111 = tflux_diff_z - Tfull_diff_z2
            # aaa1111 = qtflux_diff_coarse_z - qtflux_diff_z
            tflux_diff_z[0, :, :] = tflux_diff_z[0, :, :] - fluxbtfull  # Correcting the fluxes
            qtflux_diff_z[0, :, :] = qtflux_diff_coarse_z[0, :, :] - fluxbqt


        if flag_dict['T_instead_of_Tabs']:
            zTin = t
        else:
            zTin = tabs

        zqin = qt

        if flag_dict['qn_resolved_as_var']:
            zqnin = qn

        # Yani added
        if flag_dict['do_hor_wind_input']:
            zuin = u
            zvin = v

        if flag_dict['do_ver_wind_input']:
            zwin = w

        if flag_dict['do_surf_wind']:
            zuAbsSurfin = np.sqrt(np.square(u[0, :, :]) + np.square(v[0, :, :]))

        # approach where find tendency of hL without qp
        # use omp since heating as condensate changes to precipitation
        # of different phase also increases hL
        a_pr = 1.0 / (atmos_physics.tprmax - atmos_physics.tprmin)
        omp = np.maximum(0.0, np.minimum(1.0, (tabs - atmos_physics.tprmin) * a_pr))
        fac = (atmos_physics.L + atmos_physics.Lf * (1.0 - omp)) / atmos_physics.cp

        # follow simplified version of advect_scalar3D.f90 for vertical advection
        rho_dz = atmos_physics.vertical_diff(rho, z)


        # # Add momentum tendencies due to surface flux corrections
        # if flag_dict['do_uv_surf_fluxes_out']:
        #     zu_tend_out[:,:] = zuSurfout[:, :] # / rho_dz[0]  # I commented because I wanted to stay with flux!
        #     zv_tend_out[:,:] = zvSurfout[:, :] # / rho_dz[0]

        if flag_dict['do_momentum_output']:
            zu_adv_flux = np.zeros((zu_adv_out.shape[0]+1,zu_adv_out.shape[1],zu_adv_out.shape[2])) #need 49 levels for the flux. The first and last will be zero...
            zv_adv_flux = np.zeros((zu_adv_out.shape[0] + 1, zu_adv_out.shape[1], zu_adv_out.shape[2]))  # need 49 levels for the flux. The first and last will be zero...
            for k in range(0, zu_adv_out.shape[0]):
                zu_adv_flux[k + 1, :, :] = zu_adv_flux[k, :, :] - zu_adv_out[k, :, :] * rho_dz[k]
                zv_adv_flux[k + 1, :, :] = zv_adv_flux[k, :, :] - zv_adv_out[k, :, :] * rho_dz[k]

            zu_tend_out = zu_tend_out + zu_adv_flux[0:-1,:,:] # Ignore the upper level which should be zero (TO VERIFY)
            zv_tend_out = zv_tend_out + zv_adv_flux[0:-1,:,:] # Ignore the upper level which should be zero (TO VERIFY)
            zw_tend_out = zw_tend_out + zw_adv_out


        # print('mod include y as proxy for solar rad, sst, albedo')
        if flag_dict['dist_From_eq_in']:
            zalbedo_rad[:, :] = np.abs(y[:, None] - np.mean(y))

        if flag_dict['calc_tkz_z']:
            # for k in range(n_z-1):
            # kc = k + 1
            # tkh_z[k, j, i] = rdz5*(tkz_z[k,j,i] * Pr1[k,j,i] + tkz_z[kc,j,i] * Pr1[kc,j,i])
            if flag_dict['calc_tkz_z_correction']:
                tkh_z[:, :, :] = tkz_z[:, :, :] # For momentum parameterization... Can work only after I fix the ADAM...
                # tkh_z[:, :, :] = tkz_z[:, :, :] * Pr1[:, :, :] - tkz_z_res[:, :, :] * Pr1_res[:, :, :]
            else:
                tkh_z[:, :, :] = tkz_z[:, :, :] * Pr1[:, :, :]

# Added for the U, V diffusion calculation - eventually might want to predict it in the main RF ?
        if flag_dict['calc_tkz_xy']:
            if flag_dict['calc_tkz_xy_correction']:
                tkh_z[:, :, :] = tkz_z[:, :, :] - tkz_z_res[:, :, :]
            else:
                tkh_z[:, :, :] = tkz_z[:, :, :]

        # n_z n_y n_x ->
        # n_z n_y n_x n_file
        # Loop over y
        for j in range(zTin.shape[1]):
            # Randomly choose a few x's
            truncate_edge = 1  # Yani added to avoid the boundaries which are less acurate
            # ind_x = np.random.randint(0 + truncate_edge, zTin.shape[2] - truncate_edge, n_x_samp)

            ind_x = sample(range(truncate_edge,zTin.shape[2]-truncate_edge), n_x_samp)
            # Numpy has some strange behavior when indexing and slicing are
            # combined. See: http://stackoverflow.com/q/27094438
            Tin[:, j, :, ifile] = zTin[0:n_z_input, j, :][:, ind_x]
            qin[:, j, :, ifile] = zqin[0:n_z_input, j, :][:, ind_x]
            Tout[:, j, :, ifile] = zTout[:, j, :][:, ind_x]
            qout[:, j, :, ifile] = zqout[:, j, :][:, ind_x]
            if flag_dict['qn_resolved_as_var']:
                qnin[:, j, :, ifile] = zqnin[0:n_z_input, j, :][:, ind_x]
            # Yani added
            if flag_dict['do_hor_wind_input']:
                uin[:, j, :, ifile] = zuin[0:n_z_input, j, :][:, ind_x]
                vin[:, j, :, ifile] = zvin[0:n_z_input, j, :][:, ind_x]
            if flag_dict['do_ver_wind_input']:
                win[:, j, :, ifile] = zwin[0:n_z_input, j, :][:, ind_x]
            if flag_dict['do_q_surf_fluxes_out']:
                qSurfout[j, :, ifile] = zqSurfout[j, :][ind_x]  # Yani added - Need to check!
                tSurfout[j, :, ifile] = ztSurfout[j, :][ind_x]  # Yani added - Need to check!

            if flag_dict['output_precip']:
                precip_out[j, :, ifile] = zprecip_out[j, :][ind_x]
            if flag_dict['do_surf_wind']:
                uAbsSurfin[j, :, ifile] = zuAbsSurfin[j, :][ind_x]  # Yani added - Need to check!
            if flag_dict['do_radiation_output']:
                Qradout[0:flag_dict['rad_level'], j, :, ifile] = Qrad[0:flag_dict['rad_level'], j, :][:, ind_x]
            if flag_dict['calc_tkz_z']:
                tkh_zout[0:flag_dict['tkz_levels'], j, :, ifile] = tkh_z[0:flag_dict['tkz_levels'], j, :][:, ind_x]
            if flag_dict['dist_From_eq_in']:
                albedo_rad[j, :, ifile] = zalbedo_rad[j, :][ind_x]

            if flag_dict['do_momentum_output']:
                u_tend_out[0:flag_dict['strat_corr_level'], j, :, ifile] = zu_tend_out[0:flag_dict['strat_corr_level'], j, :][:, ind_x]
                v_tend_out[0:flag_dict['strat_corr_level'], j, :, ifile] = zv_tend_out[0:flag_dict['strat_corr_level'], j, :][:, ind_x]
                w_tend_out[0:flag_dict['strat_corr_level'], j, :, ifile] = zw_tend_out[0:flag_dict['strat_corr_level'], j, :][:, ind_x]

            # Add momentum tendencies due to surface flux corrections
            if flag_dict['do_uv_surf_fluxes_out']:
                uSurfout[j, :, ifile] = zuSurfout[j, :][ind_x] # / rho_dz[0]*dz?   # I commented because I wanted to stay with flux!
                vSurfout[j, :, ifile] = zvSurfout[j, :][ind_x] # / rho_dz[0]*dz?   #I already divided by dz in the matlab code...

            # Reshape array to be n_z n_y n_samp*n_file

    Tin = np.moveaxis(Tin, 2, 3)
    qin = np.moveaxis(qin, 2, 3)

    Tin = np.reshape(Tin, (n_z_input, n_y, -1))
    qin = np.reshape(qin, (n_z_input, n_y, -1))

    if flag_dict['calc_tkz_z']:
        tkh_zout = np.moveaxis(tkh_zout, 2, 3)
        tkh_zout = np.reshape(tkh_zout, (flag_dict['tkz_levels'], n_y, -1))

    # Yani added
    if flag_dict['do_hor_wind_input']:
        uin = np.moveaxis(uin, 2, 3)
        vin = np.moveaxis(vin, 2, 3)
        uin = np.reshape(uin, (n_z_input, n_y, -1))
        vin = np.reshape(vin, (n_z_input, n_y, -1))
    if flag_dict['do_ver_wind_input']:
        win = np.moveaxis(win, 2, 3)
        win = np.reshape(win, (n_z_input, n_y, -1))

    if flag_dict['do_surf_wind']:
        uAbsSurfin = np.moveaxis(uAbsSurfin, 1, 2)
        uAbsSurfin = np.reshape(uAbsSurfin, (n_y, -1))
    if flag_dict['dist_From_eq_in']:
        albedo_rad = np.moveaxis(albedo_rad, 1, 2)
        albedo_rad = np.reshape(albedo_rad, (n_y, -1))

    if flag_dict['do_momentum_output']:
        u_tend_out = np.moveaxis(u_tend_out, 2, 3)
        v_tend_out = np.moveaxis(v_tend_out, 2, 3)
        w_tend_out = np.moveaxis(w_tend_out, 2, 3)

        u_tend_out= np.reshape(u_tend_out, (n_z_input, n_y, -1))
        v_tend_out= np.reshape(v_tend_out, (n_z_input, n_y, -1))
        w_tend_out= np.reshape(w_tend_out, (n_z_input, n_y, -1))

    if flag_dict['do_uv_surf_fluxes_out']:
        uSurfout = np.moveaxis(uSurfout, 1, 2)
        vSurfout = np.moveaxis(vSurfout, 1, 2)

        uSurfout = np.reshape(uSurfout, (n_y, -1))
        vSurfout = np.reshape(vSurfout, (n_y, -1))


        # Shuffle data and store it in separate training and testing files
    if do_shuffle:
        n_trn_exs = Tin.shape[2]
        randinds = np.random.permutation(n_trn_exs)
        i70 = int(train_size * np.size(randinds))
        randind_trn = randinds[:i70]
        randind_tst = randinds[i70:]
    else:
        i70 = int(train_size * Tin.shape[2])
        # randind_trn =np.arange(0,i70,1)
        randind_trn = np.random.permutation(i70)
        tst_list = np.arange(i70, int(Tin.shape[2]), 1)
        randind_tst = np.random.permutation(tst_list)

    data_specific_description = create_specific_data_string_desc(flag_dict)

    train_input_list = []
    test_input_list = []




    # Choosing the lists that will be dumped in the pickle file...
    if flag_dict['Tin_feature']:
        train_input_list.append(np.float32(Tin[0:flag_dict['sed_level'], :, randind_trn]))
        test_input_list.append(np.float32(Tin[0:flag_dict['sed_level'], :, randind_tst]))
    if flag_dict['qin_feature']:
        train_input_list.append(np.float32(qin[0:flag_dict['sed_level'], :, randind_trn]))
        test_input_list.append(np.float32(qin[0:flag_dict['sed_level'], :, randind_tst]))
    if flag_dict['qn_resolved_as_var']:
        train_input_list.append(np.float32(qnin[:, :, randind_trn]))
        test_input_list.append(np.float32(qnin[:, :, randind_tst]))
    if flag_dict['Tin_z_grad_feature']:
        T_grad_in_tr = create_z_grad_plus_surf_var(Tin[:, :, randind_trn])
        train_input_list.append(T_grad_in_tr)
        T_grad_in_test = create_z_grad_plus_surf_var(Tin[:, :, randind_tst])
        test_input_list.append(T_grad_in_test)
    if flag_dict['qin_z_grad_feature']:
        q_grad_in_tr = create_z_grad_plus_surf_var(qin[:, :, randind_trn])
        train_input_list.append(q_grad_in_tr)
        q_grad_in_test = create_z_grad_plus_surf_var(qin[:, :, randind_tst])
        test_input_list.append(q_grad_in_test)
    if flag_dict['Tin_s_diff_feature']:
        T_s_diff_in_tr = create_difference_from_surface(Tin[:, :, randind_trn])
        train_input_list.append(T_s_diff_in_tr)
        T_s_diff_in_test = create_difference_from_surface(Tin[:, :, randind_tst])
        test_input_list.append(T_s_diff_in_test)
    if flag_dict['qin_s_diff_feature']:
        q_s_diff_in_tr = create_difference_from_surface(qin[:, :, randind_trn])
        train_input_list.append(q_s_diff_in_tr)
        q_s_diff_in_test = create_difference_from_surface(qin[:, :, randind_tst])
        test_input_list.append(q_s_diff_in_test)
    if flag_dict['do_hor_wind_input']:
        # uin = np.square(uin[:,:,:])
        # vin = np.square(vin[:, :, :])
        # Using minus v instead
        # vin[:, 0:np.int(vin.shape[1] / 2), :] = -vin[:, 0:np.int(vin.shape[1] / 2), :]
        #if flag_dict['do_momentum_output']:
            # uin = vertical_smooth(uin)
            # vin = vertical_smooth(vin)
        train_input_list.extend([np.float32(uin[:, :, randind_trn]), np.float32(vin[:, :, randind_trn])])
        test_input_list.extend([np.float32(uin[:, :, randind_tst]), np.float32(vin[:, :, randind_tst])])
    if flag_dict['do_ver_wind_input']:
        if flag_dict['do_momentum_output']:
            #win = vertical_smooth(win)
            train_input_list.append(np.float32(win[:, :, randind_trn]))
            test_input_list.append(np.float32(win[:, :, randind_tst]))
        else:
            train_input_list.append(np.float32(win[:, :, randind_trn]))
            test_input_list.append(np.float32(win[:, :, randind_tst]))
    if flag_dict['do_surf_wind']:
        train_input_list.append(np.float32(uAbsSurfin[:, randind_trn]))
        test_input_list.append(np.float32(uAbsSurfin[:, randind_tst]))
    if flag_dict['dist_From_eq_in']:
        train_input_list.append(np.float32(albedo_rad[:, randind_trn]))
        test_input_list.append(np.float32(albedo_rad[:, randind_tst]))
    if flag_dict['predict_tendencies']:
        train_input_list.extend([np.float32(Tout[:, :, randind_trn]), np.float32(qout[:, :, randind_trn])])
        test_input_list.extend([np.float32(Tout[:, :, randind_tst]), np.float32(qout[:, :, randind_tst])])
    if flag_dict['do_radiation_output']:
        train_input_list.append(np.float32(Qradout[0:flag_dict['rad_level'], :, randind_trn]))
        test_input_list.append(np.float32(Qradout[0:flag_dict['rad_level'], :, randind_tst]))
    if flag_dict['do_q_surf_fluxes_out']:
        train_input_list.append(np.float32(tSurfout[:, randind_trn]))
        test_input_list.append(np.float32(tSurfout[:, randind_tst]))
        train_input_list.append(np.float32(qSurfout[:, randind_trn]))
        test_input_list.append(np.float32(qSurfout[:, randind_tst]))
    if flag_dict['output_precip']:
        train_input_list.append(np.float32(precip_out[:, randind_trn]))
        test_input_list.append(np.float32(precip_out[:, randind_tst]))
    if flag_dict['calc_tkz_z']:
        train_input_list.append(np.float32(tkh_zout[0:flag_dict['tkz_levels'], :, randind_trn]))
        test_input_list.append(np.float32(tkh_zout[0:flag_dict['tkz_levels'], :, randind_tst]))
    if flag_dict['do_momentum_output']:
        # train_input_list.extend([np.float32(u_tend_out[1:, :, randind_trn]), np.float32(v_tend_out[1:, :, randind_trn]), np.float32(w_tend_out[1:, :, randind_trn])])
        # test_input_list.extend([np.float32(u_tend_out[1:, :, randind_tst]), np.float32(v_tend_out[1:, :, randind_tst]), np.float32(w_tend_out[1:, :, randind_tst])])
        if 'abs_mom_flux' in flag_dict.keys():
            if flag_dict['abs_mom_flux']:
                train_input_list.extend(
                    [np.abs(np.float32(u_tend_out[1:, :, randind_trn])), np.abs(np.float32(v_tend_out[1:, :, randind_trn]))])
                test_input_list.extend(
                    [np.float32(np.abs(u_tend_out[1:, :, randind_tst])), np.abs(np.float32(v_tend_out[1:, :, randind_tst]))])
            else:
                train_input_list.extend(
                    [np.float32(u_tend_out[1:, :, randind_trn]), np.float32(v_tend_out[1:, :, randind_trn])])
                test_input_list.extend(
                    [np.float32(u_tend_out[1:, :, randind_tst]), np.float32(v_tend_out[1:, :, randind_tst])])
        else:
            train_input_list.extend([np.float32(u_tend_out[1:, :, randind_trn]), np.float32(v_tend_out[1:, :, randind_trn])])
            test_input_list.extend([np.float32(u_tend_out[1:, :, randind_tst]), np.float32(v_tend_out[1:, :, randind_tst])])
       # JY: Jan 2021:I added a minus sign to the output to be consistent with the minus I add to the v input in the SH.
#         v_tend_out[:, 0:np.int(vin.shape[1] / 2), :] = -v_tend_out[:, 0:np.int(vin.shape[1] / 2), :]
#         train_input_list.extend([np.float32(u_tend_out[1:, :, randind_trn]), np.float32(v_tend_out[1:, :, randind_trn])])
#         test_input_list.extend([np.float32(u_tend_out[1:, :, randind_tst]), np.float32(v_tend_out[1:, :, randind_tst])])
        # train_input_list.append(np.float32(v_tend_out[1:, :, randind_trn])) # I do not want to include the surface level of the flux which is 0 by def
        # test_input_list.append(np.float32(v_tend_out[1:, :, randind_tst])) # I do not want to include the surface level
    if flag_dict['do_uv_surf_fluxes_out']:
        # vSurfout[0:np.int(vSurfout.shape[0] / 2), :] = -vSurfout[0:np.int(vSurfout.shape[0] / 2), :]
        train_input_list.append(np.float32(uSurfout[:, randind_trn]))
        test_input_list.append(np.float32(uSurfout[:, randind_tst]))
        train_input_list.append(np.float32(vSurfout[:, randind_trn]))
        test_input_list.append(np.float32(vSurfout[:, randind_tst]))




    train_input_list.extend([y, z, p, rho])
    test_input_list.extend([y, z, p, rho])

    pickle.dump(train_input_list, open(output_dir + expt + data_specific_description + '_training.pkl', 'wb'))
    pickle.dump(test_input_list, open(output_dir + expt + data_specific_description + '_testing.pkl', 'wb'))

def write_netcdf_rf(est_str, datasource, output_vert_vars, output_vert_dim, rain_only=False,
                    no_cos=False, use_rh=False,scale_per_column=False,
                    rewight_outputs=False,weight_list=[1,1],is_cheyenne=False,exclusion_flag=False,ind1_exc=0,ind2_exc=0):
    # Set output filename
    if is_cheyenne == False: #On aimsir/esker
        base_dir = '/net/aimsir/archive1/janniy/'
    else:
        base_dir = '/glade/scratch/janniy/'
    if exclusion_flag:
        output_filename = base_dir + 'mldata_tmp/gcm_regressors/' + est_str + str(ind1_exc) + str(ind2_exc)+ '.nc'
    else:
        output_filename = base_dir + 'mldata_tmp/gcm_regressors/'+est_str+'.nc'
    # Load rf and preprocessors
    if exclusion_flag:
        est, _, errors, f_ppi, o_ppi, f_pp, o_pp, y, z, p, rho = \
            pickle.load(open(base_dir + 'mldata_tmp/regressors/' + est_str  + str(ind1_exc) + str(ind2_exc)+ '.pkl', 'rb'))
    else:
        est, _, errors, f_ppi, o_ppi, f_pp, o_pp, y, z, p, rho = \
            pickle.load(open(base_dir + 'mldata_tmp/regressors/' + est_str + '.pkl', 'rb'))

    # determine the maximum number of nodes and the number of features/outputs
    estimators = est.estimators_
    n_trees = len(estimators)
    n_nodes = np.zeros(n_trees, dtype=np.int32)
    for itree in range(n_trees):
        tree = estimators[itree].tree_
        n_nodes[itree] = tree.node_count
    max_n_nodes = np.amax(n_nodes)
    print("Maximum number of nodes across trees:")
    print(max_n_nodes)
    print("Average number of nodes across trees:")
    print(np.mean(n_nodes))
    n_features = estimators[0].tree_.n_features
    n_outputs = estimators[0].tree_.n_outputs

    # populate arrays that describe trees
    children_left = np.zeros((max_n_nodes, n_trees), dtype=np.int32)
    children_right = np.zeros((max_n_nodes, n_trees), dtype=np.int32)
    split_feature = np.zeros((max_n_nodes, n_trees), dtype=np.int32)
    n_node_samples = np.zeros((max_n_nodes, n_trees), dtype=np.int32)
    threshold = np.zeros((max_n_nodes, n_trees), dtype=np.float32)
    values_predicted = np.zeros((n_outputs, max_n_nodes, n_trees), dtype=np.float32)  # Yani modified to reduce spave

    # note for python, slices don't include upper index!
    # inverse transform outputs here to speed up the GCM parameterization
    n_leaf_nodes = 0
    n_samples_leaf_nodes = 0
    for itree in range(n_trees):
        tree = estimators[itree].tree_
        children_left[:n_nodes[itree], itree] = tree.children_left
        children_right[:n_nodes[itree], itree] = tree.children_right
        split_feature[:n_nodes[itree], itree] = tree.feature
        threshold[:n_nodes[itree], itree] = tree.threshold
        n_node_samples[:n_nodes[itree], itree] = tree.n_node_samples
        for inode in range(n_nodes[itree]):
            # values_predicted[:,inode,itree] = np.float32(ml_load.inverse_transform_data(o_ppi, o_pp, (tree.value[inode,:]).T, z))  # Yani modified to reduce spave (float32)
            o_dict = ml_load.unpack_list((tree.value[inode, :]).T, output_vert_vars, output_vert_dim)
            values_predicted[:, inode, itree] = np.float32(
                ml_load.inverse_transform_data_generalized(o_ppi, o_pp, o_dict, output_vert_vars, z, scale_per_column,
                                                           rewight_outputs=rewight_outputs,
                                                           weight_list=weight_list))  # Makes sure we get our outputs in the correct units.

            if children_left[inode, itree] == children_right[inode, itree]:  # leaf node
                n_leaf_nodes = n_leaf_nodes + 1
                n_samples_leaf_nodes = n_samples_leaf_nodes + n_node_samples[inode, itree]

    print("Average number of leaf nodes across trees:")
    print(n_leaf_nodes / n_trees)

    # chance of not being included in bootstrap sample is (1-1/n)^n
    # which is 1/e for large n
    # note each tree has only about (1-1/e)*n_trn_exs due to bagging
    # which is 63% of them
    # only seem to keep one when there are non-unique samples
    print("Average number of samples per leaf node:")
    print(n_samples_leaf_nodes / n_leaf_nodes)

    # Grab input and output normalization
    # if f_ppi['name']=='StandardScaler':
    #  fscale_mean = f_pp.mean_
    #  fscale_stnd = f_pp.scale_
    if f_ppi['name'] != 'NoScaler':
        raise ValueError('Incorrect scaler name - Cannot treat any other case - in RF no need to')

    # Write to file
    ncfile = Dataset(output_filename, 'w', format="NETCDF3_CLASSIC")
    # Write the dimensions
    ncfile.createDimension('dim_nodes', max_n_nodes)
    ncfile.createDimension('dim_trees', n_trees)
    ncfile.createDimension('dim_features', n_features)
    ncfile.createDimension('dim_outputs', n_outputs)

    # Create variable entries in the file
    nc_n_nodes = ncfile.createVariable('n_nodes', np.dtype('int32').char, ('dim_trees'))
    nc_children_left = ncfile.createVariable('children_left', np.dtype('int32').char, ('dim_nodes', 'dim_trees'))
    nc_children_right = ncfile.createVariable('children_right', np.dtype('int32').char, ('dim_nodes', 'dim_trees'))
    nc_split_feature = ncfile.createVariable('split_feature', np.dtype('int32').char, ('dim_nodes', 'dim_trees'))
    nc_threshold = ncfile.createVariable('threshold', np.dtype('float32').char, ('dim_nodes', 'dim_trees'))
    nc_values_predicted = ncfile.createVariable('values_predicted', np.dtype('float32').char,
                                                ('dim_outputs', 'dim_nodes', 'dim_trees'))

    # Write variables and close file
    nc_n_nodes[:] = n_nodes
    nc_children_left[:] = children_left
    nc_children_right[:] = children_right
    nc_split_feature[:] = split_feature
    nc_threshold[:] = threshold
    nc_values_predicted[:] = np.float32(values_predicted)
    # nc_zdim_out_var_list[:] = output_vert_dim
    # nc_name_out_var_list[:] = str_out

    # Write global file attributes
    ncfile.description = est_str
    ncfile.close()


def write_netcdf_nn(est_str, datasource, rain_only=False, no_cos=False, use_rh=False, is_cheyenne=False):
    # Set output filename
    if is_cheyenne == False:  # On aimsir/esker
        base_dir = '/net/aimsir/archive1/janniy/'
    else:
        base_dir = '/glade/work/janniy/'

    output_filename = base_dir + 'mldata/gcm_regressors/' + est_str + '.nc'
    # Load rf and preprocessors
    est, _, errors, f_ppi, o_ppi, f_pp, o_pp, y, z, p, rho = \
        pickle.load(open(base_dir + 'mldata/regressors/' + est_str + '.pkl', 'rb'))
    # Need to transform some data for preprocessors to be able to export params
    f, o, _, _, _, _, = ml_load.LoadData(datasource,
                                         max_z=max(z),
                                         rain_only=rain_only,
                                         no_cos=no_cos,
                                         use_rh=use_rh)
    f_scl = ml_load.transform_data(f_ppi, f_pp, f, z)
    _ = ml_load.transform_data(o_ppi, o_pp, o, z)
    # Also need to use the predict method to be able to export ANN params
    _ = est.predict(f_scl)

    # Grab weights
    w1 = est.get_parameters()[0].weights
    w2 = est.get_parameters()[1].weights
    b1 = est.get_parameters()[0].biases
    b2 = est.get_parameters()[1].biases

    # Grab input and output normalization
    if f_ppi['name'] == 'StandardScaler':
        fscale_mean = f_pp.mean_
        fscale_stnd = f_pp.scale_
    else:
        raise ValueError('Incorrect scaler name')

    if o_ppi['name'] == 'SimpleO':
        Nlev = len(z)
        oscale = np.zeros(b2.shape)
        oscale[:Nlev] = 1.0 / o_pp[0]
        oscale[Nlev:] = 1.0 / o_pp[1]
    elif o_ppi['name'] == 'StandardScaler':
        oscale_mean = o_pp.mean_
        oscale_stnd = o_pp.scale_
    else:
        raise ValueError('Incorrect scaler name')

        # Write weights to file
    ncfile = Dataset(output_filename, 'w', format="NETCDF3_CLASSIC")
    # Write the dimensions
    ncfile.createDimension('N_in', w1.shape[0])
    ncfile.createDimension('N_h1', w1.shape[1])
    ncfile.createDimension('N_out', w2.shape[1])
    # Create variable entries in the file
    nc_w1 = ncfile.createVariable('w1', np.dtype('float32').char,
                                  ('N_h1', 'N_in'))  # Reverse dims
    nc_w2 = ncfile.createVariable('w2', np.dtype('float32').char,
                                  ('N_out', 'N_h1'))
    nc_b1 = ncfile.createVariable('b1', np.dtype('float32').char,
                                  ('N_h1'))
    nc_b2 = ncfile.createVariable('b2', np.dtype('float32').char,
                                  ('N_out'))
    nc_fscale_mean = ncfile.createVariable('fscale_mean',
                                           np.dtype('float32').char, ('N_in'))
    nc_fscale_stnd = ncfile.createVariable('fscale_stnd',
                                           np.dtype('float32').char, ('N_in'))
    if o_ppi['name'] == 'SimpleO':
        nc_oscale = ncfile.createVariable('oscale',
                                          np.dtype('float32').char,
                                          ('N_out'))
    else:
        nc_oscale_mean = ncfile.createVariable('oscale_mean', np.dtype('float32').char, ('N_out'))
        nc_oscale_stnd = ncfile.createVariable('oscale_stnd', np.dtype('float32').char, ('N_out'))

    # Write variables and close file - transpose because fortran reads it in
    # "backwards"
    nc_w1[:] = w1.T
    nc_w2[:] = w2.T
    nc_b1[:] = b1
    nc_b2[:] = b2
    nc_fscale_mean[:] = fscale_mean
    nc_fscale_stnd[:] = fscale_stnd

    if o_ppi['name'] == 'SimpleO':
        nc_oscale[:] = oscale
    else:
        nc_oscale_mean[:] = oscale_mean
        nc_oscale_stnd[:] = oscale_stnd

    # Write global file attributes
    ncfile.description = est_str
    ncfile.close()


def create_z_grad_plus_surf_var(variable):
    T_grad_in = np.zeros(variable.shape)
    T_grad_in[0, :, :] = variable[0, :, :]  # The surface temperature
    T_grad_in[1:variable.shape[0], :, :] = variable[1:variable.shape[0], :, :] - variable[0:variable.shape[0] - 1, :, :]
    return T_grad_in


def create_difference_from_surface(variable):
    T_s_duff = np.zeros(variable.shape)
    T_s_duff[0, :, :] = variable[0, :, :]  # The surface temperature
    for ind in range(variable.shape[0] - 1):
        print(ind)
        T_s_duff[ind + 1, :, :] = variable[0, :, :] - variable[ind + 1, :, :]
    return T_s_duff

def vertical_smooth(variable):
    var_vert_avg = np.zeros(variable.shape)
    var_vert_avg[:-1, :, :] =0.5 *( variable[:-1, :, :] + variable[1:, :, :])
    var_vert_avg[-1,:,:] = variable[-1, :, :]
    return var_vert_avg

def create_specific_data_string_desc(flag_dict):
    """create the name of the output files according to input flags
    Args:
     flag_dict (dict): including the specific configuration that we want to calculate the outputs for
    """
    if "adv_u_v_grid" in flag_dict:
        if flag_dict['adv_u_v_grid']:
            str1 = '_u_v_grid_flux'
        else:
            str1 = '_u_v_flux'
    elif 'no_c_grid' in flag_dict.keys():
        if flag_dict['no_c_grid']:
            str1 = '_no_c_grid_flux'
        else:
            str1 = '_u_v_flux'
    else:
        str1 = '_u_v_flux'
    if 'abs_mom_flux' in flag_dict.keys():
        if flag_dict['abs_mom_flux']:
            str1 = str1 + 'abs_mom'

    data_specific_description = str(flag_dict['do_dqp'])[0] + str(flag_dict['ver_adv_correct'])[0] + \
                                str(flag_dict['do_hor_wind_input'])[0] + str(flag_dict['do_ver_wind_input'])[0] + \
                                str(flag_dict['do_z_diffusion'])[0] + str(flag_dict['do_q_T_surf_fluxes'])[0] + \
                                str(flag_dict['do_surf_wind'])[0] + str(flag_dict['do_sedimentation'])[0] + \
                                str(flag_dict['do_radiation_output'])[0] + str(flag_dict['rad_level']) + \
                                str(flag_dict['do_qp_as_var'])[0] + str(flag_dict['do_fall_tend'])[0] + \
                                str(flag_dict['Tin_feature'])[0] + str(flag_dict['Tin_z_grad_feature'])[0] + \
                                str(flag_dict['qin_feature'])[0] + str(flag_dict['qin_z_grad_feature'])[0] + str(
        flag_dict['input_upper_lev']) + \
                                str(flag_dict['predict_tendencies'])[0] +  \
                                str(flag_dict['do_qp_diff_corr_to_T'])[0] + \
                                str(flag_dict['do_q_T_surf_fluxes_correction'])[0] + \
                                str(flag_dict['do_t_strat_correction'])[0] + \
                                str(flag_dict['output_precip'])[0] + str(flag_dict['do_radiation_in_Tz'])[0] + \
                                str(flag_dict['do_z_diffusion_correction'])[0] + \
                                str(flag_dict['calc_tkz_z'])[0] + str(flag_dict['calc_tkz_z_correction'])[0] + str(
        flag_dict['resolution']) + \
                                str(flag_dict['tkz_levels']) + str(flag_dict['Tin_s_diff_feature'])[0] + \
                                str(flag_dict['qin_s_diff_feature'])[0] + \
                                str(flag_dict['dist_From_eq_in'])[0] + str(flag_dict['T_instead_of_Tabs'])[0] + \
                                str(flag_dict['tabs_resolved_init'])[0] + str(flag_dict['qn_coarse_init'])[0] + \
                                str(flag_dict['qn_resolved_as_var'])[0] + str(flag_dict['sed_level']) + \
                                str(flag_dict['strat_corr_level']) + \
                                str(flag_dict['do_surf_flux_hemispheric_symmetric_correction'])[0] + \
                                str(flag_dict['Flux_for_TFULL'])[0] + '_X01' + str1

    return data_specific_description


def print_simulation_decription(filename):
    i = 4
    print('do_dqp=', filename[i])
    i = i + 1
    print('ver_adv_correct=', filename[i])
    i = i + 1
    print('do_hor_wind_input=', filename[i])
    i = i + 1
    print('do_ver_wind_input=', filename[i])
    i = i + 1
    print('do_z_diffusion=', filename[i])
    i = i + 1
    print('do_q_T_surf_fluxes=', filename[i])
    i = i + 1
    print('do_surf_wind=', filename[i])
    i = i + 1
    print('do_sedimentation=', filename[i])
    i = i + 1
    print('do_radiation_output=', filename[i])
    i = i + 1
    print('rad_level=', filename[i:i + 2])
    i = i + 2
    print('do_qp_as_var=', filename[i])
    i = i + 1
    print('do_fall_tend=', filename[i])
    i = i + 1
    print('Tin_feature=', filename[i])
    i = i + 1
    print('Tin_z_grad_feature=', filename[i])
    i = i + 1
    print('qin_feature=', filename[i])
    i = i + 1
    print('qin_z_grad_feature=', filename[i])
    i = i + 1
    print('input_upper_lev=', filename[i:i + 2])
    i = i + 2
    print('predict_tendencies=', filename[i])
    i = i + 1
    print('do_qp_diff_corr_to_T=', filename[i])
    i = i + 1
    print('do_q_T_surf_fluxes_correction=', filename[i])
    i = i + 1
    print('do_t_strat_correction=', filename[i])
    i = i + 1
    print('output_precip=', filename[i])
    i = i + 1
    print('do_radiation_in_Tz', filename[i])
    i = i + 1
    print('do_z_diffusion_correction', filename[i])
    i = i + 1
    print('calc_tkz_z=', filename[i])
    i = i + 1
    print('calc_tkz_z_correction=', filename[i])
    i = i + 1
    print('resolution=', filename[i:i + 2])
    i = i + 2
    print('tkz_levels=', filename[i:i + 2])
    i = i + 1
    print('Tin_s_diff_feature=', filename[i])
    i = i + 1
    print('qin_s_diff_feature=', filename[i])
    i = i + 1
    print['dist_From_eq_in=', filename[i]]
    i = i + 1
    print['T_instead_of_Tabs=', filename[i]]
    i = i + 1
    print['tabs_resolved_init=', filename[i]]
    i = i + 1
    print['qn_coarse_init=', filename[i]]
    i = i + 1
    print['qn_resolved_as_var=', filename[i]]
    i = i + 1
    print['strat_corr_level=', filename[i]]
    i = i + 2
    print['sed_level=', filename[i]]
