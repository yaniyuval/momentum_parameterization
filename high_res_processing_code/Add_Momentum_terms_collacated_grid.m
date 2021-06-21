% Calc momentum subgrid terms and winds on a collocated grid
clear all
close all
clc


resolutions = [4,8,16,32];
% resolutions = [4,8];
is_cheyenne = 1;

num_of_files = 100;
index_counting = 1;
time_init = 810000+ num_of_files*450.*(index_counting-1);
time_end = min(time_init + num_of_files*450.*(index_counting), 2204550); % Changed after added more data for test


% time_init = 810450;
% time_end = time_init;

time_step = 450;
times_1 = time_init:time_step:time_end;

if is_cheyenne ==1
    exper_path = ['/glade/scratch/janniy/bill_crm_data/'];
else
    exper_path = ['/archive1/pog/bill_crm_data/'];
end

exper = {'qobs'};

cd '/glade/u/home/janniy/matlab_analysis_code/running_scripts'

filename_profile = 'sounding_z_rho_rhow.txt';
delimiterIn = ' ';
headerlinesIn = 1;
sounding = importdata(filename_profile,delimiterIn,headerlinesIn);
rhow_input = flip(sounding.data(:,3));
rho = flip(sounding.data(:,2));
zin = flip(sounding.data(:,1));

filename_profile = 'sounding_z_pres0_tabs0.txt';
sounding = importdata(filename_profile,delimiterIn,headerlinesIn);
pres0 = flip(sounding.data(:,2));
tabs0 = flip(sounding.data(:,3));
num_z = length(rho);
rhow(1:num_z,1) = rhow_input;
rhow(num_z+1,1) = 2*rhow_input(num_z)-rhow_input(num_z-1);


sprintf('Main loop')
i_exper = 1
%     parfor dummy_time = 1:length(times_1)
for dummy_time = 1:length(times_1)
    sprintf('starting measuring time')
    tStart = tic;
    tic
    time_index = times_1(dummy_time);
    cycle = 1;
    do_loop = 1;
    while do_loop
        filename_base = [exper_path, exper{i_exper}, 'km12x576/', exper{i_exper}, 'km12x576_576x1440x48_ctl_288_', num2str(time_index, '%010d'), '_000', num2str(cycle)];
        filename = [filename_base, '.nc4'];
        if exist(filename,'file')
            do_loop = 0;
        else
            cycle = cycle+1;
            if cycle>8
                disp(filename)
                error('no filename')
            end
        end
    end
    disp(filename)
    if is_cheyenne
        outfilename_janni =['/glade/scratch/janniy/ML_convection_data/', exper{i_exper},'/',exper{i_exper}, 'km12x576_576x1440x48_ctl_288_', num2str(time_index, '%010d'), '_000', num2str(cycle), '_diff_coarse_space_corrected_tkz'];
    else
        outfilename_janni =['/net/aimsir/archive1/janniy/ML_convection_data/', exper{i_exper}, 'km12x576_576x1440x48_ctl_288_', num2str(time_index, '%010d'), '_000', num2str(cycle), '_diff_coarse_space_corrected_tkz'];
    end
    sum_files = 0;
    for res_i = resolutions

        outfilename_janni_corrected  = outfilename_janni;
        outfilename_janni_tmp = [outfilename_janni_corrected, num2str(res_i), '.nc4' ];
        if exist(outfilename_janni_tmp, 'file') == 2
            sum_files = sum_files + 1; %
            sprintf('the file exist:')
            disp(outfilename_janni_tmp)
        end
    end
    
    %%%%%%%%% IF the 32 coarsening happend - I assumed I can skip all
    %%%%%%%%% files...
            outfilename_janni_tmp = [outfilename_janni_corrected, num2str(32), '.nc4' ];
        file = outfilename_janni_tmp;
        ncid = netcdf.open(file,'nowrite');
        var = 'U_NORM_GRID';
        u_variables_added = 0;
        try %Checking if
            ID = netcdf.inqVarID(ncid,var);
            u_variables_added  = 1;
            sprintf('U_NORM added')
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        
        var = 'V_NORM_GRID';
        v_variables_added = 0;
        try %Checking if
            ID = netcdf.inqVarID(ncid,var);
            v_variables_added  = 1;
            sprintf('V_NORM added')
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        var = 'U_ADV_NORM_GRID_RESOLVED';
        u_adv_variables_added = 0;
        try %Checking if
            ID = netcdf.inqVarID(ncid,var);
            u_adv_variables_added = 1;
            sprintf('U_ADV_NORM_GRID_RESOLVED')
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        var = 'U_ADV_NORM_GRID_COARSE';
        u_adv_variables_added_coarse = 0;
        try %Checking if
            ID = netcdf.inqVarID(ncid,var);
            u_adv_variables_added_coarse = 1;
            sprintf('U_ADV_NORM_GRID_COARSE')
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        
        
        
        
        v_adv_variables_added = 0;
        var_qv = 'V_ADV_NORM_GRID_RESOLVED';
        try %Checking if
            ID = netcdf.inqVarID(ncid,var_qv);
            v_adv_variables_added = 1;
            sprintf('V_ADV_NORM_GRID_RESOLVED')
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        v_adv_variables_added_coarse = 0;
        var_qv = 'V_ADV_NORM_GRID_COARSE';
        try %Checking if
            ID = netcdf.inqVarID(ncid,var_qv);
            v_adv_variables_added_coarse = 1;
            sprintf('V_ADV_NORM_GRID_COARSE')
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        u_surf_variable_added = 0;
        var_qp = 'U_SURF_FLUX_NORM_GRID_RESOLVED';
        try %Checking if
            ID = netcdf.inqVarID(ncid,var_qp);
            u_surf_variable_added  = 1;
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        u_surf_variable_added_coarse = 0;
        var_qp = 'U_SURF_FLUX_NORM_GRID_COARSE';
        try %Checking if
            ID = netcdf.inqVarID(ncid,var_qp);
            u_surf_variable_added_coarse  = 1;
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        v_surf_variable_added = 0;
        var_u = 'V_SURF_FLUX_NORM_GRID_RESOLVED';
        try %Checking if
            ID = netcdf.inqVarID(ncid,var_u);
            v_surf_variable_added = 1;
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        
        v_surf_variable_added_coarse = 0;
        var_u = 'V_SURF_FLUX_NORM_GRID_COARSE';
        try %Checking if
            ID = netcdf.inqVarID(ncid,var_u);
            v_surf_variable_added_coarse = 1;
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        netcdf.close(ncid);
        
        if u_variables_added == 1 && v_variables_added == 1  && u_adv_variables_added == 1 && v_adv_variables_added == 1  && u_surf_variable_added == 1 && v_surf_variable_added ==1
            sprintf('All vars  where already added to the 32 file, continue')
            continue;
        end
    %%%%%%%%%%
    
    
    
    
    
    sprintf('Janniy, JY: consider recalculating diffusive processes for the non-C-grid')
    
    pres = ncread(filename,'p');
    z = ncread(filename,'z');
    x = ncread(filename,'x');
    y = ncread(filename,'y');
    
    % For calculating the surface velocity.
    u_high = my_ncread(filename,'U');
    v_high = my_ncread(filename,'V');
    w_high = my_ncread(filename,'W');
    
    
    num_x = length(x);
    num_y = length(y);
    num_z = length(z);
    
    fluxbu = zeros(num_y,num_x);
    fluxbv = zeros(num_y,num_x);
    fluxbu_samson = zeros(num_y,num_x);
    fluxbv_samson = zeros(num_y,num_x);
    
    
    
    
    dz = 0.5*(z(1)+z(2));
    adzw = zeros(length(num_z));% Yani added
    for k=2:num_z
        adzw(k) = (z(k)-z(k-1))/dz;
    end
    adzw(1) = 1.;
    adzw(num_z+1) = adzw(num_z);
    adz = zeros(length(num_z));% Yani added
    
    for k=2:num_z-1
        adz(k) = 0.5*(z(k+1)-z(k-1))/dz;
    end
    adz(1) = 1.;
    adz(num_z) = adzw(num_z);
    
    
    rdz2=1./(dz*dz);
    rdz=1./dz;
    
    
    
    % Calculate coarse grained surface fluxes and advection
    
    umin = 1.0;
    cd=1.1e-3;
    wrk=(log(10/1.e-4)/log(z(1)/1.e-4))^2;
    for i=1:num_x
        for j=1:num_y
            
            if i<num_x
                ic=i+1;
            else
                ic=1;
            end
            
            if j<num_y
                jc=j+1;
            else
                jc=j;
            end
            
            if i == 1
                ib = num_x;
            else
                ib = i - 1;
            end
            
            if j == 1
                jb = 1;
            else
                jb = j -1;
            end
            
            ubot=0.5*(u_high(1,j,ic)+u_high(1,j,i));
            vbot=0.5*(v_high(1,jc,i)+v_high(1,j,i));
            windspeed=sqrt(ubot^2+vbot^2+umin^2);
            fluxbu_samson(j,i) = -rho(1)*(u_high(1,j,i))*cd*windspeed*wrk*rdz*rhow(1);
            fluxbv_samson(j,i) = -rho(1)*(v_high(1,j,i))*cd*windspeed*wrk*rdz*rhow(1);
            ubot=u_high(1,j,i);
            vbot=0.25*(v_high(1,j,i) + v_high(1,jc,i) + v_high(1,jc,ib) + v_high(1,j,ib));
            windspeed=sqrt(ubot^2+vbot^2+umin^2);
            fluxbu(j,i) = -rho(1)*(u_high(1,j,i))*cd*windspeed*wrk*rdz*rhow(1); %Note that we need to  multiply by things to get tendency (in fortran by - 1./(rho(k)*adz(k)))
            vbot=v_high(1,j,i);
            ubot=0.25*(u_high(1,j,i) + u_high(1,jb,i) + u_high(1,jb,ic) + u_high(1,j,ic));
            windspeed=sqrt(ubot^2+vbot^2+umin^2);
            fluxbv(j,i) = -rho(1)*(v_high(1,j,i))*cd*windspeed*wrk*rdz*rhow(1);
        end
    end
    
    [dudt_advect,dvdt_advect,dwdt_advect] = advect2_mom_z(num_x,num_y,num_z,dz,rho,rhow,adz,adzw,u_high,v_high,w_high);
    
    
    
    
    
    for res = resolutions
        if res == 16
            num_blocks_x = 36;
            num_blocks_y = 90;
            num_blocks_z = 48;
        elseif res == 32
            num_blocks_x = 36./2;
            num_blocks_y = 90./2;
            num_blocks_z = 48;
        elseif res == 8
            num_blocks_x = 36.*2;
            num_blocks_y = 90.*2;
            num_blocks_z = 48;
        elseif res == 4
            num_blocks_x = 36.*4;
            num_blocks_y = 90.*4;
            num_blocks_z = 48;
        end
        
        
        
        
        outfilename_janni_tmp = [outfilename_janni_corrected, num2str(res), '.nc4' ];
        file = outfilename_janni_tmp;
        ncid = netcdf.open(file,'nowrite');
        var = 'U_NORM_GRID';
        u_variables_added = 0;
        try %Checking if
            ID = netcdf.inqVarID(ncid,var);
            u_variables_added  = 1;
            sprintf('U_NORM added')
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        
        var = 'V_NORM_GRID';
        v_variables_added = 0;
        try %Checking if
            ID = netcdf.inqVarID(ncid,var);
            v_variables_added  = 1;
            sprintf('V_NORM added')
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        var = 'U_ADV_NORM_GRID_RESOLVED';
        u_adv_variables_added = 0;
        try %Checking if
            ID = netcdf.inqVarID(ncid,var);
            u_adv_variables_added = 1;
            sprintf('U_ADV_NORM_GRID_RESOLVED')
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        var = 'U_ADV_NORM_GRID_COARSE';
        u_adv_variables_added_coarse = 0;
        try %Checking if
            ID = netcdf.inqVarID(ncid,var);
            u_adv_variables_added_coarse = 1;
            sprintf('U_ADV_NORM_GRID_COARSE')
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        
        
        
        
        v_adv_variables_added = 0;
        var_qv = 'V_ADV_NORM_GRID_RESOLVED';
        try %Checking if
            ID = netcdf.inqVarID(ncid,var_qv);
            v_adv_variables_added = 1;
            sprintf('V_ADV_NORM_GRID_RESOLVED')
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        v_adv_variables_added_coarse = 0;
        var_qv = 'V_ADV_NORM_GRID_COARSE';
        try %Checking if
            ID = netcdf.inqVarID(ncid,var_qv);
            v_adv_variables_added_coarse = 1;
            sprintf('V_ADV_NORM_GRID_COARSE')
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        u_surf_variable_added = 0;
        var_qp = 'U_SURF_FLUX_NORM_GRID_RESOLVED';
        try %Checking if
            ID = netcdf.inqVarID(ncid,var_qp);
            u_surf_variable_added  = 1;
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        u_surf_variable_added_coarse = 0;
        var_qp = 'U_SURF_FLUX_NORM_GRID_COARSE';
        try %Checking if
            ID = netcdf.inqVarID(ncid,var_qp);
            u_surf_variable_added_coarse  = 1;
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        v_surf_variable_added = 0;
        var_u = 'V_SURF_FLUX_NORM_GRID_RESOLVED';
        try %Checking if
            ID = netcdf.inqVarID(ncid,var_u);
            v_surf_variable_added = 1;
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        
        v_surf_variable_added_coarse = 0;
        var_u = 'V_SURF_FLUX_NORM_GRID_COARSE';
        try %Checking if
            ID = netcdf.inqVarID(ncid,var_u);
            v_surf_variable_added_coarse = 1;
        catch exception
            if strcmp(exception.identifier,'MATLAB:imagesci:netcdf:libraryFailure')
            end
        end
        
        netcdf.close(ncid);
        
        if u_variables_added == 1 && v_variables_added == 1  && u_adv_variables_added == 1 && v_adv_variables_added == 1  && u_surf_variable_added == 1 && v_surf_variable_added ==1
            sprintf('All vars  where already added to the file, continue')
            continue;
        end
        
        u_coarse = zeros(num_z,num_blocks_y,num_blocks_x);
        v_coarse = zeros(num_z,num_blocks_y,num_blocks_x);
        w_coarse = zeros(num_z,num_blocks_y,num_blocks_x);
        
        fluxbu_coarse= zeros(num_blocks_y,num_blocks_x);
        fluxbv_coarse= zeros(num_blocks_y,num_blocks_x);
        fluxbu_resolved= zeros(num_blocks_y,num_blocks_x);
        fluxbv_resolved= zeros(num_blocks_y,num_blocks_x);
        
        fluxbu_samson_coarse= zeros(num_blocks_y,num_blocks_x);
        fluxbv_samson_coarse= zeros(num_blocks_y,num_blocks_x);
        fluxbu_samson_resolved= zeros(num_blocks_y,num_blocks_x);
        fluxbv_samson_resolved= zeros(num_blocks_y,num_blocks_x);
        
        dudt_advect_coarse = zeros(num_z,num_blocks_y,num_blocks_x);
        dvdt_advect_coarse = zeros(num_z,num_blocks_y,num_blocks_x);
        
        multiple_space = res;
        multiple_space_weight1 = (res-1)./res ;
        multiple_space_weight2 = (1)./res ;
        for i=1:num_blocks_x
            i_indices = [(i-1)*multiple_space+1:i*multiple_space];
            if i== num_blocks_x
                i_indices_x = [(i-1)*multiple_space+2:(i)*multiple_space]; 
                i_indices_x2 = [(i-1)*multiple_space+1,1];
            else
                i_indices_x = [(i-1)*multiple_space+2:(i)*multiple_space]; 
                i_indices_x2 = [(i-1)*multiple_space+1,(i)*multiple_space+1];
            end
            
            for j=1:num_blocks_y
                j_indices = [(j-1)*multiple_space+1:j*multiple_space];
                
                if j== num_blocks_y
                    j_indices_y = [(j-1)*multiple_space+2:(j)*multiple_space]; 
                    j_indices_y2 = [(j-1)*multiple_space+1];
                else
                    j_indices_y = [(j-1)*multiple_space+2:(j)*multiple_space]; 
                    j_indices_y2 = [(j-1)*multiple_space+1,(j)*multiple_space+1];
                end
            
                
                select = fluxbu_samson(j_indices,i_indices_x); %Modified x indices by shifting it
                select2 =  fluxbu_samson(j_indices,i_indices_x2);
                fluxbu_samson_coarse(j,i) = multiple_space_weight1*mean(select(:)) + multiple_space_weight2*mean(select2(:));
                
                select = fluxbu(j_indices,i_indices_x); %Modified x indices by shifting it
                select2 =  fluxbu(j_indices,i_indices_x2);
                fluxbu_coarse(j,i) = multiple_space_weight1*mean(select(:)) + multiple_space_weight2*mean(select2(:));
                
                
                select = fluxbv_samson(j_indices_y,i_indices);
                select2 =  fluxbv_samson(j_indices_y2,i_indices);
                fluxbv_samson_coarse(j,i) = multiple_space_weight1*mean(select(:)) + multiple_space_weight2*mean(select2(:));
                
                select = fluxbv(j_indices_y,i_indices);
                select2 =  fluxbv(j_indices_y2,i_indices);
                fluxbv_coarse(j,i) = multiple_space_weight1*mean(select(:)) + multiple_space_weight2*mean(select2(:));
                
                
                
                for k=1:48
                    
                    % Coarse grain such that u,v and w will be on the exact
                    % same grid.
                    select = u_high(k,j_indices,i_indices_x); %Modified x indices by shifting it
                    select2 =  u_high(k,j_indices,i_indices_x2);
                    u_coarse(k,j,i) = multiple_space_weight1*mean(select(:)) + multiple_space_weight2*mean(select2(:));
                    
                    select = dudt_advect(k,j_indices,i_indices_x); %Modified x indices by shifting it
                    select2 =  dudt_advect(k,j_indices,i_indices_x2);
                    dudt_advect_coarse(k,j,i) = multiple_space_weight1*mean(select(:)) + multiple_space_weight2*mean(select2(:));
                    
                    
                    
                    
                    
                    %                     select = v_high(i_indices,j_indices,k);
                    select = v_high(k,j_indices_y,i_indices);
                    select2 =  v_high(k,j_indices_y2,i_indices);
                    v_coarse(k,j,i) = multiple_space_weight1*mean(select(:)) + multiple_space_weight2*mean(select2(:));
                    %                     v_coarse(i,j,k) = mean(select(:));
                    
                    select = dvdt_advect(k,j_indices_y,i_indices);
                    select2 =  dvdt_advect(k,j_indices_y2,i_indices);
                    dvdt_advect_coarse(k,j,i) = multiple_space_weight1*mean(select(:)) + multiple_space_weight2*mean(select2(:));
                    
                    
                    
                    select = w_high(k,j_indices,i_indices);
                    w_coarse(k,j,i) = mean(select(:));
                end
            end
        end
        
        % 	: Surface momentum fluxes
        umin = 1.0;
        cd=1.1e-3;
        wrk=(log(10/1.e-4)/log(z(1)/1.e-4))^2;
        
        resc_fact_res = 1/res;
        resc_fact_res_2 = resc_fact_res/2;
        resc_fact_res_2_1min = 1- resc_fact_res_2;
        resc_fact_res_4 = resc_fact_res/4;
        fact1_uv = 0.25 + resc_fact_res_4;
        fact1_uv_min = 0.25 - resc_fact_res_4;
        
        for i=1:num_blocks_x
            for j=1:num_blocks_y
                
                if i<num_blocks_x
                    ic=i+1;
                else
                    ic=1;
                end
                
                if j<num_blocks_y
                    jc=j+1;
                else
                    jc=j;
                end
                
                if i == 1
                    ib = num_blocks_x;
                else
                    ib = i - 1;
                end
                
                if j == 1
                    jb = 1;
                else
                    jb = j -1;
                end
                
                ubot=(resc_fact_res_2 * u_coarse(1,j,ic)+resc_fact_res_2_1min * u_coarse(1,j,i)); % Trying to shift 6km like in SAM
                vbot=(resc_fact_res_2 * v_coarse(1,jc,i)+resc_fact_res_2_1min * v_coarse(1,j,i));
                windspeed=sqrt(ubot^2+vbot^2+umin^2);
                fluxbu_samson_resolved(j,i) = -rho(1)*(u_coarse(1,j,i))*cd*windspeed*wrk*rdz*rhow(1);
                fluxbv_samson_resolved(j,i) = -rho(1)*(v_coarse(1,j,i))*cd*windspeed*wrk*rdz*rhow(1);
                ubot=u_coarse(1,j,i);
                vbot=v_coarse(1,j,i);
                windspeed=sqrt(ubot^2+vbot^2+umin^2);
                fluxbu_resolved(j,i) = -rho(1)*(u_coarse(1,j,i))*cd*windspeed*wrk*rdz*rhow(1); %Note that we need to  multiply by things to get tendency (in fortran by - 1./(rho(k)*adz(k)))
                vbot=v_coarse(1,j,i);
                ubot=u_coarse(1,j,i);
                windspeed=sqrt(ubot^2+vbot^2+umin^2);
                fluxbv_resolved(j,i) = -rho(1)*(v_coarse(1,j,i))*cd*windspeed*wrk*rdz*rhow(1);
            end
        end
        
        % 	: vertical momentum advection
        [dudt_advect_resolved,dvdt_advect_resolved,dwdt_advect_resolved] = advect2_mom_z_no_cgrid(num_blocks_x,num_blocks_y,num_z,dz,rho,rhow,adz,adzw,u_coarse,v_coarse,w_coarse);
        
        
        
        % Calculating the subgrid terms...
        
        dudt_advect_resolved = dudt_advect_coarse - dudt_advect_resolved;
        dvdt_advect_resolved = dvdt_advect_coarse - dvdt_advect_resolved;
        
        fluxbu_resolved = fluxbu_coarse - fluxbu_resolved;
        fluxbu_samson_resolved = fluxbu_samson_coarse - fluxbu_samson_resolved;
        
        fluxbv_resolved = fluxbv_coarse - fluxbv_resolved;
        fluxbv_samson_resolved = fluxbv_samson_coarse - fluxbv_samson_resolved;
        
        perm = [3 2 1];
        w_coarse_flip = permute(w_coarse, perm);
        u_coarse_flip = permute(u_coarse, perm);
        v_coarse_flip = permute(v_coarse, perm);
        
        dudt_advect_resolved_flip = permute(dudt_advect_resolved, perm);
        dvdt_advect_resolved_flip = permute(dvdt_advect_resolved, perm);
        
        dudt_advect_coarse_flip = permute(dudt_advect_coarse, perm);
        dvdt_advect_coarse_flip = permute(dvdt_advect_coarse, perm);
        
        perm2d = [2 1];
        fluxbu_resolved_flip = permute(fluxbu_resolved, perm2d);
        fluxbv_resolved_flip= permute(fluxbv_resolved, perm2d);
        
        fluxbu_coarse_flip = permute(fluxbu_coarse, perm2d);
        fluxbv_coarse_flip= permute(fluxbv_coarse, perm2d);
        
        
        fluxbu_samson_resolved_flip = permute(fluxbu_samson_resolved, perm2d);
        fluxbv_samson_resolved_flip= permute(fluxbv_samson_resolved, perm2d);
        
        fluxbu_samson_coarse_flip = permute(fluxbu_samson_coarse, perm2d);
        fluxbv_samson_coarse_flip= permute(fluxbv_samson_coarse, perm2d);
        
        
        outfilename_janni_tmp = [outfilename_janni_corrected, num2str(res), '.nc4' ];
        disp(outfilename_janni_tmp)
        ncid = netcdf.open(outfilename_janni_tmp,'WRITE');
        
        
        xdimid = netcdf.inqDimID(ncid,'x');
        ydimid = netcdf.inqDimID(ncid,'y');
        zdimid = netcdf.inqDimID(ncid,'z');
        if u_variables_added == 0
            u_coarse_varid_init = netcdf.defVar(ncid,'U_NORM_GRID','NC_FLOAT',[xdimid ydimid zdimid]);
            netcdf.putAtt(ncid,u_coarse_varid_init,'units','m/s');
            netcdf.putVar(ncid,u_coarse_varid_init,u_coarse_flip);
        end
        
        if v_variables_added == 0
            v_coarse_varid_init = netcdf.defVar(ncid,'V_NORM_GRID','NC_FLOAT',[xdimid ydimid zdimid]);
            netcdf.putAtt(ncid,v_coarse_varid_init,'units','m/s');
            netcdf.putVar(ncid,v_coarse_varid_init,v_coarse_flip);
        end
        
        if u_adv_variables_added == 0
            u_adv_varid_init = netcdf.defVar(ncid,'U_ADV_NORM_GRID_RESOLVED','NC_FLOAT',[xdimid ydimid zdimid]);
            netcdf.putAtt(ncid,u_adv_varid_init,'units','m/s^2');
            netcdf.putVar(ncid,u_adv_varid_init,dudt_advect_resolved_flip);
        end
        
        if u_adv_variables_added_coarse == 0
            u_adv_varid_init_coarse = netcdf.defVar(ncid,'U_ADV_NORM_GRID_COARSE','NC_FLOAT',[xdimid ydimid zdimid]);
            netcdf.putAtt(ncid,u_adv_varid_init_coarse,'units','m/s^2');
            netcdf.putVar(ncid,u_adv_varid_init_coarse,dudt_advect_coarse_flip);
        end
        
        
        if v_adv_variables_added == 0
            v_adv_varid_init = netcdf.defVar(ncid,'V_ADV_NORM_GRID_RESOLVED','NC_FLOAT',[xdimid ydimid zdimid]);
            netcdf.putAtt(ncid,v_adv_varid_init,'units','m/s^2');
            netcdf.putVar(ncid,v_adv_varid_init,dvdt_advect_resolved_flip);
        end
        
        if v_adv_variables_added_coarse == 0
            v_adv_varid_init_coarse = netcdf.defVar(ncid,'V_ADV_NORM_GRID_COARSE','NC_FLOAT',[xdimid ydimid zdimid]);
            netcdf.putAtt(ncid,v_adv_varid_init_coarse,'units','m/s^2');
            netcdf.putVar(ncid,v_adv_varid_init_coarse,dvdt_advect_coarse_flip);
        end
        
        
        if u_surf_variable_added == 0
            u_surf_flux_varid_init = netcdf.defVar(ncid,'U_SURF_FLUX_NORM_GRID_RESOLVED','NC_FLOAT',[xdimid ydimid]);
            u_surf_flux_sam_varid_init  = netcdf.defVar(ncid,'U_SURF_FLUX_NORM_GRID_SAM_RESOLVED','NC_FLOAT',[xdimid ydimid]);
            netcdf.putAtt(ncid,u_surf_flux_varid_init,'units','m/s^2 * kg/m^3');
            netcdf.putAtt(ncid,u_surf_flux_sam_varid_init,'units','m/s^2 * kg/m^3');
            
            netcdf.putVar(ncid,u_surf_flux_varid_init,fluxbu_resolved_flip);
            netcdf.putVar(ncid,u_surf_flux_sam_varid_init,fluxbu_samson_resolved_flip);
        end
        
        
        if u_surf_variable_added_coarse == 0
            u_surf_flux_varid_init_coarse = netcdf.defVar(ncid,'U_SURF_FLUX_NORM_GRID_COARSE','NC_FLOAT',[xdimid ydimid]);
            u_surf_flux_sam_varid_init_coarse  = netcdf.defVar(ncid,'U_SURF_FLUX_NORM_GRID_SAM_COARSE','NC_FLOAT',[xdimid ydimid]);
            netcdf.putAtt(ncid,u_surf_flux_varid_init_coarse,'units','m/s^2 * kg/m^3');
            netcdf.putAtt(ncid,u_surf_flux_sam_varid_init_coarse,'units','m/s^2 * kg/m^3');
            
            netcdf.putVar(ncid,u_surf_flux_varid_init_coarse,fluxbu_coarse_flip);
            netcdf.putVar(ncid,u_surf_flux_sam_varid_init_coarse,fluxbu_samson_coarse_flip);
        end
        
        
        if v_surf_variable_added == 0
            v_surf_flux_varid_init  = netcdf.defVar(ncid,'V_SURF_FLUX_NORM_GRID_RESOLVED','NC_FLOAT',[xdimid ydimid]);
            v_surf_flux_sam_varid_init  = netcdf.defVar(ncid,'V_SURF_FLUX_NORM_GRID_SAM_RESOLVED','NC_FLOAT',[xdimid ydimid]);
            netcdf.putAtt(ncid,v_surf_flux_varid_init,'units','m/s^2 * kg/m^3');
            netcdf.putAtt(ncid,v_surf_flux_sam_varid_init,'units','m/s^2 * kg/m^3');
            
            netcdf.putVar(ncid,v_surf_flux_varid_init,fluxbv_resolved_flip);
            netcdf.putVar(ncid,v_surf_flux_sam_varid_init,fluxbv_samson_resolved_flip);
        end
        
        if v_surf_variable_added_coarse == 0
            v_surf_flux_varid_init_coarse = netcdf.defVar(ncid,'V_SURF_FLUX_NORM_GRID_COARSE','NC_FLOAT',[xdimid ydimid]);
            v_surf_flux_sam_varid_init_coarse  = netcdf.defVar(ncid,'V_SURF_FLUX_NORM_GRID_SAM_COARSE','NC_FLOAT',[xdimid ydimid]);
            netcdf.putAtt(ncid,v_surf_flux_varid_init_coarse,'units','m/s^2 * kg/m^3');
            netcdf.putAtt(ncid,v_surf_flux_sam_varid_init_coarse,'units','m/s^2 * kg/m^3');
            
            netcdf.putVar(ncid,v_surf_flux_varid_init_coarse,fluxbv_coarse_flip);
            netcdf.putVar(ncid,v_surf_flux_sam_varid_init_coarse,fluxbv_samson_coarse_flip);
        end
        
        
        
        
        % close output netcdf
        netcdf.close(ncid);
        
    end
end





