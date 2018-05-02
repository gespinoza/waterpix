# -*- coding: utf-8 -*-
"""
Authors: Gonzalo E. Espinoza-Dávalos
         IHE Delft 2017
Contact: g.espinoza@un-ihe.org
Repository: https://github.com/gespinoza/waterpix
Module: waterpix
"""

from __future__ import division
import datetime as dt
from warnings import filterwarnings
import pandas as pd
import netCDF4
import csv
from waterpix.functions import (calculate_first_round, calculate_second_round,
                                return_empty_df_columns, get_neighbors,
                                percolation_fit_error,
                                replace_with_closest, budyko,
                                monthly_reducer, array_interpolation)

from scipy.optimize import least_squares

np = pd.np
filterwarnings("ignore")


def run(input_nc, output_nc,
        default_thetasat=0.45, default_rootdepth=100.0,
        default_eff=0.80, min_greenpx_proportion=0.10, min_qratio=0.10,
        infz_bounds=(150.0, 15000.0), perc_fit_min_no_of_values=5,
        et_separation_no_periods=2, baseflow_filter=0.5,
        perc_fit_parms_bounds=((0.1, 4.5), (7500, 10.0)),
        tolerance_monthly_greenpx=5, tolerance_yearly_waterbal=10,
        incrunoff_propfactor_bounds=(1.0, 15.0), logfile = False):
    
    if logfile:
        logname = output_nc.split('.')[0]+'.txt'
        csv_file = open(logname, 'wb')
        writer = csv.writer(csv_file, delimiter=':')
        writer.writerow(['input_nc',input_nc])
        writer.writerow(['output_nc',output_nc])
        writer.writerow(['default_thetasat',default_thetasat])
        writer.writerow(['default_rootdepth',default_rootdepth])
        writer.writerow(['default_eff',default_eff])
        writer.writerow(['min_greenpx_proportion',min_greenpx_proportion])
        writer.writerow(['min_qratio',min_qratio])
        writer.writerow(['infz_bounds',infz_bounds])
        writer.writerow(['perc_fit_min_no_of_values',perc_fit_min_no_of_values])
        writer.writerow(['et_separation_no_periods',et_separation_no_periods])
        writer.writerow(['baseflow_filter',baseflow_filter])
        writer.writerow(['perc_fit_parms_bounds',perc_fit_parms_bounds])
        writer.writerow(['tolerance_monthly_greenpx',tolerance_monthly_greenpx])
        writer.writerow(['tolerance_yearly_waterbal',tolerance_yearly_waterbal])
        writer.writerow(['incrunoff_propfactor_bounds',incrunoff_propfactor_bounds])
        csv_file.close()
        
    '''
    Executes the main module of waterpix
    '''
    # Read file and get lat, lon, and time data
    started = dt.datetime.now()
    print 'Reading input netcdf ...'
    inp_nc = netCDF4.Dataset(input_nc, 'r')
    ncv = inp_nc.variables
    inp_crs = ncv['crs']
    inp_lat = ncv['latitude']
    inp_lon = ncv['longitude']
    inp_time = ncv['time_yyyymm']
    inp_basinb = ncv['BasinBuffer']
    # Lists
    lat_ls = list(inp_lat[:])
    lon_ls = list(inp_lon[:])
    time_ls = list(inp_time[:])
    time_dt = [pd.to_datetime(i, format='%Y%m')
               for i in time_ls]
    # Length of vectors
    lat_n = len(lat_ls)
    lon_n = len(lon_ls)
    time_n = len(time_ls)
    # Read years
    years_ls = set()
    years_ls = [i.year for i in time_dt
                if i.year not in years_ls and not years_ls.add(i.year)]
    years_n = len(years_ls)
    time_indeces = {}
    for j in range(years_n):
        temp_ls = [int(i.strftime('%Y%m')) for i in
                   pd.date_range(str(years_ls[j]) + '0101',
                                 str(years_ls[j]) + '1231', freq='MS')]
        time_indeces[years_ls[j]] = [time_ls.index(i) for i in temp_ls]

    for key in time_indeces.keys():
        if time_indeces[key] != range(time_indeces[key][0],
                                      time_indeces[key][-1] + 1):
            raise Exception('The year {0} in the netcdf file is incomplete'
                            ' or the dates are non-consecutive')
    # Create ouput NetCDF
    print 'Creating output netcdf ...'
    out_nc = netCDF4.Dataset(output_nc, 'w', format="NETCDF4")
    std_fv = -9999
    # Copy dimensions and variables (latitude, longitude, and time)
    lat_dim = out_nc.createDimension(inp_lat.standard_name, lat_n)
    lon_dim = out_nc.createDimension(inp_lon.standard_name, lon_n)
    time_dim = out_nc.createDimension('time_yyyymm', time_n)
    year_dim = out_nc.createDimension('time_yyyy', years_n)
    # Copy variables:
    # Reference system
    crs_var = out_nc.createVariable(inp_crs.standard_name, 'i', (),
                                    fill_value=std_fv)
    crs_var.standard_name = inp_crs.standard_name
    crs_var.grid_mapping_name = inp_crs.grid_mapping_name
    crs_var.crs_wkt = inp_crs.crs_wkt
    # Latitude
    lat_var = out_nc.createVariable(inp_lat.standard_name, 'f8',
                                    (inp_lat.standard_name),
                                    fill_value=inp_lat._FillValue)
    lat_var.units = inp_lat.units
    lat_var.standard_name = inp_lat.standard_name
    # Longitude
    lon_var = out_nc.createVariable(inp_lon.standard_name, 'f8',
                                    (inp_lon.standard_name),
                                    fill_value=inp_lon._FillValue)
    lon_var.units = inp_lon.units
    lon_var.standard_name = inp_lon.standard_name
    # Time (month)
    time_var = out_nc.createVariable('time_yyyymm', 'l', ('time_yyyymm'),
                                     fill_value=inp_time._FillValue)
    time_var.standard_name = inp_time.standard_name
    time_var.format = inp_time.format
    # Time (year)
    year_var = out_nc.createVariable('time_yyyy', 'l', ('time_yyyy'),
                                     fill_value=std_fv)
    year_var.standard_name = 'time_yyyy'
    year_var.format = 'yyyy'
    # FillValues
    p_fv = ncv['Precipitation_M']._FillValue
    et_fv = ncv['Evapotranspiration_M']._FillValue
    eto_fv = ncv['ReferenceET_M']._FillValue
    lai_fv = ncv['LeafAreaIndex_M']._FillValue
    swi_fv = ncv['SWI_M']._FillValue
    swio_fv = ncv['SWIo_M']._FillValue
    swix_fv = ncv['SWIx_M']._FillValue
    qratio_fv = ncv['RunoffRatio_Y']._FillValue
    rainydays_fv = ncv['RainyDays_M']._FillValue
    thetasat_fv = ncv['SaturatedWaterContent']._FillValue
    rootdepth_fv = ncv['RootDepth']._FillValue
    # Copy data
    lat_var[:] = lat_ls
    lon_var[:] = lon_ls
    time_var[:] = time_ls
    year_var[:] = years_ls
    # Create output NetCDF variables:
    # Surface runoff (monthly)
    ss_var = out_nc.createVariable('SurfaceRunoff_M', 'f8',
                                   ('time_yyyymm', 'latitude', 'longitude'),
                                   fill_value=std_fv)
    ss_var.long_name = 'Surface runoff (fast)'
    ss_var.units = 'mm/month'
    ss_var.grid_mapping = 'crs'
    # Surface runoff (yearly)
    ssy_var = out_nc.createVariable('SurfaceRunoff_Y', 'f8',
                                    ('time_yyyy', 'latitude', 'longitude'),
                                    fill_value=std_fv)
    ssy_var.long_name = 'Surface runoff (fast)'
    ssy_var.units = 'mm/year'
    ssy_var.grid_mapping = 'crs'
    # Baseflow (monthly)
    bf_var = out_nc.createVariable('Baseflow_M', 'f8',
                                   ('time_yyyymm', 'latitude', 'longitude'),
                                   fill_value=std_fv)
    bf_var.long_name = 'Baseflow (slow)'
    bf_var.units = 'mm/month'
    bf_var.grid_mapping = 'crs'
    # Baseflow (yearly)
    bfy_var = out_nc.createVariable('Baseflow_Y', 'f8',
                                    ('time_yyyy', 'latitude', 'longitude'),
                                    fill_value=std_fv)
    bfy_var.long_name = 'Baseflow (slow)'
    bfy_var.units = 'mm/year'
    bfy_var.grid_mapping = 'crs'
    # Total runoff (monthly)
    sr_var = out_nc.createVariable('TotalRunoff_M', 'f8',
                                   ('time_yyyymm', 'latitude', 'longitude'),
                                   fill_value=std_fv)
    sr_var.long_name = 'Total runoff'
    sr_var.units = 'mm/month'
    sr_var.grid_mapping = 'crs'
    # Total runoff (yearly)
    sry_var = out_nc.createVariable('TotalRunoff_Y', 'f8',
                                    ('time_yyyy', 'latitude', 'longitude'),
                                    fill_value=std_fv)
    sry_var.long_name = 'Total runoff'
    sry_var.units = 'mm/year'
    sry_var.grid_mapping = 'crs'
    # Storage change - soil moisture (monthly)
    dsm_var = out_nc.createVariable('StorageChange_M', 'f8',
                                    ('time_yyyymm', 'latitude', 'longitude'),
                                    fill_value=std_fv)
    dsm_var.long_name = 'Change in soil moisture storage'
    dsm_var.units = 'mm/month'
    dsm_var.grid_mapping = 'crs'
    # Storage change - soil moisture (yearly)
    dsmy_var = out_nc.createVariable('StorageChange_Y', 'f8',
                                     ('time_yyyy', 'latitude', 'longitude'),
                                     fill_value=std_fv)
    dsmy_var.long_name = 'Change in soil moisture storage'
    dsmy_var.units = 'mm/year'
    dsmy_var.grid_mapping = 'crs'
    # Percolation (monthly)
    per_var = out_nc.createVariable('Percolation_M', 'f8',
                                    ('time_yyyymm', 'latitude', 'longitude'),
                                    fill_value=std_fv)
    per_var.long_name = 'Percolation'
    per_var.units = 'mm/month'
    per_var.grid_mapping = 'crs'
    # Percolation (yearly)
    pery_var = out_nc.createVariable('Percolation_Y', 'f8',
                                     ('time_yyyy', 'latitude', 'longitude'),
                                     fill_value=std_fv)
    pery_var.long_name = 'Percolation'
    pery_var.units = 'mm/year'
    pery_var.grid_mapping = 'crs'
    # Supply (monthly)
    sup_var = out_nc.createVariable('Supply_M', 'f8',
                                    ('time_yyyymm', 'latitude', 'longitude'),
                                    fill_value=std_fv)
    sup_var.long_name = 'Supply'
    sup_var.units = 'mm/month'
    sup_var.grid_mapping = 'crs'
    # Supply (yearly)
    supy_var = out_nc.createVariable('Supply_Y', 'f8',
                                     ('time_yyyy', 'latitude', 'longitude'),
                                     fill_value=std_fv)
    supy_var.long_name = 'Supply'
    supy_var.units = 'mm/year'
    supy_var.grid_mapping = 'crs'
    # Green Evapotranspiration (yearly)
    etg_var = out_nc.createVariable('ETgreen_Y', 'f8',
                                    ('time_yyyy', 'latitude', 'longitude'),
                                    fill_value=std_fv)
    etg_var.long_name = 'Green evapotranspiration'
    etg_var.units = 'mm/year'
    etg_var.grid_mapping = 'crs'
    # Green Evapotranspiration (monthly)
    etgm_var = out_nc.createVariable('ETgreen_M', 'f8',
                                    ('time_yyyymm', 'latitude', 'longitude'),
                                    fill_value=std_fv)
    etgm_var.long_name = 'Green evapotranspiration m'
    etgm_var.units = 'mm/month'
    etgm_var.grid_mapping = 'crs'
    # Blue Evapotranspiration (yearly)
    etb_var = out_nc.createVariable('ETblue_Y', 'f8',
                                    ('time_yyyy', 'latitude', 'longitude'),
                                    fill_value=std_fv)
    etb_var.long_name = 'Blue evapotranspiration'
    etb_var.units = 'mm/year'
    etb_var.grid_mapping = 'crs'
    # Blue Evapotranspiration (monthly)
    etbm_var = out_nc.createVariable('ETblue_M', 'f8',
                                    ('time_yyyymm', 'latitude', 'longitude'),
                                    fill_value=std_fv)
    etbm_var.long_name = 'Blue evapotranspiration m'
    etbm_var.units = 'mm/month'
    etbm_var.grid_mapping = 'crs'
    # Rainfed pixels
    gpix_var = out_nc.createVariable('RainfedPixels_Y', 'l',
                                     ('time_yyyy', 'latitude', 'longitude'),
                                     fill_value=std_fv)
    gpix_var.long_name = 'Rainfed pixels'
    gpix_var.units = '-'
    gpix_var.grid_mapping = 'crs'
    # Round code (yearly)
    rco_var = out_nc.createVariable('RoundCode', 'l',
                                    ('time_yyyy', 'latitude', 'longitude'),
                                    fill_value=0.0)
    rco_var.grid_mapping = 'crs'
    # Root depth soil moisture (monthly)
    rdsm_var = out_nc.createVariable('RootDepthSoilMoisture_M', 'f8',
                                     ('time_yyyymm', 'latitude', 'longitude'),
                                     fill_value=rootdepth_fv)
    rdsm_var.long_name = 'Root depth soil moisture'
    rdsm_var.units = 'cm3/cm3'
    rdsm_var.grid_mapping = 'crs'
    # Calibration parameter - infiltration depth (yearly)
    infz_var = out_nc.createVariable('InfiltrationDepth_Y', 'f8',
                                     ('time_yyyy', 'latitude', 'longitude'),
                                     fill_value=std_fv)
    infz_var.long_name = 'Infiltration depth'
    infz_var.units = 'mm'
    infz_var.grid_mapping = 'crs'
    # Incremental surface runoff (monthly)
    incss_var = out_nc.createVariable('IncrementalRunoff_M', 'f8',
                                      ('time_yyyymm', 'latitude', 'longitude'),
                                      fill_value=std_fv)
    incss_var.long_name = 'Incremental runoff'
    incss_var.units = 'mm/month'
    incss_var.grid_mapping = 'crs'
    # Incremental surface runoff (yearly)
    incssy_var = out_nc.createVariable('IncrementalRunoff_Y', 'f8',
                                       ('time_yyyy', 'latitude', 'longitude'),
                                       fill_value=std_fv)
    incssy_var.long_name = 'Incremental runoff'
    incssy_var.units = 'mm/year'
    incssy_var.grid_mapping = 'crs'
    # Incremental percolation (monthly)
    incper_var = out_nc.createVariable('IncrementalPercolation_M', 'f8',
                                       ('time_yyyymm',
                                        'latitude', 'longitude'),
                                       fill_value=std_fv)
    incper_var.long_name = 'Incremental Percolation'
    incper_var.units = 'mm/month'
    incper_var.grid_mapping = 'crs'
    # Incremental percolation (yearly)
    incpery_var = out_nc.createVariable('IncrementalPercolation_Y', 'f8',
                                        ('time_yyyy', 'latitude', 'longitude'),
                                        fill_value=std_fv)
    incpery_var.long_name = 'Incremental Percolation'
    incpery_var.units = 'mm/year'
    incpery_var.grid_mapping = 'crs'
    # Water use efficiency (monthly)
    effi_var = out_nc.createVariable('WaterUseEfficiency_M', 'f8',
                                     ('time_yyyymm', 'latitude', 'longitude'),
                                     fill_value=std_fv)
    effi_var.long_name = 'Water use efficiency'
    effi_var.units = '-'
    effi_var.grid_mapping = 'crs'
    # Water use efficiency (yearly)
    effiy_var = out_nc.createVariable('WaterUseEfficiency_Y', 'f8',
                                      ('time_yyyy', 'latitude', 'longitude'),
                                      fill_value=std_fv)
    effiy_var.long_name = 'Water use efficiency'
    effiy_var.units = '-'
    effiy_var.grid_mapping = 'crs'
    # Percolation fit - parameter 'a' (yearly)
    a_var = out_nc.createVariable('a_Y', 'f8',
                                  ('time_yyyy', 'latitude', 'longitude'),
                                  fill_value=std_fv)
    a_var.grid_mapping = 'crs'
    a_var.long_name = 'a parameter in the eqn: perc = a*rdsm^b'
    # Percolation fit - parameter 'b' (yearly)
    b_var = out_nc.createVariable('b_Y', 'f8',
                                  ('time_yyyy', 'latitude', 'longitude'),
                                  fill_value=std_fv)
    b_var.long_name = 'b parameter in the eqn: perc = a*rdsm^b'
    b_var.grid_mapping = 'crs'
    # Pre-process first round
    print 'Evapotranspiration separation (blue & green)'
    budyko_v = np.vectorize(budyko)
    inp_bas_vals = np.array(inp_basinb[:])
    # Year loop
    for yyyy in years_ls:
        print '\tyear: {0}'.format(yyyy)
        yyyyi = years_ls.index(yyyy)
        ti1 = time_indeces[yyyy][0]
        ti2 = time_indeces[yyyy][-1] + 1
        # Read values & apply reducer
        p = monthly_reducer(ncv['Precipitation_M'][ti1:ti2, :, :],
                            et_separation_no_periods,
                            p_fv)
        et = monthly_reducer(ncv['Evapotranspiration_M'][ti1:ti2, :, :],
                             et_separation_no_periods,
                             et_fv)
        eto = monthly_reducer(ncv['ReferenceET_M'][ti1:ti2, :, :],
                              et_separation_no_periods,
                              eto_fv)
        # Budyko
        inp_bas_vals = np.array(inp_basinb[:])
        phi = np.where(inp_bas_vals, eto/p, np.nan)
        phi[np.isinf(phi)] = np.nan
        phi[phi == 0] = np.nan
        et_p_bk = budyko_v(phi)
        et_p_bk[np.isnan(et_p_bk)] = 0
        green_et = np.minimum(1.1*et_p_bk*p, et)
        blue_et = et - green_et
        green_et_yr = np.nansum(green_et, axis=0)
        blue_et_yr = np.nansum(blue_et, axis=0)
        blue_et_yr[inp_bas_vals==0] = np.nan
        # Store values
        etg_var[yyyyi, :, :] = green_et_yr
        etb_var[yyyyi, :, :] = blue_et_yr
        # Correction for coastal no data areas in basinmask
        blue_et_yr[np.where((green_et_yr==0) & (blue_et_yr==0))] = np.nan
        # Green pixels
        gpix_array = np.where(np.isclose(blue_et_yr, 0), 1,
                              np.where(inp_bas_vals, 0, np.nan))
        # Check percentage of green pixels
        if np.nanmean(gpix_array) < min_greenpx_proportion:
            gpix_array = np.where(blue_et_yr < np.nanpercentile(
                blue_et_yr, 100*min_greenpx_proportion),
                                  1, np.nan)
        if np.isnan(np.nanmean(gpix_array)):
            gpix_array = np.where(blue_et_yr < np.nanpercentile(
                blue_et_yr, 100*min_greenpx_proportion),
                                  1, np.nan)
        # Store green pixels
        gpix_var[yyyyi, :, :] = gpix_array
    # First round
    print 'FIRST ROUND'
    print 'Running...'
    # Year loop
    for yyyy in years_ls:
        print '\tyear: {0}'.format(yyyy)
        yyyyi = years_ls.index(yyyy)
        ti1 = time_indeces[yyyy][0]
        ti2 = time_indeces[yyyy][-1] + 1
        # Cells loops
        for loni, lati in np.ndindex(lon_n, lat_n):
            if inp_basinb[lati, loni]:
                if gpix_var[yyyyi, lati, loni] == 1:
                    # Read data
                    p = np.array(ncv['Precipitation_M'][ti1:ti2,
                                                        lati, loni])
                    et = np.array(ncv['Evapotranspiration_M'][ti1:ti2,
                                                              lati, loni])
                    lai = np.array(ncv['LeafAreaIndex_M'][ti1:ti2,
                                                          lati, loni])
                    swi = np.array(ncv['SWI_M'][ti1:ti2,
                                                lati, loni])
                    swio = np.array(ncv['SWIo_M'][ti1:ti2,
                                                  lati, loni])
                    swix = np.array(ncv['SWIx_M'][ti1:ti2,
                                                  lati, loni])
                    rainydays = np.array(ncv['RainyDays_M'][ti1:ti2,
                                                            lati, loni])
                    qratio = float(ncv['RunoffRatio_Y'][yyyyi, lati, loni])
                    # Check for NoData values
                    p[np.isclose(p, p_fv)] = np.nan
                    et[np.isclose(et, et_fv)] = np.nan
                    lai[np.isclose(lai, lai_fv)] = np.nan
                    # Check for NoData values - arrays
                    swi[np.isclose(swi, swi_fv)] = np.nan
                    if np.isnan(swi).any():
                        swi_arr = np.array(ncv['SWI_M'])
                        swi_arr[np.isclose(swi_arr, swi_fv)] = np.nan
                        swi = replace_with_closest(swi, swi_arr,
                                                   (lati, loni), (ti1, ti2))
                    swio[np.isclose(swio, swio_fv)] = np.nan
                    if np.isnan(swio).any():
                        swio_arr = np.array(ncv['SWIo_M'])
                        swio_arr[np.isclose(swio_arr, swio_fv)] = np.nan
                        swio = replace_with_closest(swio, swio_arr,
                                                    (lati, loni), (ti1, ti2))
                    swix[np.isclose(swix, swix_fv)] = np.nan
                    if np.isnan(swix).any():
                        swix_arr = np.array(ncv['SWIx_M'])
                        swix_arr[np.isclose(swix_arr, swix_fv)] = np.nan
                        swix = replace_with_closest(swix, swix_arr,
                                                    (lati, loni), (ti1, ti2))
                    if np.isclose(qratio, qratio_fv):
                        qratio_arr = np.array(ncv['RunoffRatio_Y'])
                        qratio_arr[qratio_arr < min_qratio] = min_qratio
                        qratio = replace_with_closest(qratio,
                                                      qratio_arr,
                                                      (lati, loni),
                                                      (yyyyi, yyyyi + 1))
                    elif qratio < min_qratio:
                        qratio = min_qratio
                    rainydays[np.isclose(rainydays, rainydays_fv)] = np.nan
                    if np.isnan(rainydays).any():
                        rainydays_arr = np.array(ncv['RainyDays_M'])
                        rainydays = replace_with_closest(rainydays,
                                                         rainydays_arr,
                                                         (lati, loni),
                                                         (ti1, ti2))
                    thetasat = float(ncv['SaturatedWaterContent'][lati,
                                                                  loni])
                    if np.isnan(thetasat) or thetasat == thetasat_fv:
                        thetasat = default_thetasat
                    rootdepth = float(ncv['RootDepth'][lati, loni])
                    if np.isnan(rootdepth) or rootdepth == rootdepth_fv:
                        rootdepth = default_rootdepth
                    # Dataframe
                    if not (np.isnan(swi).any() or
                            np.isnan(swio).any() or
                            np.isnan(swix).any()):
                        df = pd.DataFrame(data={'p': p, 'et': et,
                                                'lai': lai, 'swi': swi,
                                                'swio': swio, 'swix': swix,
                                                'rainydays': rainydays})
                        # Calculate first round
                        df_out, second_round = calculate_first_round(
                            df, (thetasat, rootdepth, qratio, baseflow_filter),
                            infz_bounds, tolerance_yearly_waterbal)
                    else:
                        second_round = 0
                        df_out = return_empty_df_columns(pd.DataFrame())
                    # Store values in output NetCDF
                    if not second_round:
                        ss_var[ti1:ti2,
                               lati, loni] = np.array(df_out['Qsw'])
                        bf_var[ti1:ti2,
                               lati, loni] = np.array(df_out['Qgw'])
                        sr_var[ti1:ti2,
                               lati, loni] = np.array(df_out['Qtot'])
                        dsm_var[ti1:ti2,
                                lati, loni] = np.array(df_out['dsm'])
                        per_var[ti1:ti2,
                                lati, loni] = np.array(df_out['perc'])
                        rdsm_var[ti1:ti2,
                                 lati, loni] = np.array(df_out['thetarz'])
                        infz_var[yyyyi,
                                 lati, loni] = float(df_out['infz'][0])
                        rco_var[yyyyi,
                                lati, loni] = 0
                        etbm_var[ti1:ti2,
                                lati, loni] = 0
                        etgm_var[ti1:ti2,
                                lati, loni] = np.array(df['et'])
                        sup_var[ti1:ti2,
                                lati, loni] = 0
                        incss_var[ti1:ti2,
                                lati, loni] = 0
                        incper_var[ti1:ti2,
                                lati, loni] = 0
                    else:
                        rco_var[yyyyi, lati, loni] = int(second_round)
                else:
                    second_round = 10
                    rco_var[yyyyi, lati, loni] = int(second_round)
            else:
                gpix_var[yyyyi, lati, loni] = std_fv
    # Pre-process second round
    print 'Calculating infz and rdsm-perc fits'
    infz_array_all = np.zeros((years_n, lat_n, lon_n))
    infz_array_all[:] = np.nan
    perc_fit_parms_first_guess = (np.mean([perc_fit_parms_bounds[0][0],
                                           perc_fit_parms_bounds[1][0]]),
                                  np.mean([perc_fit_parms_bounds[0][1],
                                           perc_fit_parms_bounds[1][1]]))
    for yyyy in years_ls:
        print '\tyear: {0}'.format(yyyy)
        # Time indeces
        yyyyi = years_ls.index(yyyy)
        ti1 = time_indeces[yyyy][0]
        ti2 = time_indeces[yyyy][-1] + 1
        # Estimation of infz
        infz_array_in = np.array(infz_var[yyyyi, :, :])
        infz_array_in[np.isclose(infz_array_in, std_fv)] = np.nan
        infz_array_out = array_interpolation(inp_lon[:], inp_lat[:],
                                             infz_array_in, infz_bounds[0],
                                             False)
        infz_array_all[yyyyi, :, :] = np.where(rco_var[yyyyi, :, :] > 0,
                                               infz_array_out[:, :],
                                               infz_array_in[:, :])
        # Fit rdsm and percolation
        # percolation_fit_func(yyyyi, perc_fit_min_no_of_values). Returns a,b,
        for loni, lati in np.ndindex(lon_n, lat_n):
            if rco_var[yyyyi, lati, loni] > 0:
                n_nb = 3  # minimum_neighboring_cells_offset
                rdsm_fit = []
                while len(rdsm_fit) < perc_fit_min_no_of_values:
                    tot_neighbors_ls = get_neighbors(lati, loni,
                                                     lat_n, lon_n,
                                                     n_nb)
                    # Vector with values
                    rdsm_fit = np.array(
                        [rdsm_var[ti1:ti2, y, x] for y, x in tot_neighbors_ls]
                        ).flatten()
                    perc_fit = np.array(
                        [per_var[ti1:ti2, y, x] for y, x in tot_neighbors_ls]
                        ).flatten()
                    # Remove small percolation values (~0)
                    rdsm_perc_cond = np.logical_and(perc_fit > 0.01,
                                                    rdsm_fit != std_fv)
                    rdsm_fit = rdsm_fit[rdsm_perc_cond]
                    perc_fit = perc_fit[rdsm_perc_cond]
                    n_nb += 1
                # Fit
                fit_res = least_squares(percolation_fit_error,
                                        x0=perc_fit_parms_first_guess,
                                        bounds=perc_fit_parms_bounds,
                                        args=(rdsm_fit, perc_fit),
                                        loss='soft_l1')
                # Store fit parameters in output netcdf
                a_var[yyyyi, lati, loni] = fit_res.x[0]
                b_var[yyyyi, lati, loni] = fit_res.x[1]
                infz_var[yyyyi, lati, loni] = infz_array_all[yyyyi,
                                                             lati, loni]
    # Second round
    print 'SECOND ROUND'
    print 'Running...'
    # Year loop
    for yyyy in years_ls:
        print '\tyear: {0}'.format(yyyy)
        # Time indeces
        yyyyi = years_ls.index(yyyy)
        ti1 = time_indeces[yyyy][0]
        ti2 = time_indeces[yyyy][-1] + 1
        # Cells loops
        for loni, lati in np.ndindex(lon_n, lat_n):
            if rco_var[yyyyi, lati, loni] > 0:
                # Read data
                p = np.array(ncv['Precipitation_M'][ti1:ti2,
                                                    lati, loni])
                et = np.array(ncv['Evapotranspiration_M'][ti1:ti2,
                                                          lati, loni])
                eto = np.array(ncv['ReferenceET_M'][ti1:ti2, lati, loni])
                lai = np.array(ncv['LeafAreaIndex_M'][ti1:ti2,
                                                      lati, loni])
                swi = np.array(ncv['SWI_M'][ti1:ti2,
                                            lati, loni])
                swio = np.array(ncv['SWIo_M'][ti1:ti2,
                                              lati, loni])
                swix = np.array(ncv['SWIx_M'][ti1:ti2,
                                              lati, loni])
                rainydays = np.array(ncv['RainyDays_M'][ti1:ti2,
                                                        lati, loni])
                qratio = float(ncv['RunoffRatio_Y'][yyyyi, lati, loni])
                # Check for NoData values
                p[np.isclose(p, p_fv)] = np.nan
                et[np.isclose(et, et_fv)] = np.nan
                lai[np.isclose(lai, lai_fv)] = np.nan
                # Check for NoData values - arrays
                swi[np.isclose(swi, swi_fv)] = np.nan
                if np.isnan(swi).any():
                    swi_arr = np.array(ncv['SWI_M'])
                    swi_arr[np.isclose(swi_arr, swi_fv)] = np.nan
                    swi = replace_with_closest(swi, swi_arr,
                                               (lati, loni), (ti1, ti2))
                swio[np.isclose(swio, swio_fv)] = np.nan
                if np.isnan(swio).any():
                    swio_arr = np.array(ncv['SWIo_M'])
                    swio_arr[np.isclose(swio_arr, swio_fv)] = np.nan
                    swio = replace_with_closest(swio, swio_arr,
                                                (lati, loni), (ti1, ti2))
                swix[np.isclose(swix, swix_fv)] = np.nan
                if np.isnan(swix).any():
                    swix_arr = np.array(ncv['SWIx_M'])
                    swix_arr[np.isclose(swix_arr, swix_fv)] = np.nan
                    swix = replace_with_closest(swix, swix_arr,
                                                (lati, loni), (ti1, ti2))
                if np.isclose(qratio, qratio_fv):
                    qratio_arr = np.array(ncv['RunoffRatio_Y'])
                    qratio_arr[qratio_arr < min_qratio] = min_qratio
                    qratio = replace_with_closest(qratio,
                                                  qratio_arr,
                                                  (lati, loni),
                                                  (yyyyi, yyyyi + 1))
                elif qratio < min_qratio:
                    qratio = min_qratio
                rainydays[np.isclose(rainydays, rainydays_fv)] = np.nan
                if np.isnan(rainydays).any():
                    rainydays_arr = np.array(ncv['RainyDays_M'])
                    rainydays = replace_with_closest(rainydays,
                                                     rainydays_arr,
                                                     (lati, loni), (ti1, ti2))
                thetasat = float(ncv['SaturatedWaterContent'][lati, loni])
                if np.isnan(thetasat) or thetasat == thetasat_fv:
                    thetasat = default_thetasat
                rootdepth = float(
                    ncv['RootDepth'][lati, loni])
                if np.isnan(rootdepth) or rootdepth == rootdepth_fv:
                    rootdepth = default_rootdepth
                # Additional parameters for second round
                infz = float(infz_var[yyyyi, lati, loni])
                a = float(a_var[yyyyi, lati, loni])
                b = float(b_var[yyyyi, lati, loni])
                green_et_yr = float(etg_var[yyyyi, lati, loni])
                blue_et_yr = float(etb_var[yyyyi, lati, loni])
                # Dataframe
                df = pd.DataFrame(data={'p': p, 'et': et, 'eto': eto,
                                        'lai': lai, 'swi': swi,
                                        'swio': swio, 'swix': swix,
                                        'qratio': qratio,
                                        'rainydays': rainydays})
                # Calculate second round
                df_out = calculate_second_round(df, (thetasat, rootdepth,
                                                     qratio, infz, a, b,
                                                     green_et_yr, blue_et_yr,
                                                     baseflow_filter),
                                                default_eff,
                                                tolerance_monthly_greenpx,
                                                incrunoff_propfactor_bounds)
                # Store values in output NetCDF
                ss_var[ti1:ti2,
                       lati, loni] = pd.np.array(df_out['Qsw'])
                incss_var[ti1:ti2,
                          lati, loni] = pd.np.array(df_out['delta_Qsw'])
                bf_var[ti1:ti2,
                       lati, loni] = pd.np.array(df_out['Qgw'])
                sr_var[ti1:ti2,
                       lati, loni] = pd.np.array(df_out['Qtot'])
                dsm_var[ti1:ti2,
                        lati, loni] = pd.np.array(df_out['dsm'])
                per_var[ti1:ti2,
                        lati, loni] = pd.np.array(df_out['perc'])
                incper_var[ti1:ti2,
                           lati, loni] = pd.np.array(df_out['delta_perc'])
                sup_var[ti1:ti2,
                        lati, loni] = pd.np.array(df_out['supply'])
                rdsm_var[ti1:ti2,
                         lati, loni] = pd.np.array(df_out['thetarz'])
                effi_var[ti1:ti2,
                         lati, loni] = pd.np.array(df_out['eff'])
                gpix_var[yyyyi, lati, loni] = df_out['rainfed'][0]
                etbm_var[ti1:ti2,
                         lati, loni] = pd.np.array(df_out['et_blue'])
                etgm_var[ti1:ti2,
                         lati, loni] = pd.np.array(df_out['et_green'])
    # Calculate yearly variables
    print 'Calculating values per year...'
    for yyyy in years_ls:
        # Time indeces
        yyyyi = years_ls.index(yyyy)
        ti1 = time_indeces[yyyy][0]
        ti2 = time_indeces[yyyy][-1] + 1
        # Sums used in efficiency calculation
        supply_yearly_val = np.sum(sup_var[ti1:ti2, :, :], axis=0)
        inc_ss_yearly_val = np.sum(incss_var[ti1:ti2, :, :], axis=0)
        inc_per_yearly_val = np.sum(incper_var[ti1:ti2, :, :], axis=0)
        # Store values
        ssy_var[yyyyi, :, :] = np.sum(ss_var[ti1:ti2, :, :], axis=0)
        incssy_var[yyyyi, :, :] = inc_ss_yearly_val
        bfy_var[yyyyi, :, :] = np.sum(bf_var[ti1:ti2, :, :], axis=0)
        sry_var[yyyyi, :, :] = np.sum(sr_var[ti1:ti2, :, :], axis=0)
        dsmy_var[yyyyi, :, :] = np.sum(dsm_var[ti1:ti2, :, :], axis=0)
        pery_var[yyyyi, :, :] = np.sum(per_var[ti1:ti2, :, :], axis=0)
        incpery_var[yyyyi, :, :] = inc_per_yearly_val
        supy_var[yyyyi, :, :] = supply_yearly_val
        # Water use efficiency
        effiy_var[yyyyi, :, :] = np.nanmean(effi_var[ti1:ti2, :, :], axis=0)
    # Finishing
    print 'Closing netcdf...'
    out_nc.close()
    ended = dt.datetime.now()
    print 'Time elapsed: {0}'.format(ended - started)
    # Return noutput NetCDF file location
    return output_nc
