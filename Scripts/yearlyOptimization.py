import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# from weeklyOptimization import start_optimization_single_week
from weeklyOptimization_Sweeping import start_optimization_single_week

OPTIMIZATION_NUM_ITEMS = 24 * 7
N_INTERVALS = int(8760 / OPTIMIZATION_NUM_ITEMS)
FCTR_SUBSAMPLE_DATA = 2 # take every nth time interval to speed up stuff
def start_optimization_single_year(input_path, y, datelist):
    # create_csv_files(input_path, file_names)
    profits_singleYear = pd.DataFrame()
    fill_factor_HH_Year, fill_factor_RF_lo_Year, fill_factor_RF_hi_Year = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    feedInHH_Year, feedOutHH_Year, feedInRF_lo_Year = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    feedOutRF_lo_Year, feedInRF_hi_Year, feedOutRF_hi_Year = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    spotPrices_Year, capacity_RF_Excess_Year, capacity_RF = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    file_names = dict(PowerPricesBuy='params_PowerPricesBuy_' + str(y) + '.csv',
                      PowerPricesSell='params_PowerPricesSell_' + str(y) + '.csv',
                      SpotPrices = 'params_SpotPrices_' + str(y) + '.csv',
                      ResLoad='params_ResLoad_' + str(y) + '.csv',
                      T='params_T_' + str(y) + '.csv',
                      Capacities_HH='params_Capacities_HH_' + str(y) + '.csv',
                      CapacityPrices_HH=f'params_CapacityPrices_Household_{y}.csv',
                      CapacityPrices_RF=f'params_CapacityPrices_Redox_{y}.csv',
                      Capacity_RF_LFP=f'params_Capacity_RF_LFP_{y}.csv')
    try:
        # iOpt: index of time Interval to optimize (days, weeks, ...)
        #for iOpt in range(0, N_INTERVALS, FCTR_SUBSAMPLE_DATA):
        for d in datelist:
            (profits_singleDay,
             spotPrices_singleDay,
             capacity_RF_Excess_singleDay,
             capacity_RF_singleDay,
             fill_factor_HH_day,
             fill_factor_RF_lo_day,
             fill_factor_RF_hi_day,
             feedInHH_day,
             feedOutHH_day,
             feedInRF_lo_day,
             feedOutRF_lo_day,
             feedInRF_hi_day,
             feedOutRF_hi_day) = start_optimization_single_week(input_path, file_names, y, d, OPTIMIZATION_NUM_ITEMS)

            profits_singleYear = pd.concat([profits_singleYear, profits_singleDay], axis=0)
            spotPrices_Year = pd.concat([spotPrices_Year, spotPrices_singleDay], axis=0)
            capacity_RF_Excess_Year = pd.concat([capacity_RF_Excess_Year, capacity_RF_Excess_singleDay], axis=0)
            capacity_RF = pd.concat([capacity_RF, capacity_RF_singleDay], axis=0)
            fill_factor_HH_Year = pd.concat([fill_factor_HH_Year, fill_factor_HH_day], axis=0)
            fill_factor_RF_lo_Year = pd.concat([fill_factor_RF_lo_Year, fill_factor_RF_lo_day], axis=0)
            fill_factor_RF_hi_Year = pd.concat([fill_factor_RF_hi_Year, fill_factor_RF_hi_day], axis=0)
            feedInHH_Year = pd.concat([feedInHH_Year, feedInHH_day], axis=0)
            feedOutHH_Year = pd.concat([feedOutHH_Year, feedOutHH_day], axis=0)
            feedInRF_lo_Year = pd.concat([feedInRF_lo_Year, feedInRF_lo_day], axis=0)
            feedOutRF_lo_Year = pd.concat([feedOutRF_lo_Year, feedOutRF_lo_day], axis=0)
            feedInRF_hi_Year = pd.concat([feedInRF_hi_Year, feedInRF_hi_day], axis=0)
            feedOutRF_hi_Year = pd.concat([feedOutRF_hi_Year, feedOutRF_hi_day], axis=0)

    except Exception as e:
        # pass
        print(str(e))
    finally:
        profits_singleYear.to_csv(
            input_path + r'Outputs\profits_yearly_' + str(y) + '.csv', sep=';')
        spotPrices_Year.to_csv(
            input_path + r'Outputs\spotPrices_' + str(y) + '.csv', sep=';')
        capacity_RF_Excess_Year.to_csv(
            input_path + r'Outputs\capacity_RF_Excess_' + str(y) + '.csv', sep=';')
        capacity_RF.to_csv(
            input_path + r'Outputs\capacity_RF_' + str(y) + '.csv', sep=';')
        fill_factor_HH_Year.to_csv(
            input_path + r'Outputs\fill_factor_HH_' + str(y) + '.csv', sep=';')
        fill_factor_RF_lo_Year.to_csv(
            input_path + r'Outputs\fill_factor_RF_lo_' + str(y) + '.csv', sep=';')
        fill_factor_RF_hi_Year.to_csv(
            input_path + r'Outputs\fill_factor_RF_hi_' + str(y) + '.csv', sep=';')
        feedInHH_Year.to_csv(
            input_path + r'Outputs\feedInHH_' + str(y) + '.csv', sep=';')
        feedOutHH_Year.to_csv(
            input_path + r'Outputs\feedOutHH_' + str(y) + '.csv', sep=';')
        feedInRF_lo_Year.to_csv(
            input_path + r'Outputs\feedInRF_lo_' + str(y) + '.csv', sep=';')
        feedOutRF_lo_Year.to_csv(
            input_path + r'Outputs\feedOutRF_lo_' + str(y) + '.csv', sep=';')
        feedInRF_hi_Year.to_csv(
            input_path + r'Outputs\feedInRF_hi_' + str(y) + '.csv', sep=';')
        feedOutRF_hi_Year.to_csv(
            input_path + r'Outputs\feedOutRF_hi_' + str(y) + '.csv', sep=';')

        return profits_singleYear