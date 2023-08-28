# v3
# find optimal battery charging regime assuming both
# optimization of demand and spot price optimization is possible

# the model considers two types of batteries: Household batteries and large-scale redox flow batteries
# The first is mostly interested in optimizing the demand-to-PV usage, the latter in optimizing spot prices

# For a given House hold capacity C_H the Household Battery usage is optimized.
# In parallel for this fixed capacity the Redox Battery capacity and its usage is optimized
# This is done on a Day-to-Day basis (assumption: daily independence)




# !!! THREADS !!!!!!
# 20Good Optimization Modeling Practices with Pyomo. June 2023
# Boosting performance
# • Threads
# Solver.options['Threads'] = int((psutil.cpu_count(logical=True) +
# psutil.cpu_count(logical=False))/2


# (got, 11.04.2023)

#import concurrent.futures
from multiprocessing import Pool
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from yearlyOptimization import start_optimization_single_year

# T = 100   #observations
# CAPACITY_MAX = 10 #in [GWh]
# FCTR_BATTERY_DEGREGATION = 0.99

MODELNAME = 'V4_2023Q2'
input_path = r'X:\Daten\Strom\lfPFC\04_Investitionsrechnung\Speicher\Model\\' + MODELNAME + r'\\'
#profit_factor = 365 # 8760 / (len(pd.read_csv(input_path + 'Inputs\\params_T_2023.csv', header=None)) - 1)
#N_HOUSEHOLDS_WITH_PV = 5e6
#BATTERY_COST_INITIAL = 40 # €/kWh
COST_DEGRESSION = 0.02 # yearly cost degression
Y_START = 2039
Y_END = 2039

# -----------------------------------------------------------------------------
# def create_csv_files(input_path, file_names):
#     input_path = input_path + r'Inputs\\'
#     df_T = pd.Series( data = range(T),name='T')
#     df_pricesSell = pd.DataFrame(data={'T': df_T,
#                                    'params_PowerPricesSell': rnd.rand(T)})
#     df_pricesBuy = pd.DataFrame(data={'T': df_T,
#                                    'params_PowerPricesBuy': df_pricesSell['PowerPricesSell'] + 0.1})
#
#     # df_capMax = pd.Series(data = [CAPACITY_MAX], name='CapacityMax')
#     df_ResLoad = pd.DataFrame(data={'T': df_T,
#                                  'params_ResLoad': 10*(rnd.rand(T) + np.sin(T/np.size(T)))})
#     #--Write--
#     df_T.to_csv(input_path + file_names['T'], index=False)
#     df_pricesBuy.to_csv(input_path + file_names['PowerPricesBuy'], index=False)
#     df_pricesSell.to_csv(input_path + file_names['PowerPricesSell'], index=False)
#     df_capMax.to_csv(input_path + file_names['CapacityMax'], index=False)
#     df_ResLoad.to_csv(input_path + file_names['ResLoad'], index=False)
# -----------------------------------------------------------------------------

def start_optimization(input_path, multithreading=False):
    '''
    Starts up the Optimization scheme for battery usage
    writes csv-output to output-Folder

    :param input_path: path of input csv-files for optimization
    :param year_list: list of years that should be optimized
    :return: list of optimized profits for each year
    '''

    year_list = range(Y_START, Y_END+1, 1)

    df_profits, years = pd.DataFrame(), []
    for y in year_list:

        try:
            print("Starting year {}: {}".format(y, datetime.now()))
            profits_singleYear = start_optimization_single_year(input_path, y, multithreading)
            print("Finished optimization for year {}: {}".format(y, datetime.now()))

            if df_profits.columns.tolist() == []:
                df_profits = pd.DataFrame(columns=profits_singleYear.columns)
            df_profits.loc[y] = profits_singleYear.sum(axis=0).tolist()

            print("Finished Total year {}: {}".format(y, datetime.now()))
        except Exception as e:
            # write results found up to now
            df_profits.to_csv(input_path + r'Outputs\Profits.csv', sep=';', mode='w')
            print(str(e))
        except ValueError as e:
            # write results found up to now
            df_profits.to_csv(input_path + r'Outputs\Profits.csv', sep=';', mode='w')
            print(str(e))
        # finally:

        years.append(y)

    return df_profits

print('finished')


# def start_optimization_MT(input_path):
#     '''
#     Ruft jeweils die funktion "start_optimization" auf mit einer Liste <year_list> von den Jahren,
#     die diese Instanz rechnen soll
#     :param input_path:
#     :return: Schreibt Parameter der Optimierung raus
#     '''
#
#     nJobs = 1
#     ystart = 2035
#     yend = 2049
#
#     for y in range(ystart, yend+1):
#         args = []
#         for iJob in range(nJobs):
#             datelist = get_datelist_singleJob(y, nJobs, iJob, nItems=7*24, nSubsample=2)
#             args.append((input_path, datelist))
#
#         with Pool(nJobs) as p:
#             results = p.starmap(start_optimization, args)
#         print(results)
if __name__ == '__main__':

    # single thread
    profits = start_optimization(input_path, multithreading=True)
    #profits.to_csv(input_path + r'Outputs\Profits.csv', sep=';', mode='w')

    # multi threaded
    # start_optimization_MT(input_path)