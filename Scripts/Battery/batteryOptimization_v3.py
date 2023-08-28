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

import concurrent.futures
import requests
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from numpy import random as rnd
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

T = 100   #observations
# CAPACITY_MAX = 10 #in [GWh]
# FCTR_BATTERY_DEGREGATION = 0.99

MODELNAME = 'V3_WithRedox'
input_path = r'X:\Daten\Strom\lfPFC\04_Investitionsrechnung\Speicher\Model\\' + MODELNAME + r'\\'
#profit_factor = 365 # 8760 / (len(pd.read_csv(input_path + 'Inputs\\params_T_2023.csv', header=None)) - 1)
#N_HOUSEHOLDS_WITH_PV = 5e6
#BATTERY_COST_INITIAL = 40 # €/kWh
COST_DEGRESSION = 0.02 # yearly cost degression
Y_START = 2040
Y_END = 2040

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
#-----------------------------------------------------------------------------


exe = os.path.join("X:\\", "Analysen", "Tools", "OpenSolver2.9.4_Beta_LinearWin", "Solvers", "win64", "ipopt", "ipopt")

def load_data(file_path, file_names, model, d: int):
    #dict_T = {'T': {None: t_index}}
    #data.load(dict_T, set=model.T)

    data = pyo.DataPortal()
    data.load(filename=file_path + file_names["T"], set=model.T)
    #data.load(filename=file_path + file_names["ResLoad"], param=model.ResLoad)
    data.load(filename=file_path + file_names["ResLoad"], param=model.ResLoad)
    data.load(filename=file_path + file_names["PowerPricesBuy"], param=model.PowerPricesBuy)
    data.load(filename=file_path + file_names["PowerPricesSell"], param=model.PowerPricesSell)
    data.load(filename=file_path + file_names["SpotPrices"], param=model.SpotPrices)
    data.load(filename=file_path + file_names["CapacityPrices_HH"], param=model.capacityPrice_HH)
    data.load(filename=file_path + file_names["CapacityPrices_RF"], param=model.capacityPrice_RF)
    data.load(filename=file_path + file_names["Capacity_RF_LFP"], param=model.cap_RF_LFP)

    t_index = [t for t in range(d*24, (d+1)*24)]
    data._data[None]["T_daily"] = {None: t_index}

    #data.load(filename= file_path + file_names["CapacityMax"], param=model.capMax)
    #data.load(filename=file_path + file_names["StorageStartVolumes"], param=model.StorageStartVolumes)
    #data.load(filename=file_path + file_names["StorageEndVolumeMin"], param=model.StorageEndVolumeMin)
    #data.load(filename=file_path + file_names["StorageMaxVolumes"], param=model.StorageMaxVolumes, format='array')
    #data.load(filename=file_path + file_names["StorageMaxExtractionFactor"], param=model.StorageMaxExtractionFactor)
    return data


def get_model(input_path, file_names, y, capacity_HH):
    '''
    inputs:
        input_path: repository where input files are stored
        file_names: Dictionary with file names
        y: Year of optimization
        d: Day of optimization
        capacity: maximum Capacity as constraint for in- and out-flows

    optimization problem: maximize total profit from battery for given timeframe T
                          Both optimizing own demand and spot price fluctuations are assumed possible
    Variables:
        x_in_HH: battery charging with power from own solar  -> profit = -x_in_HH(t)* p_sell(t)
        x_out_HH: battery discharging for power to avoid buying power -> profit =  + x_out_HH(t)* p_buy(t)
        x_in_RF_lo: battery charging with power from market that is included in LFP assumptions -> will be included in spot price
        x_out_RF: battery charging with power from market -> + x_out_RF(t)* p_sell(t)
        x_in_RF_hi: battery charging with power from market that is in EXCESS of LFP assumptions-> will shift spot price
    Params:
        x_tot(t): charging status -> x_tot(t) = sum(t'<t){x_in_HH(t')+x_in_RF(t')-x_out_HH(t') - x_out_RF(t')}
        C: battery capacity -> x_tot(t) < C for all t in T
        RL(t): Residual load profile ->x_in_HH(t)<-RL(t), x_out_HH(t)<+RL(t)
        p_buy(t): price to pay to buy electricity
        p_sell(t): price awarded to sell electricity
    '''

    #parameters
    # dt_df = pd.read_csv(input_path + file_names['T'])
    model = pyo.AbstractModel()
    #model.T = pyo.Set(domain=pyo.NonNegativeReals, initialize=dt_df)
    #####
    ## model.T_daily = pyo.Set()
    #tRange = range(d*24, (d+1)*24)

    model.T = pyo.Set()
    model.T_daily = pyo.Set() #pyo.Set(initialize=[t for t in tRange])
    #####

    #model.X = pyo.Set()

    model.ResLoad = pyo.Param(model.T, domain=pyo.Reals)
    model.PowerPricesBuy = pyo.Param(model.T, domain= pyo.Reals)
    model.PowerPricesSell = pyo.Param(model.T, domain=pyo.Reals)
    model.SpotPrices = pyo.Param(model.T, domain=pyo.Reals)
    model.capacityPrice_HH = pyo.Param(domain=pyo.NonNegativeReals, default=0) #BATTERY_COST_INITIAL*(1-COST_DEGRESSION)**(y-Y_START))
    model.capacityPrice_RF = pyo.Param(domain=pyo.NonNegativeReals, default=0)
    model.capMax_HH = pyo.Param(domain=pyo.NonNegativeReals, default=capacity_HH) # kwh per Household-> MWh total
    model.cap_RF_LFP = pyo.Param(domain=pyo.NonNegativeReals)
    model.cap_RF_Excess = pyo.Var(domain=pyo.NonNegativeReals, initialize=capacity_HH) # initiale guess: same as HH
    model.x_in_HH = pyo.Var(model.T_daily, domain=pyo.NonNegativeReals, initialize=0)
    model.x_out_HH = pyo.Var(model.T_daily, domain=pyo.NonNegativeReals, initialize=0)
    model.x_in_RF_lo = pyo.Var(model.T_daily, domain=pyo.NonNegativeReals, initialize=0)
    model.x_out_RF_lo = pyo.Var(model.T_daily, domain=pyo.NonNegativeReals, initialize=0)
    model.x_in_RF_hi = pyo.Var(model.T_daily, domain=pyo.NonNegativeReals, initialize=0) # more than assumed in LFP -> shift price
    model.x_out_RF_hi = pyo.Var(model.T_daily, domain=pyo.NonNegativeReals, initialize=0)

    # model.nbCycles = pyo.Var(model.T, domain = pyo.Integers, initialize=0)

    # constraints:
    # x_in>0 , x_out>0, sum(t<t0)(x_in[t]-x_out[t]) < C for all t0<T

    # --------constraints for Household:  variables xin_HH, xout_HH
    def constraint_inFlow_rule_HHa(m, t):
        return m.x_in_HH[t] >= 0
    model.constraint_inFlow_rule_HHa = pyo.Constraint(model.T_daily, rule=constraint_inFlow_rule_HHa)

    def constraint_inFlow_rule_HHb(m, t):
        return m.x_in_HH[t] <= max(-m.ResLoad[t], 0.0001)
    model.constraint_inFlow_rule_HHb = pyo.Constraint(model.T_daily, rule=constraint_inFlow_rule_HHb)

    def constraint_inFlow_rule_HHc(m, t):
        return m.x_out_HH[t] >= 0
    model.constraint_inFlow_rule_HHc = pyo.Constraint(model.T_daily, rule=constraint_inFlow_rule_HHc)

    def constraint_inFlow_rule_HHd(m, t):
        return m.x_out_HH[t] <= max(m.ResLoad[t], 0.0001)
    model.constraint_inFlow_rule_HHd = pyo.Constraint(model.T_daily, rule=constraint_inFlow_rule_HHd)

    # --------constraints for Redox:  variables xin_RF, xout_RF
    def constraint_inFlow_rule2a_lo(m, t):
        return m.x_in_RF_lo[t] >= 0
    model.constraint_inFlow_rule2a_lo = pyo.Constraint(model.T_daily, rule=constraint_inFlow_rule2a_lo)

    def constraint_outFlow_rule2b_lo(m, t):
        return m.x_out_RF_lo[t] >= 0
    model.constraint_outFlow_rule2b_lo = pyo.Constraint(model.T_daily, rule=constraint_outFlow_rule2b_lo)

    def constraint_inFlow_rule2a_hi(m, t):
        return m.x_in_RF_hi[t] >= 0
    model.constraint_inFlow_rule2a_hi = pyo.Constraint(model.T_daily, rule=constraint_inFlow_rule2a_hi)

    def constraint_outFlow_rule2b_hi(m, t):
        return m.x_out_RF_hi[t] >= 0
    model.constraint_outFlow_rule2b_hi = pyo.Constraint(model.T_daily, rule=constraint_outFlow_rule2b_hi)

    # ---------constraints by capacity
    def constraint_inFlow_ruleHH(m, t):
        #ncycles = (sum(m.x_in_HH[tt] + m.x_in_RF[tt] for tt in m.T if tt < t+1) / m.capMax)
        return m.x_in_HH[t] <= m.capMax_HH #* pyo.exp(ncycles * pyo.log(FCTR_BATTERY_DEGREGATION))
    model.constraint_inFlow_ruleHH = pyo.Constraint(model.T_daily, rule=constraint_inFlow_ruleHH)
    def constraint_inFlow_ruleRF_lo(m, t):
        return m.x_in_RF_lo[t] <= m.cap_RF_LFP #* pyo.exp(ncycles * pyo.log(FCTR_BATTERY_DEGREGATION))
    model.constraint_inFlow_ruleRF_lo = pyo.Constraint(model.T_daily, rule=constraint_inFlow_ruleRF_lo)

    def constraint_inFlow_ruleRF_hi(m, t):
        return m.x_in_RF_hi[t] <= m.cap_RF_Excess #* pyo.exp(ncycles * pyo.log(FCTR_BATTERY_DEGREGATION))
    model.constraint_inFlow_ruleRF_hi = pyo.Constraint(model.T_daily, rule=constraint_inFlow_ruleRF_hi)

    def capacity_rule_HH(m, tend):
        # Approximate number of cycles by sum(x_in(t))/capMax
        #ncycles = (sum(m.x_in_HH[t] + m.x_in_RF[t] for t in m.T if t < tend+1) / m.capMax)
        return sum((m.x_in_HH[t] - m.x_out_HH[t]) for t in m.T_daily if t < tend + 1) <= \
               m.capMax_HH #* pyo.exp(ncycles * pyo.log(FCTR_BATTERY_DEGREGATION))
    model.Constraint_capacityHH = pyo.Constraint(model.T_daily, rule=capacity_rule_HH)

    def capacity_rule_RF_lo(m, tend):
        # Approximate number of cycles by sum(x_in(t))/capMax
        #ncycles = (sum(m.x_in_HH[t] + m.x_in_RF[t] for t in m.T if t < tend+1) / m.capMax)
        return sum((m.x_in_RF_lo[t] - m.x_out_RF_lo[t]) for t in m.T_daily if t < tend + 1) <= \
               m.cap_RF_LFP #* pyo.exp(ncycles * pyo.log(FCTR_BATTERY_DEGREGATION))
    model.capacity_rule_RF_lo = pyo.Constraint(model.T_daily, rule=capacity_rule_RF_lo)
    def capacity_rule_RF_hi(m, tend):
        # Approximate number of cycles by sum(x_in(t))/capMax
        #ncycles = (sum(m.x_in_HH[t] + m.x_in_RF[t] for t in m.T if t < tend+1) / m.capMax)
        return sum((m.x_in_RF_hi[t] - m.x_out_RF_hi[t]) for t in m.T_daily if t < tend + 1) <= \
               m.cap_RF_Excess #* pyo.exp(ncycles * pyo.log(FCTR_BATTERY_DEGREGATION))
    model.capacity_rule_RF_hi = pyo.Constraint(model.T_daily, rule=capacity_rule_RF_hi)
    def neutrality_rule_HH(m):
        return sum((m.x_in_HH[t] - m.x_out_HH[t]) for t in m.T_daily) == 0
    model.neutrality_rule_HH = pyo.Constraint(rule=neutrality_rule_HH)
    def neutrality_rule_RF_lo(m):
        return sum((m.x_in_RF_lo[t] - m.x_out_RF_hi[t]) for t in m.T_daily) == 0
    model.neutrality_rule_RF_lo = pyo.Constraint(rule=neutrality_rule_RF_lo)

    def neutrality_rule_RF_hi(m):
        return sum((m.x_in_RF_hi[t] - m.x_out_RF_hi[t]) for t in m.T_daily) == 0
    model.neutrality_rule_RF_hi = pyo.Constraint(rule=neutrality_rule_RF_hi)

    def positive_charge_rule_HH(m, tend):
        return sum((m.x_in_HH[t] - m.x_out_HH[t]) for t in m.T_daily if t < tend + 1) >= 0
    model.positive_charge_rule_HH = pyo.Constraint(model.T_daily, rule=positive_charge_rule_HH)

    def positive_charge_rule_RF(m, tend):
        return sum((m.x_in_RF_lo[t] + m.x_in_RF_hi[t] - m.x_out_RF_lo[t] - m.x_out_RF_hi[t]) for t in m.T_daily if t < tend + 1) >= 0
    model.positive_charge_rule_RF = pyo.Constraint(model.T_daily, rule=positive_charge_rule_RF)

    def binary_charge_rule_HH(m, t):
        return (m.x_in_HH[t]) * (m.x_out_HH[t]) <= 100
    model.binary_charge_rule_HH = pyo.Constraint(model.T_daily, rule=binary_charge_rule_HH)
    def binary_charge_rule_RF(m, t):
        # both in or both out
        return (m.x_in_RF_lo[t] + m.x_in_RF_hi[t]) * (m.x_in_RF_lo[t] + m.x_in_RF_hi[t]) <= 100
    model.binary_charge_rule_RF = pyo.Constraint(model.T_daily, rule=binary_charge_rule_RF)
    # def binary_charge_rule1(m, t):
    #     return m.x_in_HH[t] * m.x_out_HH[t] <= 0.001
    # model.binary_charge_rule1 = pyo.Constraint(model.T, rule=binary_charge_rule1)
    #
    # def binary_charge_rule2(m, t):
    #     return m.x_in_RF[t] * m.x_out_RF[t] <= 0.001
    # model.binary_charge_rule2 = pyo.Constraint(model.T, rule=binary_charge_rule2)

    #objective
    def obj_expression(m, y):
        # return (sum(m.x_out_HH[t] * m.PowerPricesBuy[t] - m.x_in_HH[t] * m.PowerPricesSell[t] +
        #            m.x_out_RF[t] * m.PowerPricesSell[t] - m.x_in_RF[t] * m.PowerPricesBuy[t] for t in m.T) -
        #            m.capMax * m.capacityPrice/profit_factor)
        profit_factor = 8760 / len(m.T_daily.ordered_data())
        return (sum(m.x_out_HH[t] * m.PowerPricesBuy[t] - m.x_in_HH[t] * m.PowerPricesSell[t] +
                    (m.x_out_RF_lo[t] - m.x_in_RF_lo[t] + m.x_out_RF_hi[t] - m.x_in_RF_hi[t]) *
                    calc_powerPrice(m.x_in_RF_hi[t] - m.x_out_RF_hi[t], m.SpotPrices[t]) for t in m.T_daily) -
                (m.capMax_HH * m.capacityPrice_HH + (m.cap_RF_LFP + m.cap_RF_Excess) * m.capacityPrice_RF)/profit_factor)

    model.Obj = pyo.Objective(rule=obj_expression, sense=pyo.maximize)
    return model

def calc_powerPrice(menge, startPrice):
    '''
    Soll den endogenen Effekt der Batterien auf Spot-Preise schätzen
    Da bereits Annahmen zu Menge M an Batterien in der LFP und damit den Preisen enthalten ist:
    inkludiere nur den excess amount of Battery usage
    menge [Kwh]:     netto menge die aus System gezogen wird: x_in - x_out
    startPrices [€]: geschätzter Spotpreis
    returns:         endogenen Preis in abhängigkeit von Batterieein- und ausspeicherung
    '''

    alpha_steig = 5e-3 # €/MWh
    anteil_preissensitiv = 0.2
    return startPrice + alpha_steig * anteil_preissensitiv * menge

def instantiate_model(model, data):
    instance = model.create_instance(data)
    return instance

def start_solver(model_instance):
    opt = pyo.SolverFactory('ipopt', executable=exe, tee=True)
    opt.options['tol'] = 1e+3
    opt.options['max_iter'] = 800

    ######
    # d = 0
    # tRange = range(d*24, (d+1)*24)
    # model_instance.T = pyo.Set(initialize=[t for t in tRange])
    #####

    try:
        results = opt.solve(model_instance, tee=True)
    except ValueError:
        print(model_instance)
        opt.options['max_iter'] = 400
        results = opt.solve(model_instance, tee=True)

    print(results.write)

    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        print("this is feasible and optimal")
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print("do something about it? or exit?")
    else:
        # something else is wrong
        print(str(results.solver))

    return results

def write_results(instance, results):
    feedInHH = []
    feedOutHH = []
    feedInRF_lo = []
    feedOutRF_lo = []
    feedInRF_hi = []
    feedOutRF_hi = []
    fill_factor_HH, fill_factor_RF_lo, fill_factor_RF_hi  = [], [], []

    list_prices_buy = [instance.PowerPricesBuy[t] for t in instance.T_daily]
    list_prices_sell = [instance.PowerPricesSell[t] for t in instance.T_daily]

    for t in instance.T_daily.ordered_data():
        feedInHH.append(instance.x_in_HH[t].value)
        feedOutHH.append(instance.x_out_HH[t].value)
        feedInRF_lo.append(instance.x_in_RF_lo[t].value)
        feedOutRF_lo.append(instance.x_out_RF_lo[t].value)
        feedInRF_hi.append(instance.x_in_RF_hi[t].value)
        feedOutRF_hi.append(instance.x_out_RF_hi[t].value)
        fill_factor_HH.append(sum((feedInHH[tt] - feedOutHH[tt]) for tt in range(np.size(feedInHH)))/instance.capMax_HH.value)
        fill_factor_RF_lo.append(sum((feedInRF_lo[tt] - feedOutRF_lo[tt]) for tt in range(np.size(feedInHH))) / instance.cap_RF_LFP.value)
        fill_factor_RF_hi.append(sum((feedInRF_hi[tt] - feedOutRF_hi[tt]) for tt in range(np.size(feedInHH))) / instance.cap_RF_Excess.value)

        #capMax.append(instance.capMax[t].value)
        #nbCycles.append(instance.nbCycles[t].value)
    capacity_HH = instance.capMax_HH.value
    capacityPrice_HH = instance.capacityPrice_HH.value   # €/MWh/a
    capacity_RF = instance.cap_RF_LFP.value + instance.cap_RF_Excess.value
    capacity_RF_Excess = instance.cap_RF_Excess.value
    capacityPrice_RF = instance.capacityPrice_RF.value

    # prices_sell = [calc_powerPrice(fIn2 - fOut2, psell) for fIn2, fOut2, psell in
    #                zip(feedIn2, feedOut2, list_prices_sell)]
    # prices_buy = [calc_powerPrice(fIn2 - fOut2, pbuy) for fIn2, fOut2, pbuy in
    #                zip(feedIn2, feedOut2, list_prices_buy)]
    spotPrices = [calc_powerPrice(instance.x_in_RF_hi[t].value - instance.x_out_RF_hi[t].value, instance.SpotPrices[t]) for t in
                  instance.T_daily]
    profit_factor = 8760 / len(list_prices_buy)
    profit = (np.dot(feedOutHH, list_prices_buy) - np.dot(feedInHH, list_prices_sell) +
              np.dot(feedOutRF_lo, spotPrices) +
              np.dot(feedOutRF_hi, spotPrices) -
              np.dot(feedInRF_lo, spotPrices) -
              np.dot(feedInRF_hi, spotPrices) -
              (capacity_HH * capacityPrice_HH + capacity_RF * capacityPrice_RF) / profit_factor)

    results = {"profit": profit,
               "spotPrices": spotPrices,
               "capacity_RF": instance.cap_RF_LFP.value,
               "capacity_RF_Excess": capacity_RF_Excess,
               "fill_factor_HH": fill_factor_HH,
               "fill_factor_RF_lo": fill_factor_RF_lo,
               "fill_factor_RF_hi": fill_factor_RF_hi,
               "feedInHH": feedInHH,
               "feedOutHH": feedOutHH,
               "feedInRF_lo": feedInRF_lo,
               "feedOutRF_lo": feedOutRF_lo,
               "feedInRF_hi": feedInRF_hi,
               "feedOutRF_hi": feedOutRF_hi}

              # capacity*BATTERY_COST_INITIAL*(1-COST_DEGRESSION)**(y-Y_START)) * profit_factor  #/ np.size(instance.T.ordered_data())

    # pd.DataFrame(np.transpose([feedIn1,feedIn2])).to_csv(input_path + r'Outputs\feedIn_' + str(y) + '.csv', sep=';')
    # pd.DataFrame(np.transpose([feedOut1, feedOut2])).to_csv(input_path + r'Outputs\feedOut_' + str(y) + '.csv', sep=';')
    # pd.DataFrame(fill_factor).to_csv(input_path + r'Outputs\fill_factor_' + str(y) + '.csv', ';')
    #pd.DataFrame(capMax).to_csv(input_path + r'Outputs\capMax.csv')
    #pd.DataFrame(nbCycles).to_csv(input_path + r'Outputs\nbCycles.csv')
    return results

def start_optimization_single_year(input_path, y):
    # create_csv_files(input_path, file_names)
    profits_singleYear = pd.DataFrame()
    fill_factor_HH_Year, fill_factor_RF_lo_Year, fill_factor_RF_hi_Year = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    feedInHH_Year, feedOutHH_Year, feedInRF_lo_Year, feedOutRF_lo_Year, feedInRF_hi_Year, feedOutRF_hi_Year = \
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    spotPrices_Year, capacity_RF_Excess_Year = pd.DataFrame(), pd.DataFrame()

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
        for d in range(0, 10):
            (profits_singleDay,
             spotPrices_singleDay,
             capacity_RF_Excess_singleDay,
             fill_factor_HH_day,
             fill_factor_RF_lo_day,
             fill_factor_RF_hi_day,
             feedInHH_day,
             feedOutHH_day,
             feedInRF_lo_day,
             feedOutRF_lo_day,
             feedInRF_hi_day,
             feedOutRF_hi_day) = start_optimization_single_day(input_path, file_names, y, d)

            profits_singleYear = pd.concat([profits_singleYear, profits_singleDay], axis=0)
            spotPrices_Year = pd.concat([spotPrices_Year, spotPrices_singleDay], axis=0)
            capacity_RF_Excess_Year = pd.concat([capacity_RF_Excess_Year, capacity_RF_Excess_singleDay], axis=0)
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
            input_path + r'Outputs\profits_yearly' + str(y) + '.csv', sep=';')
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

def start_optimization_single_day(input_path, file_names, y, d):
    '''
    performs a capacity sweep over household batteries
    for each capacity C_HH, optimizes the Battery_HH usage aswell as the capacity and usage of Redox-Flow Batteries RF

    returns: dataFrames with data optimization results of the shape [nHours, nCapacities]
    '''
    # optimizes battery use for a single day d in year y
    profits_singleDay, fill_factor_HH_day, fill_factor_RF_lo_day, fill_factor_RF_hi_day  = [], [], [], []
    feedInHH_day, feedOutHH_day, feedInRF_lo_day, feedOutRF_lo_day, feedInRF_hi_day, feedOutRF_hi_day = [], [], [], [], [], []
    capacities_RF_Excess, capacities_RF, spotPrice_day = [], [], []

    capacities_HH = pd.read_csv(input_path + r'Inputs\\' + file_names['Capacities_HH'], header=None).values[:, 0]
    for capacity in capacities_HH:
        model = get_model(input_path + r'Inputs\\', file_names, y, capacity)
        data = load_data(input_path + r'Inputs\\', file_names, model, d)
        instance = instantiate_model(model, data)
        results_solver = start_solver(instance)
        results = write_results(instance, results_solver)

        profits_singleDay.append(results["profit_single"])
        spotPrice_day.append(results["spotPrices"])
        capacities_RF.append(results["capacity_RF"])
        capacities_RF_Excess.append(results["capacity_RF_Excess"])
        fill_factor_HH_day.append(results["fill_factor_single_HH"])
        fill_factor_RF_lo_day.append(results["fill_factor_single_RF_lo"])
        fill_factor_RF_hi_day.append(results["fill_factor_single_RF_hi"])
        feedInHH_day.append(results["feedIn_HH_single"])
        feedOutHH_day.append(results["feedOut_HH_single"])
        feedInRF_lo_day.append(results["feedIn_RF_lo_single"])
        feedOutRF_lo_day.append(results["feedOut_RF_lo_single"])
        feedInRF_hi_day.append(results["feedIn_RF_hi_single"])
        feedOutRF_hi_day.append(results["feedOut_RF_hi_single"])

    column_names = ['Capa_{}'.format(i) for i in range(len(profits_singleDay))]
    hour_list = [datetime(year=y, month=1, day=1) + timedelta(days=d) + timedelta(hours=h) for h in range(24)]
    profits_singleDay = pd.DataFrame(np.reshape(profits_singleDay, (1,-1)), columns = column_names,
                                   index=[datetime(year=y, month=1, day=1) + timedelta(days=d)])
    spotPrices_singleDay = pd.DataFrame(np.array(spotPrice_day).transpose(), columns = column_names,
                                   index=hour_list)
    capacity_RF = pd.DataFrame(np.reshape(capacities_RF, (1,-1)), columns = column_names,
                                   index=[datetime(year=y, month=1, day=1) + timedelta(days=d)])
    capacity_RF_Excess_singleDay = pd.DataFrame(np.reshape(capacities_RF_Excess, (1,-1)), columns = column_names,
                                   index=[datetime(year=y, month=1, day=1) + timedelta(days=d)])
    fill_factor_HH_day = pd.DataFrame(np.array(fill_factor_HH_day).transpose(), columns = column_names,
                                   index=hour_list)
    fill_factor_RF_lo_day = pd.DataFrame(np.array(fill_factor_RF_lo_day).transpose(), columns = column_names,
                                   index=hour_list)
    fill_factor_RF_hi_day = pd.DataFrame(np.array(fill_factor_RF_hi_day).transpose(), columns = column_names,
                                   index=hour_list)
    feedInHH_day = pd.DataFrame(np.array(feedInHH_day).transpose(), columns = column_names,
                                   index=hour_list)
    feedOutHH_day = pd.DataFrame(np.array(feedOutHH_day).transpose(), columns = column_names,
                                   index=hour_list)
    feedInRF_lo_day = pd.DataFrame(np.array(feedInRF_lo_day).transpose(), columns = column_names,
                                   index=hour_list)
    feedOutRF_lo_day = pd.DataFrame(np.array(feedOutRF_lo_day).transpose(), columns = column_names,
                                   index=hour_list)
    feedInRF_hi_day = pd.DataFrame(np.array(feedInRF_hi_day).transpose(), columns = column_names,
                                   index=hour_list)
    feedOutRF_hi_day = pd.DataFrame(np.array(feedOutRF_hi_day).transpose(), columns = column_names,
                                   index=hour_list)

    return profits_singleDay, spotPrices_singleDay, capacity_RF_Excess_singleDay,\
        fill_factor_HH_day, fill_factor_RF_lo_day, fill_factor_RF_hi_day,\
        feedInHH_day, feedOutHH_day, feedInRF_lo_day, feedOutRF_lo_day, feedInRF_hi_day, feedOutRF_hi_day
def start_optimization(input_path, year_list=[]):
    '''
    Starts up the Optimization scheme for battery usage
    writes csv-output to output-Folder

    :param input_path: path of input csv-files for optimization
    :param year_list: list of years that should be optimized
    :return: list of optimized profits for each year
    '''

    if year_list == []:
        year_list = range(2023, 2050, 1)

    df_profits, years = pd.DataFrame(), []
    for y in year_list:
        try:
            print("Starting year {}: {}".format(y, datetime.now()))
            profits_singleYear = start_optimization_single_year(input_path, y)
            print("Finished single Capacity for year {}: {}".format(y, datetime.now()))

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
    df_profits.to_csv(input_path + r'Outputs\Profits.csv', sep=';', mode='w')
    return df_profits

print('finished')

def start_optimization_MT(input_path):
    nJobs = 10
    ystart = 2023
    yend = 2050
    for iJob in range(nJobs):
        year_list = []
        for y in range(ystart + iJob, yend, step=nJobs):
            year_list.append(y)
        profits = start_optimization(input_path, year_list)

        with concurrent.futures.ThreadPoolExecutor(max_workers=nJobs) as executor:
            for index in range(24):
                future = executor.submit(start_optimization, input_path, year_list)
                print(future.result())

if __name__=='__main__':
    profits = start_optimization(input_path)
    # profits = start_optimization_MT(input_path)