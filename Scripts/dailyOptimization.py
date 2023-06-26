import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from numpy import random as rnd
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


exe = os.path.join("X:\\", "Analysen", "Tools", "OpenSolver2.9.4_Beta_LinearWin", "Solvers", "win64", "ipopt", "ipopt")
def write_results(instance, results):
    '''
    Extracts information from optimization results and writes to Dataframes

    :param instance: Concrete Pyomo instance that was optimized
    :param results: results returned by Pyomo optimization
    :return: dictionary with relevant optimization results
    '''
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
               "feedIn_HH": feedInHH,
               "feedOut_HH": feedOutHH,
               "feedIn_RF_lo": feedInRF_lo,
               "feedOut_RF_lo": feedOutRF_lo,
               "feedIn_RF_hi": feedInRF_hi,
               "feedOut_RF_hi": feedOutRF_hi}

              # capacity*BATTERY_COST_INITIAL*(1-COST_DEGRESSION)**(y-Y_START)) * profit_factor  #/ np.size(instance.T.ordered_data())

    # pd.DataFrame(np.transpose([feedIn1,feedIn2])).to_csv(input_path + r'Outputs\feedIn_' + str(y) + '.csv', sep=';')
    # pd.DataFrame(np.transpose([feedOut1, feedOut2])).to_csv(input_path + r'Outputs\feedOut_' + str(y) + '.csv', sep=';')
    # pd.DataFrame(fill_factor).to_csv(input_path + r'Outputs\fill_factor_' + str(y) + '.csv', ';')
    #pd.DataFrame(capMax).to_csv(input_path + r'Outputs\capMax.csv')
    #pd.DataFrame(nbCycles).to_csv(input_path + r'Outputs\nbCycles.csv')
    return results
def optimize_single(pyo_model, pyo_data):
    '''
    optimize a single instance of the model for a year, day and capacity, given pyomo data
    :param pyo_model:
    :param pyo_data:
    :return:
    '''
    instance = instantiate_model(pyo_model, pyo_data)
    results_solver = start_solver(instance)
    return write_results(instance, results_solver)

def start_optimization_single_day(input_path, file_names, y, d, nHours):
    '''
    performs a capacity sweep over household batteries
    for each capacity C_HH, optimizes the Battery_HH usage aswell as the capacity and usage of Redox-Flow Batteries RF

    returns: dataFrames with data optimization results of the shape [nHours, nCapacities]
    '''
    # optimizes battery use for a single day d in year y
    capacity_steps = 9
    profits_singleDay, fill_factor_HH_day, fill_factor_RF_lo_day, fill_factor_RF_hi_day  = [], [], [], []
    feedInHH_day, feedOutHH_day, feedInRF_lo_day, feedOutRF_lo_day, feedInRF_hi_day, feedOutRF_hi_day = [], [], [], [], [], []
    capacities_RF_Excess, capacities_RF, spotPrice_day = [], [], []

    capacities_HH = pd.read_csv(input_path + r'Inputs\\' + file_names['Capacities_HH'], header=None).values[:, 0]
    capacities_HH = capacities_HH[::capacity_steps]
    for capacity in capacities_HH:
        pyo_model = get_model(input_path + r'Inputs\\', file_names, y, capacity)
        data = load_data(input_path + r'Inputs\\', file_names, pyo_model, d)
        results = optimize_single(pyo_model, data)

        profits_singleDay.append(results["profit"])
        spotPrice_day.append(results["spotPrices"])
        capacities_RF.append(results["capacity_RF"])
        capacities_RF_Excess.append(results["capacity_RF_Excess"])
        fill_factor_HH_day.append(results["fill_factor_HH"])
        fill_factor_RF_lo_day.append(results["fill_factor_RF_lo"])
        fill_factor_RF_hi_day.append(results["fill_factor_RF_hi"])
        feedInHH_day.append(results["feedIn_HH"])
        feedOutHH_day.append(results["feedOut_HH"])
        feedInRF_lo_day.append(results["feedIn_RF_lo"])
        feedOutRF_lo_day.append(results["feedOut_RF_lo"])
        feedInRF_hi_day.append(results["feedIn_RF_hi"])
        feedOutRF_hi_day.append(results["feedOut_RF_hi"])

    column_names = ['Capa_{}'.format(capacity_steps * i) for i in range(len(profits_singleDay))]
    hour_list = [datetime(year=y, month=1, day=1) + timedelta(days=d) + timedelta(hours=h) for h in range(nHours)]
    profits_singleDay = pd.DataFrame(np.reshape(profits_singleDay, (1, -1)), columns=column_names,
                                   index=[datetime(year=y, month=1, day=1) + timedelta(days=d)])
    spotPrices_singleDay = pd.DataFrame(np.array(spotPrice_day).transpose(), columns = column_names,
                                   index=hour_list)
    capacity_RF = pd.DataFrame(np.reshape(capacities_RF, (1, -1)), columns = column_names,
                                   index=[datetime(year=y, month=1, day=1) + timedelta(days=d)])
    capacity_RF_Excess_singleDay = pd.DataFrame(np.reshape(capacities_RF_Excess, (1, -1)), columns = column_names,
                                   index=[datetime(year=y, month=1, day=1) + timedelta(days=d)])
    fill_factor_HH_day = pd.DataFrame(np.array(fill_factor_HH_day).transpose(), columns=column_names,
                                   index=hour_list)
    fill_factor_RF_lo_day = pd.DataFrame(np.array(fill_factor_RF_lo_day).transpose(), columns=column_names,
                                   index=hour_list)
    fill_factor_RF_hi_day = pd.DataFrame(np.array(fill_factor_RF_hi_day).transpose(), columns=column_names,
                                   index=hour_list)
    feedInHH_day = pd.DataFrame(np.array(feedInHH_day).transpose(), columns=column_names,
                                   index=hour_list)
    feedOutHH_day = pd.DataFrame(np.array(feedOutHH_day).transpose(), columns=column_names,
                                   index=hour_list)
    feedInRF_lo_day = pd.DataFrame(np.array(feedInRF_lo_day).transpose(), columns=column_names,
                                   index=hour_list)
    feedOutRF_lo_day = pd.DataFrame(np.array(feedOutRF_lo_day).transpose(), columns=column_names,
                                   index=hour_list)
    feedInRF_hi_day = pd.DataFrame(np.array(feedInRF_hi_day).transpose(), columns=column_names,
                                   index=hour_list)
    feedOutRF_hi_day = pd.DataFrame(np.array(feedOutRF_hi_day).transpose(), columns=column_names,
                                   index=hour_list)

    return profits_singleDay, spotPrices_singleDay, capacity_RF_Excess_singleDay, capacity_RF, \
        fill_factor_HH_day, fill_factor_RF_lo_day, fill_factor_RF_hi_day,\
        feedInHH_day, feedOutHH_day, feedInRF_lo_day, feedOutRF_lo_day, feedInRF_hi_day, feedOutRF_hi_day


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
        results = opt.solve(model_instance, tee=False)
    except ValueError:
        print(model_instance)
        opt.options['max_iter'] = 400
        results = opt.solve(model_instance, tee=False)

    # print(results.write)

    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        # print("this is feasible and optimal")
        pass
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print("do something about it? or exit?")
    else:
        # something else is wrong
        print(str(results.solver))

    return results


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
        return m.x_in_HH[t] <= m.capMax_HH / 4 #* pyo.exp(ncycles * pyo.log(FCTR_BATTERY_DEGREGATION))
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

    # def capacity_rule_HH_2(m, tend):
    #     # Approximate number of cycles by sum(x_in(t))/capMax
    #     #ncycles = (sum(m.x_in_HH[t] + m.x_in_RF[t] for t in m.T if t < tend+1) / m.capMax)
    #     return sum((m.x_in_HH[t] - m.x_out_HH[t]) for t in m.T_daily if t < tend + 1) >= 0
    # model.Constraint_capacityHH2 = pyo.Constraint(model.T_daily, rule=capacity_rule_HH_2)
    # def capacity_rule_RF_lo_2(m, tend):
    #     # Approximate number of cycles by sum(x_in(t))/capMax
    #     #ncycles = (sum(m.x_in_HH[t] + m.x_in_RF[t] for t in m.T if t < tend+1) / m.capMax)
    #     return sum((m.x_in_RF_lo[t] - m.x_out_RF_lo[t]) for t in m.T_daily if t < tend + 1) >= 0
    # model.capacity_rule_RF_lo_2 = pyo.Constraint(model.T_daily, rule=capacity_rule_RF_lo_2)
    # def capacity_rule_RF_hi_2(m, tend):
    #     # Approximate number of cycles by sum(x_in(t))/capMax
    #     #ncycles = (sum(m.x_in_HH[t] + m.x_in_RF[t] for t in m.T if t < tend+1) / m.capMax)
    #     return sum((m.x_in_RF_hi[t] - m.x_out_RF_hi[t]) for t in m.T_daily if t < tend + 1) >= 0
    # model.capacity_rule_RF_hi_2 = pyo.Constraint(model.T_daily, rule=capacity_rule_RF_hi_2)


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

    def positive_charge_rule_RF_lo(m, tend):
        return sum((m.x_in_RF_lo[t] - m.x_out_RF_lo[t]) for t in m.T_daily if t < tend + 1) >= 0
    model.positive_charge_rule_RF_lo = pyo.Constraint(model.T_daily, rule=positive_charge_rule_RF_lo)

    def positive_charge_rule_RF_Hi(m, tend):
        return sum((m.x_in_RF_hi[t] - m.x_out_RF_hi[t]) for t in m.T_daily if t < tend + 1) >= 0
    model.positive_charge_rule_RF_Hi = pyo.Constraint(model.T_daily, rule=positive_charge_rule_RF_Hi)

    def binary_charge_rule_HH(m, t):
        return (m.x_in_HH[t]) * (m.x_out_HH[t]) <= 100
    model.binary_charge_rule_HH = pyo.Constraint(model.T_daily, rule=binary_charge_rule_HH)
    def binary_charge_rule_RF(m, t):
        # both in or both out
        return (m.x_in_RF_lo[t] + m.x_in_RF_hi[t]) * (m.x_out_RF_lo[t] + m.x_out_RF_hi[t]) <= 100
    model.binary_charge_rule_RF = pyo.Constraint(model.T_daily, rule=binary_charge_rule_RF)

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
    anteil_preissensitiv = 1.0
    return startPrice + alpha_steig * anteil_preissensitiv * menge