
from metacountregressor.solution import ObjectiveFunction
from metacountregressor.metaheuristics import (harmony_search,differential_evolution,simulated_annealing)
from metacountregressor import helperprocess
import pandas as pd

df = pd.read_csv("d_hm.csv")
y = df['Y']  # Frequency of crashes
X = df.drop(columns=['Y']) # setup X based on data
X.columns
print(X.columns)

manual_fit_spec = {
'fixed_terms': ['const'],
'rdm_terms':  [ 'X1:uniform'],
'rdm_cor_terms': [],
'grouped_terms': [],
'hetro_in_means': ['X2:normal', 'Z1:normal','X3:normal'],
'transformations': ['no', 'no', 'no', 'no', 'no', 'no'],
'dispersion': 0
}
arguments = {'test_percentage': 0.2, 'complexity_level': 6, 'reg_penalty':0} #Objective args
arguments['Manual_Fit'] = manual_fit_spec
initial_solution = None
obj_fun = ObjectiveFunction(X, y, **arguments)




# CONSTRAINED SEARCH
model_terms = {
    'Y': 'Y',         # Replace 'FREQ' with the name of your dependent variable
    'group': None,    # Replace 'group_column' with the name of your grouping column (or None if not used)
    'panels': None,      # Replace 'panel_column' with the name of your panel column (or None if not used)
    'Offset': None                # Replace None with the name of your offset column if not using one
}
variable_decisions = {
'X1': {'levels': [0,1,2,3,4,6], 'transformations': ['no'], 'distributions': ['u', 'ln', 'tn']},
'X2': {'levels': [1], 'transformations': ['no'], 'distributions': []},
'X3':{'levels': [1,3,6], 'transformations': ['no'], 'distributions': ['u', 'ln', 'tn']},
'Z1': {'levels': [0,1,2,3,4,6], 'transformations': ['no'], 'distributions': ['u', 'ln', 'tn']}
}

a_des, X = helperprocess.set_up_analyst_constraints(X, model_terms, variable_decisions)

arguments['decisions'] = a_des
arguments['model_types'] = [[0]]
arguments['instance'] = 'constrained' # GIVE NAME
arguments['algorithm']='hs'
arguments_unconstrained = arguments.copy()
arguments_unconstrained['decisions'] = None
arguments_unconstrained['instance'] = 'unconstrained'
obj_fun = ObjectiveFunction(X, y, **arguments)
results = harmony_search(obj_fun)
helperprocess.results_printer(results, arguments['algorithm'], int(arguments['is_multi']))


print('compared to')

obj_fun = ObjectiveFunction(X, y, **arguments_unconstrained)
results = harmony_search(obj_fun)
helperprocess.results_printer(results, arguments_unconstrained['algorithm'], int(arguments_unconstrained['is_multi']))