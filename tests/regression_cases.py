"""
Seeded regression cases covering every library code path used by the paper's applications.

Each case runs a small system with fixed seeds and returns a dict of numpy arrays.
``make_golden.py`` stores these arrays in ``tests/golden``; ``test_regression.py``
compares fresh runs against them, so that refactoring the libraries can be checked
to be bit-for-bit behaviour preserving.
"""
import importlib.util
import os
import random
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'MOSAIC')
DELAY_DIR = os.path.join(ROOT, 'Application_delay')
IMMUNE_DIR = os.path.join(ROOT, 'Application_Immune_response')
TN_DIR = os.path.join(ROOT, 'Application_Temporal_network')


def load_module(directory, name):
    """Import ``<directory>/<name>.py`` as a fresh module, with its folder on sys.path."""
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location(name, os.path.join(directory, name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def seed(s):
    random.seed(s)
    np.random.seed(s)


def simulation_outputs(sim):
    """Populations, waiting times and counters of a finished simulation."""
    out = {'populations': np.asarray(sim.population_compiled)}
    for i, ch in enumerate(sim.reaction_channel_list):
        out['wait_%d' % i] = np.asarray(ch.wait_times, dtype=float)
        out['accepted_%d' % i] = np.asarray([ch.number_of_accepted_reactions, ch.number_of_rejected_reactions])
    return out


class Param:
    def __init__(self, Tend, N_simulations, timepoints):
        self.Tend = Tend
        self.N_simulations = N_simulations
        self.unit = 'h'
        self.timepoints = timepoints


# --------------------------------------------------------------------------- core library
def case_core_toy():
    """README system: Gamma + Weibull channels with overlapping reactants, bimolecular Poisson channel."""
    mosaic = load_module(ROOT, 'MOSAIC')
    param = Param(10, 3, 100)
    r1 = mosaic.Reaction_channel(param, rate=1, shape_param=20, distribution='Gamma', name='A -> B')
    r1.reactants, r1.products = ['A'], ['B']
    r2 = mosaic.Reaction_channel(param, rate=4, shape_param=5, distribution='Weibull', name='B -> C + A')
    r2.reactants, r2.products = ['B'], ['C', 'A']
    r3 = mosaic.Reaction_channel(param, rate=0.03, name='A + B -> 0')
    r3.reactants, r3.products = ['A', 'B'], []
    sim = mosaic.Gillespie_simulation({'A': 300, 'B': 0, 'C': 0}, param)
    sim.reaction_channel_list = [r1, r2, r3]
    sim.run_simulations(param.Tend, verbose=False, seed=0)
    return simulation_outputs(sim)


def case_core_all_distributions():
    """One channel per distribution in a cycle, plus the min_ratio path on a tiny population."""
    mosaic = load_module(ROOT, 'MOSAIC')
    param = Param(8, 2, 50)
    specs = [('A', 'B', 'Exponential', 1), ('B', 'C', 'Gamma', 4), ('C', 'D', 'Weibull', 3),
             ('D', 'E', 'Normal', 0.3), ('E', 'F', 'LogNormal', 0.5), ('F', 'A', 'Cauchy', 0.2)]
    channels = []
    for a, b, dist, shape in specs:
        ch = mosaic.Reaction_channel(param, rate=1, shape_param=shape, distribution=dist, name='%s -> %s' % (a, b))
        ch.reactants, ch.products = [a], [b]
        channels.append(ch)
    sim = mosaic.Gillespie_simulation({k: v for k, v in zip('ABCDEF', [3, 0, 0, 0, 0, 0])}, param, min_ratio=10)
    sim.reaction_channel_list = channels
    sim.run_simulations(param.Tend, verbose=False, seed=1)
    return simulation_outputs(sim)


def case_core_delay_queue():
    """Precomputed delays (DelaySSA scheme) without overlapping reactants, including shape < 1."""
    mosaic = load_module(ROOT, 'MOSAIC')
    param = Param(10, 2, 60)
    r1 = mosaic.Reaction_channel(param, rate=1, shape_param=6, distribution='Gamma', precompute_delay=True, name='A -> A')
    r1.reactants, r1.products = ['A'], ['A']
    r2 = mosaic.Reaction_channel(param, rate=0.5, shape_param=0.5, distribution='Weibull', precompute_delay=True, name='B -> C')
    r2.reactants, r2.products = ['B'], ['C']
    sim = mosaic.Gillespie_simulation({'A': 50, 'B': 50, 'C': 0}, param)
    sim.reaction_channel_list = [r1, r2]
    sim.run_simulations(param.Tend, verbose=False, seed=2)
    return simulation_outputs(sim)


# --------------------------------------------------------------------------- delay library (Hes1)
def hes1_simulation(gil, beta, method, f, param):
    tau, h, dm, dp, ap = 20, 4.1, 0.029, 0.031, 0.01
    kon = 1 / tau
    L = []
    c = gil.Reaction_channel(param, rate=beta, distribution='Exponential', name='0 -> N')
    c.reactants, c.products = [], ['N']
    L.append(c)
    c = gil.Reaction_channel(param, rate=kon, shape_param=3, distribution='Gamma',
                             precompute_delay=(method == 'DelaySSA'), name='N -> M')
    c.reactants, c.products = ['N'], ['M']
    L.append(c)
    c = gil.Reaction_channel(param, rate=ap, distribution='Exponential', name='M -> M + P', transfer_identity=True)
    c.reactants, c.products = ['M'], ['M', 'P']
    L.append(c)
    c = gil.Reaction_channel(param, rate=dm, distribution='Exponential', name='M -> 0')
    c.reactants, c.products = ['M'], []
    L.append(c)
    c = gil.Reaction_channel(param, rate=dp, distribution='Exponential', name='P -> 0')
    c.reactants, c.products = ['P'], []
    L.append(c)
    sim = gil.Gillespie_simulation({'N': 0, 'M': 0, 'P': 0}, param, L, print_warnings=False, min_ratio=f)
    sim.beta, sim.P0, sim.tau, sim.h = beta, beta, tau, h
    return sim


def _case_hes1(method, f, loop, beta=3.0, Nsim=3):
    gil = load_module(DELAY_DIR, 'lib_MOSAIC_with_delay')
    param = Param(120, Nsim, 200)
    seed(3)
    sim = hes1_simulation(gil, beta, method, f, param)
    sim.run_simulations(param.Tend, verbose=False, delay_method=loop)
    return simulation_outputs(sim)


def case_hes1_delayssa():
    return _case_hes1('DelaySSA', 1, 'rejection')


def case_hes1_mosaic():
    return _case_hes1('MOSAIC', 1, 'rejection')


def case_hes1_mosaic_min_ratio():
    return _case_hes1('MOSAIC', 10, 'rejection', beta=0.05)


def case_hes1_anderson():
    return _case_hes1('DelaySSA', 1, 'Anderson')


# --------------------------------------------------------------------------- rate-function library (immune)
def case_immune_germinal_center():
    """The germinal center model of the paper: custom reactions, per-cell affinities, delay queues."""
    app = load_module(IMMUNE_DIR, 'Simulating_immune_response')
    app.param.Tend = 40
    app.param.N_simulations = 1
    app.param.timepoints = 40
    seed(4)
    sim = app.build_simulation()
    sim.run_simulations(app.param.Tend, verbose=False, delay_method='Standard')
    out = simulation_outputs(sim)
    keys = sorted(sim.ancestor_dict)
    out['ancestor_keys'] = np.asarray(keys)
    out['ancestors'] = np.concatenate([np.asarray(sim.ancestor_dict[k][0], dtype=float) for k in keys])
    out['affinities'] = np.concatenate([np.asarray(sim.ancestor_dict[k][1], dtype=float) for k in keys])
    return out


def case_rate_function_features():
    """Pareto waiting times, state-dependent rate (r0_function), explicit rmax and an X + X channel."""
    gil = load_module(IMMUNE_DIR, 'lib_MOSAIC_with_rate_function')
    param = Param(20, 2, 50)

    def r0_of_state(population, t):
        return 0.5 / (1 + population['Y'] / 20)

    r1 = gil.Reaction_channel(param, rate=1, shape_param=1.5, distribution='Pareto', name='X -> Y')
    r1.reactants, r1.products = ['X'], ['Y']
    r2 = gil.Reaction_channel(param, rate=0.5, distribution='Exponential', rmax=0.5, r0_function=r0_of_state, name='Y -> X')
    r2.reactants, r2.products = ['Y'], ['X']
    r3 = gil.Reaction_channel(param, rate=0.01, distribution='Exponential', name='X + X -> Z')
    r3.reactants, r3.products = ['X', 'X'], ['Z']
    # the empty 'B' and 'Bdiv' species are required by the library's germinal-center bookkeeping
    sim = gil.Gillespie_simulation({'X': 60, 'Y': 0, 'Z': 0, 'B': 0, 'Bdiv': 0}, param, [r1, r2, r3], print_warnings=False)
    sim.Bcell_affinity, sim.Ancestor, sim.ancestor_dict = {}, {}, {}
    seed(5)
    sim.run_simulations(param.Tend, verbose=False, delay_method='rejection')
    return simulation_outputs(sim)


# --------------------------------------------------------------------------- temporal networks (MOSAIC-TN)
def _case_temporal_network(N, Tend, case, use_IP, s):
    tn = load_module(TN_DIR, 'lib_MOSAIC_IP_model')
    seed(s)
    r = tn.simulate_temporal_network(N, Tend, case=case, use_IP=use_IP)
    A = r['events']
    return dict(interaction_times=np.asarray(r['interaction_times'], float),
                inter_event_times=np.asarray(r['inter_event_times'], float),
                edge_interduration=np.asarray(r['edge_interduration'], float),
                Aggregated_Network=np.asarray(r['aggregated_network'], float),
                events_t=np.array([e[0] for e in A], float),
                events_ij=np.array([[e[1], e[2]] for e in A], int).reshape(-1, 2),
                events_start=np.array([e[3] == 'start interaction' for e in A]),
                next_type=np.concatenate([np.asarray(r['next_interaction_type'][n], float) for n in range(N)]),
                next_type_len=np.array([len(r['next_interaction_type'][n]) for n in range(N)]),
                first_order=np.array(sorted(r['first_order_set'])),
                second_order=np.array(sorted(r['second_order_set'])))


def case_tn_basic():
    return _case_temporal_network(12, 300, 'Basic', False, 6)


def case_tn_first_order_ip():
    return _case_temporal_network(12, 300, 'first_order', True, 7)


def case_tn_second_order():
    return _case_temporal_network(10, 200, 'second_order', False, 8)


CASES = {
    'core_toy': case_core_toy,
    'core_all_distributions': case_core_all_distributions,
    'core_delay_queue': case_core_delay_queue,
    'hes1_delayssa': case_hes1_delayssa,
    'hes1_mosaic': case_hes1_mosaic,
    'hes1_mosaic_min_ratio': case_hes1_mosaic_min_ratio,
    'hes1_anderson': case_hes1_anderson,
    'immune_germinal_center': case_immune_germinal_center,
    'rate_function_features': case_rate_function_features,
    'tn_basic': case_tn_basic,
    'tn_first_order_ip': case_tn_first_order_ip,
    'tn_second_order': case_tn_second_order,
}
