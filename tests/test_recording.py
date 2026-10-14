"""The recorded population trajectory must equal the exact state on the recording grid."""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'MOSAIC'))

import MOSAIC as mosaic  # noqa: E402


class param:
    Tend = 20
    N_simulations = 1
    unit = 'h'
    timepoints = 200


def run_with_event_log(min_ratio, seed):
    """Run one realisation and return (recorded trajectory, exact state on the grid)."""
    r1 = mosaic.Reaction_channel(param, rate=1, shape_param=3, distribution='Gamma', name='A -> B')
    r1.reactants, r1.products = ['A'], ['B']
    r2 = mosaic.Reaction_channel(param, rate=0.3, name='B -> A')
    r2.reactants, r2.products = ['B'], ['A']
    sim = mosaic.Gillespie_simulation({'A': 5, 'B': 0}, param, min_ratio=min_ratio)
    sim.reaction_channel_list = [r1, r2]
    sim.run_simulations(0.001, verbose=False, seed=seed)  # cheap call that sets up the bookkeeping

    log = [(0.0, [5, 0])]
    original = sim.perform_reaction

    def logged(mu, reactant_id=None):
        original(mu, reactant_id)
        log.append((sim.t, list(sim.reactant_population.values())))

    sim.perform_reaction = logged
    sim.reinitialize_pop()
    sim.run_simulation(param.Tend)

    grid = np.arange(param.timepoints) * param.Tend / param.timepoints
    times = np.array([t for t, _ in log])
    states = np.array([s for _, s in log])
    exact = states[np.searchsorted(times, grid, side='right') - 1]
    return sim.population_t[:, :-1], exact


def test_recorded_trajectory_matches_exact_state_on_grid():
    for min_ratio in (1, 10):
        for seed in range(5):
            recorded, exact = run_with_event_log(min_ratio, seed)
            np.testing.assert_array_equal(recorded, exact)
