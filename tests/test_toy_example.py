"""Smoke tests for the core MOSAIC library (run with ``pytest``)."""
import os
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'MOSAIC'))

import MOSAIC as mosaic  # noqa: E402


class param:
    Tend = 10
    N_simulations = 5
    unit = 'h'
    timepoints = 100


def build_toy_system():
    r1 = mosaic.Reaction_channel(param, rate=1, shape_param=20, distribution='Gamma', name='A -> B')
    r1.reactants, r1.products = ['A'], ['B']
    r2 = mosaic.Reaction_channel(param, rate=4, shape_param=5, distribution='Weibull', name='B -> C + A')
    r2.reactants, r2.products = ['B'], ['C', 'A']
    r3 = mosaic.Reaction_channel(param, rate=0.03, name='A + B -> 0')
    r3.reactants, r3.products = ['A', 'B'], []
    sim = mosaic.Gillespie_simulation({'A': 300, 'B': 0, 'C': 0}, param)
    sim.reaction_channel_list = [r1, r2, r3]
    return sim


def test_toy_example_runs_and_reproduces_rates():
    sim = build_toy_system()
    pop = sim.run_simulations(param.Tend, verbose=False, seed=0)
    assert pop.shape == (param.N_simulations, param.timepoints, 4)
    assert np.all(pop >= 0)
    # last column is the total population
    np.testing.assert_allclose(pop[:, :, -1], pop[:, :, :-1].sum(axis=2))
    # mean inter-event times should match the requested rates within ~10%
    for channel, rate in zip(sim.reaction_channel_list[:2], [1, 4]):
        waits = np.array(channel.wait_times)
        assert len(waits) > 100
        assert abs(1 / waits.mean() - rate) / rate < 0.1


def test_seed_makes_runs_reproducible():
    pop_a = build_toy_system().run_simulations(param.Tend, verbose=False, seed=123)
    pop_b = build_toy_system().run_simulations(param.Tend, verbose=False, seed=123)
    np.testing.assert_array_equal(pop_a, pop_b)


def test_precomputed_delay_supports_shape_below_one():
    r = mosaic.Reaction_channel(param, rate=1, shape_param=0.5, distribution='Gamma',
                                precompute_delay=True, name='A -> B')
    r.reactants, r.products = ['A'], ['B']
    sim = mosaic.Gillespie_simulation({'A': 200, 'B': 0}, param)
    sim.reaction_channel_list = [r]
    pop = sim.run_simulations(param.Tend, verbose=False, seed=1)
    assert pop[:, -1, 1].mean() > 150  # most of A has turned into B after 10 mean lifetimes


def test_overlap_detection_does_not_depend_on_channel_order():
    # 'A' takes part in two channels, 'B' in one. The channel listed last only
    # involves 'B', which used to hide the overlap on 'A'.
    r1 = mosaic.Reaction_channel(param, rate=1, shape_param=3, distribution='Gamma', name='A -> B')
    r1.reactants, r1.products = ['A'], ['B']
    r2 = mosaic.Reaction_channel(param, rate=0.5, name='A -> 0')
    r2.reactants, r2.products = ['A'], []
    r3 = mosaic.Reaction_channel(param, rate=0.1, name='B -> 0')
    r3.reactants, r3.products = ['B'], []
    sim = mosaic.Gillespie_simulation({'A': 50, 'B': 0}, param)
    sim.reaction_channel_list = [r1, r2, r3]
    sim.run_simulations(param.Tend, verbose=False, seed=0)
    assert sim.overlapping_reactants is True


def test_invalid_channels_raise():
    with pytest.raises(ValueError):
        mosaic.Reaction_channel(param, rate=-1)
    with pytest.raises(ValueError):
        mosaic.Reaction_channel(param, rate=1, distribution='Not_a_distribution')
    with pytest.raises(ValueError):
        mosaic.Reaction_channel(param, rate=1, shape_param=0.5, distribution='Weibull')


def test_plot_functions_run_headless():
    sim = build_toy_system()
    sim.run_simulations(param.Tend, verbose=False, seed=0)
    sim.plot_populations()
    sim.plot_populations(reactant_list=['A'])  # second call must not reuse mutated defaults
    sim.plot_inter_event_time_distribution()
