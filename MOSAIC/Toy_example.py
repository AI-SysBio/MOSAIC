"""
Minimal MOSAIC example (same system as in the README).

Three species A, B, C with:
    A      -> B        Gamma-distributed waiting time (alpha = 20)
    B      -> C + A    Weibull-distributed waiting time (k = 5)
    A + B  -> 0        Poisson process (two reactants)

Run from this folder:  python Toy_example.py
"""
import MOSAIC as mosaic


class param:
    Tend = 10             # length of the simulation
    N_simulations = 20    # results are averaged over this many realisations
    unit = 'h'            # unit of time (plots only)
    timepoints = 100      # number of recorded timepoints (plots only)


def main():
    r1, r2, r3 = 1, 4, 0.03
    alpha1, alpha2 = 20, 5

    # Reaction channels
    reaction1 = mosaic.Reaction_channel(param, rate=r1, shape_param=alpha1, distribution='Gamma', name='A -> B')
    reaction1.reactants = ['A']
    reaction1.products = ['B']

    reaction2 = mosaic.Reaction_channel(param, rate=r2, shape_param=alpha2, distribution='Weibull', name='B -> C + A')
    reaction2.reactants = ['B']
    reaction2.products = ['C', 'A']

    reaction3 = mosaic.Reaction_channel(param, rate=r3, name='A + B -> 0')
    reaction3.reactants = ['A', 'B']
    reaction3.products = []

    # Initial population
    N_init = {'A': 300, 'B': 0, 'C': 0}

    # Run
    G_simul = mosaic.Gillespie_simulation(N_init, param)
    G_simul.reaction_channel_list = [reaction1, reaction2, reaction3]
    populations = G_simul.run_simulations(param.Tend, verbose=True, seed=0)
    print('populations array shape:', populations.shape)

    # Plots
    G_simul.plot_inter_event_time_distribution()
    G_simul.plot_populations()


if __name__ == '__main__':
    main()
