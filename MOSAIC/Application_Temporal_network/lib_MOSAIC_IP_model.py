"""
MOSAIC-TN: simulation of bursty face-to-face interactions as a non-Markovian reaction network.

Every agent is a reactant and two reaction channels are simulated with the rejection
scheme of MOSAIC:

    A_i + A_j -> [A_i A_j]      start of an interaction, Pareto-distributed waiting time
                                since the last interaction of each agent (rate rAA, shape alphaAA)
    [A_i A_j] -> A_i + A_j      end of an interaction, Pareto-distributed duration drawn
                                when the interaction starts (rate rA, shape alphaA)

The propensity of a pair is the product of the two agents' hazards divided by the total
hazard, so that an agent that has been inactive for a long time is unlikely to interact
(bursty activity). Memory and heterogeneity enter through the choice of the partner:

    case='Basic'         partners are chosen uniformly                     (MOSAIC-TN A in the paper)
    case='first_order'   previous partners are 10 times more likely        (MOSAIC-TN B in the paper)
    case='second_order'  friends of friends are 5 times more likely as well
    use_IP=True          each agent has its own activity level (gamma distributed)

``simulate_temporal_network`` runs the model and returns the raw results;
``run_MOSAIC_simulation`` is the wrapper used by ``Benchmarking_time.py`` and can write
the interaction list in the ``t i j`` format used by ``Compare_networks.py``.
"""
import random
from math import log, gamma, exp, pi, sqrt, erf, atan

import numpy as np
import pandas as pd
import pickle
import scipy.stats as stat
from scipy.special import gammainc
from sortedcontainers import SortedDict

# ----------------------------------------------------------------------------- model parameters
PARAMETERS = {
    'distribution': 'Pareto',   # waiting time between interactions of an agent
    'rAA': 0.1,
    'alphaAA': {'Basic': 0.9, 'first_order': 0.75, 'second_order': 0.8, 'second_order_IP': 0.7},
    'distribution_A': 'Pareto',  # duration of an interaction
    'rA': 0.4,
    'alphaA': 1.4,
    'weight_stranger': 0.1,      # relative probability of choosing a partner never met before
    'weight_friend': 1.0,        # ... a previous partner (first_order and second_order cases)
    'weight_friend_of_friend': 0.5,  # ... a friend of a friend (second_order case)
    'activity_gamma_shape': 4,   # individual activity levels ~ Gamma(4) when use_IP=True
}
END_OF_TIME = 10 ** 27  # sentinel key of the (always non-empty) queue of interaction ends


def getID(i, j):
    """Order-independent identifier of the pair (i, j)."""
    if i > j:
        i, j = j, i
    return '%s_%s' % (i, j)


def simulate_temporal_network(N, Tend=2000, case='Basic', use_IP=False):
    """
    Simulate N agents until time Tend and return a dict with:

        'events'                list of [t, i, j, 'start interaction' | 'end interaction']
        'interaction_times'     duration of every interaction
        'inter_event_times'     time since the previous interaction of each agent, for every new interaction
        'edge_interduration'    time since the previous interaction of the same pair
        'aggregated_network'    N x N matrix of the total interaction time of every pair
        'next_interaction_type' per agent, 1 if an interaction was with a previous partner, else 0
        'first_order_set'       pairs that have interacted at least once (empty for case='Basic')
        'second_order_set'      friends-of-friends pairs (second_order case only)
        'friends'               per agent, set of previous partners
        'activity'              individual activity levels (None unless use_IP)

    Random draws come from ``random`` (event times, acceptance) and ``numpy.random``
    (partner choice, individual activities, interaction durations); seed both for
    reproducible runs.
    """
    P = PARAMETERS
    rAA, rA, alphaA = P['rAA'], P['rA'], P['alphaA']
    alphaAA = P['alphaAA']['second_order_IP' if (case == 'second_order' and use_IP) else case]
    distribution_A = P['distribution_A']

    # individual activity levels (probability of being the initiating agent)
    p_init = None
    if use_IP:
        p_init = stat.gamma.rvs(a=P['activity_gamma_shape'], size=N)
        p_init = p_init / np.sum(p_init)

    # memory of the network
    first_order_set = set()
    second_order_set = set()
    friends = {ni: set() for ni in range(N)}
    next_interaction_type = {ni: [] for ni in range(N)}

    # observables
    events = []
    inter_event_times = []
    interaction_times = []
    edge_interduration = []
    aggregated_network = np.zeros((N, N))

    # state of the simulation
    last_interaction = {i: 0 for i in range(N)}            # time of the last interaction of each agent
    last_pair_interaction = {}                             # time of the last interaction of each pair
    for i in range(N):
        for j in range(N):
            last_pair_interaction[getID(i, j)] = 0
    interaction_end = SortedDict({END_OF_TIME: [-1, -1]})  # queue of scheduled interaction ends
    currently_interacting = set()
    t = 0

    # Upper bound of the pair propensity: the Pareto hazard alpha/t is maximal at the
    # minimum of the distribution, which is fixed to 1 time unit, so lambda_max = alphaAA.
    rmax = alphaAA / (N - 1)
    rsum = rmax * N * (N - 1) / 2

    while t < Tend:

        # 1. time of the next candidate interaction (exponential with the total upper bound)
        u = random.random()
        Dt = -log(u) / rsum

        # 2. an interaction ends before the candidate event
        t_end_min, (i, j) = next(iter(interaction_end.items()))
        if t_end_min < t + Dt:
            t = t_end_min
            events.append([t, i, j, 'end interaction'])
            interaction_end.popitem(index=0)
            currently_interacting.remove(getID(i, j))
            continue

        t += Dt

        # 3. choose the initiating agent (uniformly or according to its activity) ...
        if use_IP:
            i = np.random.choice(np.arange(N), p=p_init)
        else:
            i = np.random.randint(0, N)

        # ... and its partner, with a preference for previous partners (memory of the network)
        weights = partner_weights(i, N, case, currently_interacting, first_order_set, second_order_set)
        j = np.random.choice(np.arange(N), p=weights)
        if i > j:
            i, j = j, i

        # 4. propensity of the pair from the hazards of the two agents
        ti = t - last_interaction[i]
        tj = t - last_interaction[j]
        rate_i = rate_function(ti, r0=rAA, shape_param=alphaAA)
        rate_j = rate_function(tj, r0=rAA, shape_param=alphaAA)
        all_rates = [rate_function(t - last_interaction[k], r0=rAA, shape_param=alphaAA) for k in range(N)]
        tot_rate = np.sum(all_rates) - (rate_i + rate_j) / 2
        rate_ij = (rate_i * rate_j) / tot_rate if tot_rate > 0 else 0

        # 5. rejection step
        if rate_ij >= rmax * random.random():
            inter_event_times.append(ti)
            inter_event_times.append(tj)
            last_interaction[i] = t
            last_interaction[j] = t

            interact_id = getID(i, j)
            edge_interduration.append(last_pair_interaction[interact_id])
            last_pair_interaction[interact_id] = t

            if interact_id not in currently_interacting:
                # schedule the end of the interaction
                duration = generate_delay_from_distribution(distribution_A, rA, alphaA)
                interaction_end[t + duration] = [i, j]
                interaction_times.append(duration)
                currently_interacting.add(interact_id)

                events.append([t, i, j, 'start interaction'])
                aggregated_network[i, j] += duration
                aggregated_network[j, i] += duration
                is_friend = 1 if interact_id in first_order_set else 0
                next_interaction_type[i].append(is_friend)
                next_interaction_type[j].append(is_friend)

                # update the memory of the network
                if case != 'Basic':
                    friends[i].add(j)
                    friends[j].add(i)
                    first_order_set.add(interact_id)
                if case == 'second_order':
                    second_order_set.add(interact_id)
                    # NOTE: kept as in the original code, which registers the pairs (k, j) for the
                    # friends k of both i and j (the friends of i become friends-of-friends of j,
                    # but not the friends of j of i). This case is not used in the paper.
                    for ki in friends[j]:
                        second_order_set.add(getID(ki, j))
                    for kj in friends[i]:
                        second_order_set.add(getID(kj, j))

    return {
        'events': events,
        'interaction_times': interaction_times,
        'inter_event_times': inter_event_times,
        'edge_interduration': edge_interduration,
        'aggregated_network': aggregated_network,
        'next_interaction_type': next_interaction_type,
        'first_order_set': first_order_set,
        'second_order_set': second_order_set,
        'friends': friends,
        'activity': p_init,
    }


def partner_weights(i, N, case, currently_interacting, first_order_set, second_order_set):
    """Probability of each agent to be the partner of agent i (zero for i itself and for pairs already interacting)."""
    P = PARAMETERS
    weights = np.zeros(N)
    for j in range(N):
        interact_id = getID(i, j)
        weights[j] = P['weight_stranger']
        if i == j or interact_id in currently_interacting:
            weights[j] = 0
        elif interact_id in first_order_set:
            if case == 'first_order' or case == 'second_order':
                weights[j] = P['weight_friend']
        elif interact_id in second_order_set:
            if case == 'second_order':
                weights[j] = P['weight_friend_of_friend']
    return weights / np.sum(weights)


def run_MOSAIC_simulation(N, Tend=2000, case='Basic', use_IP=False, output_file=None):
    """
    Run the model (see ``simulate_temporal_network``) and optionally write the interaction
    list to ``output_file`` in the ``t i j`` format read by ``Compare_networks.py``.
    """
    results = simulate_temporal_network(N, Tend, case=case, use_IP=use_IP)
    if output_file is not None:
        write_interaction_file(results['events'], output_file)
    return results


# ----------------------------------------------------------------------------- waiting-time distributions
def rate_function(t, r0=1, shape_param=1, return_sdf_and_pdf=False, distribution='Pareto'):
    """
    Hazard rate at time t since the last event, for a distribution of mean 1/r0 and the given
    shape parameter. With ``return_sdf_and_pdf`` the survival function and the pdf are
    returned as well. For the Pareto distribution the minimum is fixed to 1 time unit.
    """
    if distribution == 'Exponential':
        rate = r0
        sdf = exp(-r0 * t)
        pdf = r0 * sdf

    elif distribution == 'Weibull':
        alpha = shape_param
        beta = (alpha) * (r0 * gamma((alpha + 1) / (alpha))) ** (alpha)
        rate = (t ** (alpha - 1)) * beta
        sdf = exp(-beta * t ** alpha / alpha)
        pdf = rate * sdf

    elif distribution == 'Pareto':
        alpha = shape_param
        mu = 1  # minimum of the distribution (the time resolution of the data)
        if t < mu:
            rate = 0
            sdf = 1
            pdf = 0
        else:
            rate = alpha / t
            pdf = alpha * mu ** alpha / t ** (alpha + 1)
            sdf = (mu / t) ** alpha

    elif distribution == 'Normal':
        sigma = shape_param / r0
        mu = 1 / r0
        pdf = 1 / (sigma * sqrt(2 * pi)) * exp(-(1 / 2) * (t - mu) ** 2 / (sigma ** 2))
        cdf = 1 / 2 * (1 + erf((t - mu) / (sigma * sqrt(2))))
        sdf = 1 - cdf
        if t > mu + 7 * sigma:  # the cdf is not precise enough in the far tail
            rate = (t - mu) / sigma ** 2
        else:
            rate = pdf / (1 - cdf)

    elif distribution == 'Gamma':
        alpha = shape_param
        if alpha < 1 and t == 0:
            rate = 0
            pdf = 0
            sdf = 0
        else:
            beta = alpha * r0
            pdf = (beta ** alpha) * (t ** (alpha - 1)) * exp(-beta * t) / gamma(alpha)
            cdf = gammainc(alpha, beta * t)
            rate = pdf / (1 - cdf)
            sdf = 1 - cdf

    elif distribution == 'Cauchy':
        gam = shape_param / r0
        mu = 1 / r0
        pdf = 1 / (pi * gam * (1 + ((t - mu) / gam) ** 2))
        cdf = (1 / pi) * atan((t - mu) / gam) + 1 / 2
        rate = pdf / (1 - cdf)
        sdf = 1 - cdf

    else:
        raise ValueError('Unsupported distribution: %s' % distribution)

    if return_sdf_and_pdf:
        return rate, pdf, sdf
    return rate


def generate_delay_from_distribution(distribution, rate, shape_param, N_=None):
    """Draw N_ waiting times (a single float when N_ is None) from the given distribution."""
    N = 1 if N_ is None else N_
    d = distribution.lower()
    if d in ['exponential', 'exp']:
        delay = stat.expon.rvs(loc=0, scale=1 / rate, size=N)
    elif d in ['weibull', 'weib']:
        alpha = shape_param
        if shape_param == 1:
            delay = stat.expon.rvs(loc=0, scale=1 / rate, size=N)
        else:
            beta = (alpha) * (rate * gamma((alpha + 1) / (alpha))) ** (alpha)
            delay = stat.weibull_min.rvs(alpha, loc=0, scale=(alpha / beta) ** (1 / alpha), size=N)
    elif d in ['gaussian', 'normal', 'norm']:
        mu = 1 / rate
        delay = stat.norm.rvs(loc=mu, scale=mu * shape_param, size=N)
    elif d in ['gamma', 'gam']:
        alpha = shape_param
        delay = stat.gamma.rvs(alpha, loc=0, scale=1 / (alpha * rate), size=N)
    elif d in ['lognormal', 'lognorm']:
        mu0 = 1 / rate
        sigma0 = mu0 * shape_param
        sigma = sqrt(log(1 + sigma0 ** 2 / mu0 ** 2))
        delay = stat.lognorm.rvs(sigma, loc=0, scale=mu0, size=N)
    elif d in ['cauchy', 'cau']:
        delay = stat.cauchy.rvs(loc=1 / rate, scale=shape_param, size=N)
    elif d in ['pareto', 'par']:
        delay = stat.pareto.rvs(shape_param, scale=1, size=N)  # minimum fixed to 1 time unit
    else:
        raise ValueError('Unsupported distribution: %s' % distribution)
    return delay[0] if N_ is None else delay


# ----------------------------------------------------------------------------- input / output
def write_interaction_file(interaction_list, fname, resolution=1):
    """
    Write the events of a simulation as a list of active interactions per integer
    timestep (``t i j`` per line, the format of the SocioPatterns data). Every start/end
    event is expanded to the set of pairs interacting at that time, timestamps are rounded
    to the given resolution and timesteps without events repeat the previous interactions.
    """
    All_interactions = np.array(interaction_list)
    interaction_csv = []
    current_interacting = set()
    for i in range(len(All_interactions)):
        itype = All_interactions[i, 3]
        timestamp = float(All_interactions[i, 0])
        n1, n2 = All_interactions[i, 1:3].astype(int)
        if 'start' in itype:
            current_interacting.add((n1, n2))
        if 'end' in itype:
            current_interacting.remove((n1, n2))
        for int_nodes in current_interacting:
            interaction_csv.append([timestamp / resolution, int_nodes[0], int_nodes[1]])

    df = pd.DataFrame(interaction_csv, columns=["timestamp", "node1", "node2"])
    df["timestamp"] = df["timestamp"].round().astype(int)
    df_aggregated = (
        df.groupby("timestamp")[["node1", "node2"]]
        .apply(lambda x: np.unique(x.values, axis=0))
        .reset_index())

    all_timestamps = np.arange(df_aggregated["timestamp"].min(), df_aggregated["timestamp"].max() + 1)
    prev_interactions = None
    output_lines = []
    for timestamp in all_timestamps:
        if timestamp in df_aggregated["timestamp"].values:
            interactions = df_aggregated[df_aggregated["timestamp"] == timestamp].values[0][1]
            prev_interactions = interactions
        else:
            interactions = prev_interactions
        for interaction in interactions:
            output_lines.append(f"{timestamp} {int(interaction[0])} {int(interaction[1])}")
    with open(fname, 'w') as f:
        f.write("\n".join(output_lines))


def save_dict(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, 4)


def load_dict(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    results = simulate_temporal_network(15, Tend=500, case='first_order')
    print('%d interactions simulated' % len(results['interaction_times']))
