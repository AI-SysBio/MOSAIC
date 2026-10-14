"""
MOSAIC: scalable stochastic simulation with heterogeneous and non-Markovian agents.

Reference implementation of the rejection-based Gillespie algorithm for non-Markovian
reactions (originally named REGIR) described in

    Pelissier et al., "Unifying non-Markovian dynamics and agent heterogeneity in
    scalable stochastic networks", Nature Communications 17, 3345 (2026).
    https://doi.org/10.1038/s41467-026-69817-y

Model definition
----------------
A system is a set of species (``N_init`` gives their initial populations) and a list of
``Reaction_channel`` objects. Every channel has reactants, products, a rate (the inverse
of the mean waiting time) and a waiting-time distribution with a dimensionless shape
parameter. Each individual reactant is an object with its own clock (the time since it
became available to each channel), which is what makes agent-level properties possible.

Simulation schemes (per channel)
--------------------------------
* rejection (default, the MOSAIC / REGIR scheme): events are proposed at an upper bound
  ``rmax`` of the hazard and accepted with probability hazard / rmax. Available for channels
  with exactly one reactant and a hazard that is finite at t = 0.
* precomputed delays (``precompute_delay=True``, the DelaySSA scheme): the waiting time of
  each reactant is drawn when it is created and kept in a sorted queue. Exact for any
  distribution, at the cost of O(N log N) scaling.
* channels with zero or several reactants are Poisson processes with the given rate.

Agent-level properties
----------------------
``perform_reaction_custom`` replaces the default bookkeeping of a channel by a user
function that chooses the reactants, creates the products and updates their properties;
``r0_function`` makes a rate depend on the state of the system; ``transfer_identity``
keeps the identity (and clocks) of a reactant that is also a product. The hooks
``update_rates``, ``on_reinitialize`` and ``record_extra`` of ``Gillespie_simulation``
can be overridden by subclasses (see the application folders).

For a pip-installable, actively maintained implementation of the same methods see
NoMaSS: https://github.com/AI-SysBio/NoMaSS
"""
import random
import uuid
from math import log, gamma, exp, pi, sqrt, erf, atan

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.special import gammainc
from sortedcontainers import SortedDict

END_OF_TIME = 10 ** 27        # sentinel key of the (always non-empty) queue of precomputed delays
MAX_POPULATION = 1_000_000    # the simulation stops advancing events above this total population
PRECOMPUTED_PROPENSITY = 1e-14  # negligible propensity of channels whose events come from the delay queue


# =============================================================================
# Waiting-time distributions
# =============================================================================
# Every distribution is parameterised by ``rate`` (1/mean, 1/median for Cauchy) and a
# dimensionless ``shape`` parameter. For each one we provide:
#   hazard(t, rate, shape)    the rate of the event at time t since the reactant appeared,
#   sample(rate, shape, size) random waiting times,
#   pdf(t, rate, shape)       the theoretical density (plots only),
#   fit(wait_times)           a maximum-likelihood fit (plots only).

def hazard_exponential(t, rate, shape):
    return rate


def hazard_weibull(t, rate, shape):
    # only finite at t = 0 for shape >= 1
    alpha = shape
    beta = (alpha) * (rate * gamma((alpha + 1) / (alpha))) ** (alpha)
    return (t ** (alpha - 1)) * beta


def hazard_gamma(t, rate, shape):
    alpha = shape
    if alpha < 1 and t == 0:
        return 0
    beta = alpha * rate
    pdf = (beta ** alpha) * (t ** (alpha - 1)) * exp(-beta * t) / gamma(alpha)
    cdf = gammainc(alpha, beta * t)
    return pdf / (1 - cdf)


def hazard_normal(t, rate, shape):
    # shape is the ratio std / mean
    if shape < 1 and t == 0:
        return 0
    sigma = 1 / rate * shape
    mu = 1 / rate
    if t > mu + 7 * sigma:  # the cdf is not precise enough in the far tail
        return (t - mu) / sigma ** 2
    pdf = 1 / (sigma * sqrt(2 * pi)) * exp(-(1 / 2) * (t - mu) ** 2 / (sigma ** 2))
    cdf = 1 / 2 * (1 + erf((t - mu) / (sigma * sqrt(2))))
    return pdf / (1 - cdf)


def hazard_lognormal(t, rate, shape):
    # shape is the ratio std / mean
    sigma0 = 1 / rate * shape
    mu0 = 1 / rate
    mu = log(mu0 ** 2 / sqrt(mu0 ** 2 + sigma0 ** 2))
    sigma = sqrt(log(1 + sigma0 ** 2 / mu0 ** 2))
    if t == 0:
        return 0
    pdf = 1 / (t * sigma * sqrt(2 * pi)) * exp(-(1 / 2) * (log(t) - mu) ** 2 / (sigma ** 2))
    cdf = 1 / 2 * (1 + erf((log(t) - mu) / (sigma * sqrt(2))))
    return pdf / (1 - cdf)


def hazard_cauchy(t, rate, shape):
    # shape is the scale parameter gamma of the Cauchy distribution, rate = 1/median
    gam = shape
    mu = 1 / rate
    pdf = 1 / (pi * gam * (1 + ((t - mu) / gam) ** 2))
    cdf = (1 / pi) * atan((t - mu) / gam) + 1 / 2
    return pdf / (1 - cdf)


def hazard_pareto(t, rate, shape):
    # Pareto type I with tail index ``shape``; the minimum is set so that the median is 1/rate
    alpha = shape
    mu = 1 / rate * 2 ** (-1 / alpha)
    if t < mu:
        return 0
    return alpha / t


def _lognormal_params(rate, shape):
    mu0 = 1 / rate
    sigma0 = mu0 * shape
    mu = log(mu0 ** 2 / sqrt(mu0 ** 2 + sigma0 ** 2))
    sigma = sqrt(log(1 + sigma0 ** 2 / mu0 ** 2))
    return mu0, mu, sigma


def _weibull_beta(rate, alpha):
    return (alpha) * (rate * gamma((alpha + 1) / (alpha))) ** (alpha)


def sample_exponential(rate, shape, size):
    return stat.expon.rvs(loc=0, scale=1 / rate, size=size)


def sample_weibull(rate, shape, size):
    alpha = shape
    if shape == 1:
        return stat.expon.rvs(loc=0, scale=1 / rate, size=size)
    beta = _weibull_beta(rate, alpha)
    return stat.weibull_min.rvs(alpha, loc=0, scale=(alpha / beta) ** (1 / alpha), size=size)


def sample_normal(rate, shape, size):
    mu = 1 / rate
    return stat.norm.rvs(loc=mu, scale=mu * shape, size=size)


def sample_gamma(rate, shape, size):
    alpha = shape
    return stat.gamma.rvs(alpha, loc=0, scale=1 / (alpha * rate), size=size)


def sample_lognormal(rate, shape, size):
    mu0, mu, sigma = _lognormal_params(rate, shape)
    return stat.lognorm.rvs(sigma, loc=0, scale=mu0, size=size)


def sample_cauchy(rate, shape, size):
    return stat.cauchy.rvs(loc=1 / rate, scale=shape, size=size)


def sample_pareto(rate, shape, size):
    alpha = shape
    mu = 1 / rate * 2 ** (-1 / alpha)
    return stat.pareto.rvs(alpha, scale=mu, size=size)


def pdf_exponential(t, rate, shape):
    return rate * np.exp(-rate * t)


def pdf_weibull(t, rate, shape):
    alpha = shape
    beta = _weibull_beta(rate, alpha)
    return beta * np.power(t, alpha - 1) * np.exp(-beta * np.power(t, alpha) / (alpha))


def pdf_normal(t, rate, shape):
    mu = 1 / rate
    sigma = mu * shape
    return 1 / (sigma * sqrt(2 * pi)) * np.exp(-(1 / 2) * np.power((t - mu) / sigma, 2))


def pdf_gamma(t, rate, shape):
    alpha = shape
    beta = alpha * rate
    return (beta ** alpha) * np.power(t, alpha - 1) * np.exp(-beta * t) / gamma(alpha)


def pdf_lognormal(t, rate, shape):
    mu0, mu, sigma = _lognormal_params(rate, shape)
    return 1 / (t * sigma * sqrt(2 * pi)) * np.exp(-(1 / 2) * np.power((np.log(t) - mu) / sigma, 2))


def pdf_cauchy(t, rate, shape):
    gam = shape
    mu = 1 / rate
    return 1 / (pi * gam * (1 + ((t - mu) / gam) ** 2))


def pdf_pareto(t, rate, shape):
    alpha = shape
    mu = 1 / rate * 2 ** (-1 / alpha)
    pdf = alpha * mu ** alpha / t ** (alpha + 1)
    pdf[t < mu] = 0
    return pdf


def fit_exponential(wait_times):
    return stat.expon, stat.expon.fit(wait_times, floc=0)


def fit_weibull(wait_times):
    return stat.weibull_min, stat.weibull_min.fit(wait_times, 3, floc=0, scale=30)


def fit_normal(wait_times):
    return stat.norm, stat.norm.fit(wait_times)


def fit_gamma(wait_times):
    return stat.gamma, stat.gamma.fit(wait_times, floc=0)


def fit_lognormal(wait_times):
    return stat.lognorm, stat.lognorm.fit(wait_times, floc=0)


def fit_cauchy(wait_times):
    return stat.cauchy, stat.cauchy.fit(wait_times)


def fit_pareto(wait_times):
    return stat.pareto, stat.pareto.fit(wait_times, floc=0)


DISTRIBUTIONS = {
    'Exponential': dict(aliases=('exponential', 'exp'), hazard=hazard_exponential, sample=sample_exponential, pdf=pdf_exponential, fit=fit_exponential),
    'Weibull':     dict(aliases=('weibull', 'weib'), hazard=hazard_weibull, sample=sample_weibull, pdf=pdf_weibull, fit=fit_weibull),
    'Gamma':       dict(aliases=('gamma', 'gam'), hazard=hazard_gamma, sample=sample_gamma, pdf=pdf_gamma, fit=fit_gamma),
    'Normal':      dict(aliases=('gaussian', 'normal', 'norm'), hazard=hazard_normal, sample=sample_normal, pdf=pdf_normal, fit=fit_normal),
    'LogNormal':   dict(aliases=('lognormal', 'lognorm'), hazard=hazard_lognormal, sample=sample_lognormal, pdf=pdf_lognormal, fit=fit_lognormal),
    'Cauchy':      dict(aliases=('cauchy', 'cau'), hazard=hazard_cauchy, sample=sample_cauchy, pdf=pdf_cauchy, fit=fit_cauchy),
    'Pareto':      dict(aliases=('pareto', 'par'), hazard=hazard_pareto, sample=sample_pareto, pdf=pdf_pareto, fit=fit_pareto),
}


def canonical_distribution(name):
    """Return the canonical distribution name for any accepted alias, or None."""
    for canonical, spec in DISTRIBUTIONS.items():
        if name.lower() in spec['aliases']:
            return canonical
    return None


def generate_delay_from_distribution(reaction_channel, N_=None):
    """Draw N_ waiting times for a channel (a single float when N_ is None)."""
    size = 1 if N_ is None else N_
    spec = DISTRIBUTIONS[canonical_distribution(reaction_channel.distribution)]
    delay = spec['sample'](reaction_channel.rate, reaction_channel.shape_param, size)
    return delay[0] if N_ is None else delay


# =============================================================================
# Reaction channel and reactant
# =============================================================================

class Reaction_channel():
    """
    One reaction of the system.

    Parameters
    ----------
    param_simulation : object with attribute ``Tend`` (used to bound non-monotonic hazards).
    rate : float
        Inverse of the mean waiting time (median for Cauchy). A rate of 0 switches the
        channel off.
    shape_param : float
        Dimensionless shape parameter of the distribution (see ``DISTRIBUTIONS``).
    rmax : float, optional
        Upper bound of the hazard used by the rejection step. By default it is derived
        from the distribution (and updated during the simulation for increasing hazards).
    distribution : str
        One of the names or aliases in ``DISTRIBUTIONS``.
    precompute_delay : bool
        Use the delay queue (DelaySSA scheme) instead of the rejection scheme. Needed for
        hazards that diverge at t = 0 (Gamma and Weibull with shape < 1).
    name : str
    reactants, products : list of species names (assignable after construction).
    transfer_identity : bool
        A reactant that also appears among the products keeps its identity and its clocks
        for the other channels (for example ``M -> M + P``).
    perform_reaction_custom : callable(simulation, forced_ID=None) -> (consumed, created)
        User bookkeeping replacing the default one; ``consumed`` and ``created`` map
        reactant ids to species names. Returning two empty dicts rejects the event.
    r0_function : callable(population_dict, t) -> rate
        State-dependent rate, evaluated before every step.

    Attributes
    ----------
    wait_times : list of the waiting times of the accepted events (plots and statistics).
    number_of_accepted_reactions, number_of_rejected_reactions : counters.
    """

    def __init__(self, param_simulation, rate=1, shape_param=1, rmax=None, distribution='Weibull',
                 precompute_delay=False, name='', reactants=None, products=None, transfer_identity=False,
                 perform_reaction_custom=None, r0_function=None):
        self.reactants = list(reactants) if reactants is not None else []
        self.products = list(products) if products is not None else []
        self.rate = rate
        self.shape_param = shape_param
        self.name = name
        self.distribution = distribution
        self.transfer_identity = transfer_identity
        self.precompute_delay = precompute_delay
        self.use_custom_function = perform_reaction_custom is not None
        self.perform_reaction_custom = perform_reaction_custom
        self.r0_function = r0_function

        self.rmax = rate if rmax is None else rmax
        self.wait_times = []
        self.number_of_rejected_reactions = 0
        self.number_of_accepted_reactions = 0

        if self.use_custom_function and rmax is None:
            print(' Reaction %s (Custom function):' % name)
            print('   rmax will be set as rate, make sure rmax is always above the rate of all your individual properties')

        self._configure_distribution(param_simulation)
        self.temp_rmax = self.rmax  # bound actually used by the rejection step (may be inflated by min_ratio)

    def _configure_distribution(self, param_simulation):
        """Set the hazard function, the initial upper bound ``rmax`` and whether it is fixed."""
        rate, shape_param, name = self.rate, self.shape_param, self.name
        canonical = canonical_distribution(self.distribution)

        if rate < 0:
            raise ValueError('Reaction %s: Rate cannot be negative' % (name,))

        if self.use_custom_function:
            self.rmax_fixed = True
            return

        if rate == 0 or canonical == 'Exponential':
            self._set_exponential()
            return

        if canonical is None:
            raise ValueError('Unsupported distribution: %s' % (self.distribution,))
        self.rate_function = DISTRIBUTIONS[canonical]['hazard']

        if canonical in ('Weibull', 'Gamma'):
            # increasing hazard: rmax is tracked from the oldest reactant during the simulation
            self.rmax_fixed = False
            if shape_param < 1 and self.precompute_delay == False:
                raise ValueError('Reaction %s: Shape parameter < 1 for %s distribution is not supported by the rejection method (the rate diverges at t=0); use precompute_delay = True instead' % (name, canonical))
            if shape_param == 1:
                self._set_exponential()

        elif canonical == 'Normal':
            self.rmax_fixed = False
            if shape_param <= 0:
                raise ValueError('Reaction %s: Zero or negative variance for Normal is not supported' % (name,))

        elif canonical == 'LogNormal':
            if shape_param <= 0:
                raise ValueError('Reaction %s: Zero or negative variance for LogNormal is not supported' % (name,))
            elif shape_param >= 0.25:
                # non-monotonic hazard: bound it on the simulation window
                self.rmax_fixed = True
                t = np.linspace(0, param_simulation.Tend, num=100)
                rate_t = np.array([hazard_lognormal(ti, rate, shape_param) for ti in t])
                self.rmax = np.nanmax(rate_t[rate_t != np.inf])
            else:
                # the maximum is very high and far away: treat the hazard as increasing
                self.rmax_fixed = False

        elif canonical == 'Cauchy':
            self.rmax_fixed = True
            if shape_param <= 0:
                raise ValueError('Reaction %s: Zero or negative variance for Cauchy is not supported' % (name,))
            rate_t = np.array([hazard_cauchy(ti, rate, shape_param) for ti in np.linspace(0, param_simulation.Tend, num=100)])
            self.rmax = np.max(rate_t)

        elif canonical == 'Pareto':
            self.rmax_fixed = True
            if shape_param <= 0:
                raise ValueError('Reaction %s: Zero or negative shape parameter for Pareto is not supported' % (name,))
            t0 = 1 / rate * 2 ** (-1 / shape_param)
            self.rmax = shape_param / t0  # the hazard alpha/t is maximal at the minimum t0

    def _set_exponential(self):
        self.distribution = 'Exponential'
        self.rate_function = hazard_exponential
        self.rmax_fixed = True


class Reactant():
    """An individual reactant, identified by a unique id (a 32-character hex uuid)."""

    def __init__(self, ID=None):
        self.id = self.gen_uuid() if ID is None else ID

    def gen_uuid(self):
        return uuid.UUID(int=random.getrandbits(128), version=4).hex


# =============================================================================
# Simulation
# =============================================================================

class Gillespie_simulation():
    """
    Stochastic simulation of a reaction network with individual reactants.

    Parameters
    ----------
    N_init : dict species name -> initial population.
    param : object with attributes ``Tend``, ``N_simulations``, ``timepoints`` and ``unit``.
    reaction_channel_list : list of ``Reaction_channel`` (can also be assigned afterwards).
    min_ratio : float
        Accuracy of the rejection scheme for small systems: the proposal rate of a
        non-Markovian channel is kept at least ``min_ratio`` times larger than its hazard
        bound relative to the total propensity (see the error bound in the paper).
    print_warnings : bool

    Main state (available during and after a run)
    ---------------------------------------------
    reactant_population : dict species -> current population
    reactant_list : dict species -> list of ``Reactant`` objects
    reactant_times : dict species -> channel name -> reactant id -> time at which the
        reactant became available to that channel (its clock)
    delayed_reactions : sorted dict time -> [channel index, reactant id] (delay queue)
    population_t : recorded populations of the last realisation, shape (timepoints, n_species + 1)
    population_compiled : recorded populations of all realisations
    """

    def __init__(self, N_init, param, reaction_channel_list=None, min_ratio=10, print_warnings=False):
        self.param = param
        self.param.min_ratio = min_ratio
        self.param.print_warnings = print_warnings
        self.reaction_channel_list = reaction_channel_list if reaction_channel_list is not None else []
        self.reactant_population_init = N_init

    # ------------------------------------------------------------------ hooks for application-specific models
    def update_rates(self):
        """Called at every step before the propensities: apply state-dependent rates (r0_function)."""
        for reaction_channel in self.reaction_channel_list:
            if reaction_channel.r0_function is not None:
                reaction_channel.rate = reaction_channel.r0_function(self.reactant_population, self.t)

    def on_reinitialize(self):
        """Called after the population is reset at the start of each realisation (e.g. to draw agent properties)."""
        pass

    def record_extra(self, ti):
        """Called whenever the populations are recorded at grid index ti-1 (e.g. to store agent properties)."""
        pass

    # ------------------------------------------------------------------ initialisation
    def reinitialize_pop(self):
        """Reset the populations and the delay queue to the initial state (waiting-time statistics are kept)."""
        self.reaction_channel_index = {channel.name: ri for ri, channel in enumerate(self.reaction_channel_list)}
        self.reactant_population = self.reactant_population_init.copy()
        self.reactant_list, self.reactant_times = self.initialise_reactant_list()
        self.on_reinitialize()

    def initialise_reactant_list(self):
        """Create the initial reactants, with all clocks at 0, and schedule their precomputed delays."""
        N_init = self.reactant_population
        reactant_list = dict()
        reactant_times = dict()
        for reactant_str in N_init:
            react_i = []
            react_times_i = {channel.name: dict() for channel in self.reaction_channel_list}
            for j in range(N_init[reactant_str]):
                new_reactant = Reactant()
                react_i.append(new_reactant)
                for channel in self.reaction_channel_list:
                    react_times_i[channel.name][new_reactant.id] = 0
            reactant_list[reactant_str] = react_i
            reactant_times[reactant_str] = react_times_i

        self.delayed_reactions = SortedDict({END_OF_TIME: [-1, -1]})
        self.id_delay = {channel.name: dict() for channel in self.reaction_channel_list}
        for reactant_str in N_init:
            if N_init[reactant_str] > 0:
                for channel in self.reactions_list_per_reactant[reactant_str]:
                    if channel.precompute_delay:
                        t_delay = generate_delay_from_distribution(channel, N_init[reactant_str])
                        for j in range(N_init[reactant_str]):
                            reactant_id = reactant_list[reactant_str][j].id
                            self.delayed_reactions[t_delay[j]] = [self.reaction_channel_index[channel.name], reactant_id]
                            self.id_delay[channel.name][reactant_id] = t_delay[j]
        return reactant_list, reactant_times

    def _check_channels(self):
        """Index the channels by reactant and validate the options that need the full channel list."""
        self.reactions_list_per_reactant = {reactant_str: [] for reactant_str in self.reactant_population_init}
        for reaction_channel in self.reaction_channel_list:
            for reactant in reaction_channel.reactants:
                self.reactions_list_per_reactant[reactant].append(reaction_channel)

        # a reactant involved in several channels must be looked up by id when a delayed event fires
        self.overlapping_reactants = any(len(channels) > 1 for channels in self.reactions_list_per_reactant.values())

        for channel in self.reaction_channel_list:
            if channel.precompute_delay and len(channel.reactants) != 1:
                print('   Warning: delays can only be precomputed for channels with exactly one reactant,')
                print('      setting precompute_delay = False for %s' % channel.name)
                channel.precompute_delay = False
            if channel.transfer_identity and len(set(channel.products)) < len(channel.products):
                raise ValueError('Setting transfer_identity to True for channels where products are of the same kind is ambiguous due to duplicated IDs in the dictionary')

    # ------------------------------------------------------------------ running
    def run_simulations(self, Tend, verbose=True, delay_method='rejection', seed=None):
        """
        Run ``param.N_simulations`` independent realisations and return the recorded
        populations as an array of shape (N_simulations, timepoints, n_species + 1); the
        last column is the total population.

        ``delay_method`` selects the event loop: 'rejection' (MOSAIC) or 'Anderson'
        (modified next-reaction method, used for the DelaySSA comparisons). Pass ``seed``
        for reproducible runs.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._check_channels()

        population = np.empty((self.param.N_simulations, self.param.timepoints, len(self.reactant_population_init) + 1))
        for ni in range(self.param.N_simulations):
            if verbose:
                print("   Simulation", ni + 1, "/", self.param.N_simulations, "...")
            self.reinitialize_pop()
            self.run_simulation(Tend, delay_method=delay_method)
            population[ni, :, :] = self.get_populations()
        self.population_compiled = population
        return population

    def run_simulation(self, Tend, delay_method='rejection'):
        """
        One realisation from the current state until Tend. Populations are recorded on a
        grid of ``param.timepoints`` times: the value at a grid time is the state that held
        at that time (the state before the first event that follows it).
        """
        timepoints = self.param.timepoints
        self.t = 0
        ti = 0
        self.Tend = Tend
        self.population_t = -np.ones((timepoints, len(self.reactant_population) + 1))
        prev_state = list(self.reactant_population.values())  # state holding during the current step

        if delay_method == 'Anderson':
            M = len(self.reaction_channel_list)
            T_array = np.zeros(M)                      # internal times of the channels
            P_array = np.log(1 / np.random.rand(M))    # next firing times in internal time

        while self.t < Tend:

            self.update_rates()

            if self.t >= ti * Tend / timepoints:  # a grid point was crossed: record the state that held before this event
                self.population_t[ti, :-1] = prev_state
                ti = int(self.t / Tend * (timepoints)) + 1
                self.record_extra(ti)
            prev_state = list(self.reactant_population.values())

            a_array = self.compute_propensities()
            a0 = np.sum(a_array)

            if a0 == 0:  # nothing can happen: advance the clock
                self.t += Tend / timepoints / 2

            elif sum(self.reactant_population.values()) > MAX_POPULATION:  # avoid exploding simulations
                self.t += Tend / timepoints / 2

            elif delay_method == 'Anderson':
                Delta_array = (P_array - T_array) / a_array
                t_delay_min, (mu, reactant_id) = next(iter(self.delayed_reactions.items()))
                tau = np.min(Delta_array)
                if t_delay_min - self.t <= tau:
                    tau = t_delay_min - self.t
                    self.t += tau
                    self.perform_reaction(mu, reactant_id)
                    self.reaction_channel_list[mu].number_of_rejected_reactions += 1
                else:
                    mu = np.argmin(Delta_array)
                    self.t += tau
                    self.perform_reaction(mu)
                P_array[mu] += log(1 / np.random.rand())
                T_array += (a_array * tau)

            else:  # rejection scheme
                # time to the next candidate event
                r1 = random.random()
                tau = 1 / a0 * log(1 / r1)
                # a precomputed delay expires first: that event is certain
                t_delay_min, (mu, reactant_id) = next(iter(self.delayed_reactions.items()))
                if t_delay_min < self.t + tau:
                    self.t = t_delay_min
                    self.perform_reaction(mu, reactant_id)
                else:
                    self.t += tau
                    # choose the channel proportionally to its propensity, then accept or reject
                    r2 = random.random()
                    mu = 0
                    p_sum = 0.0
                    while p_sum < r2 * a0:
                        p_sum += a_array[mu]
                        mu += 1
                    mu = mu - 1
                    self.perform_reaction(mu)

        if ti < timepoints:  # the state of the last step holds until Tend
            self.population_t[ti, :-1] = prev_state
        self.population_t = interpolate_minusones(self.population_t)
        self.population_t[:, -1] = np.sum(self.population_t, axis=1)

    def compute_propensities(self):
        """
        Propensity of every channel: number of reactant combinations times the hazard
        bound ``rmax``. Channels fed by the delay queue get a negligible propensity.
        For small systems the bound of a non-Markovian channel is inflated so that the
        total propensity stays at least ``min_ratio`` times its bound (accuracy control).
        """
        M = len(self.reaction_channel_list)
        propensity = np.zeros(M)
        N_reactants = np.zeros(M)
        self.update_all_rmax()

        for ri, reaction_channel in enumerate(self.reaction_channel_list):
            if len(reaction_channel.reactants) > 0:
                if len(set(reaction_channel.reactants)) == 1 and len(reaction_channel.reactants) == 2:
                    NA = self.reactant_population[reaction_channel.reactants[0]]
                    Nreactants = NA * (NA - 1) / 2  # A + A: number of distinct pairs
                else:
                    Nreactants = np.prod([self.reactant_population[reactant_str] for reactant_str in reaction_channel.reactants])
            else:
                Nreactants = 1
            N_reactants[ri] = Nreactants
            propensity[ri] = Nreactants * reaction_channel.rmax

        for ri, reaction_channel in enumerate(self.reaction_channel_list):
            if propensity[ri] > 0:
                ratio = np.sum(propensity) / reaction_channel.rmax
                if np.isnan(ratio) or np.isinf(ratio):
                    raise RuntimeError('rmax should never be zero: check the reaction rates')
                if reaction_channel.precompute_delay:
                    propensity[ri] = PRECOMPUTED_PROPENSITY
                elif ratio < self.param.min_ratio and reaction_channel.distribution != 'Exponential':
                    reaction_channel.temp_rmax = reaction_channel.rmax * self.param.min_ratio / ratio
                    propensity[ri] = N_reactants[ri] * reaction_channel.temp_rmax
        return propensity

    def update_all_rmax(self):
        """
        Refresh the hazard bound of channels with an increasing hazard: the clocks of a
        channel are stored in insertion order, so the oldest reactant has the largest hazard.
        """
        for reaction_channel in self.reaction_channel_list:
            if len(reaction_channel.reactants) > 0:
                reactant_str = reaction_channel.reactants[0]
                if len(self.reactant_list[reactant_str]) > 0 and reaction_channel.rmax_fixed == False and not reaction_channel.precompute_delay:
                    first_reactant_ID, t_start_min = next(iter(self.reactant_times[reactant_str][reaction_channel.name].items()))
                    tau_max = self.t - t_start_min
                    rmax = reaction_channel.rate_function(tau_max, reaction_channel.rate, reaction_channel.shape_param)
                    if np.isnan(rmax) or np.isinf(rmax):
                        if self.param.print_warnings:
                            print('Problem computed rmax is', rmax, 'for time %.2e' % tau_max, 'and reaction', reaction_channel.name)
                    elif rmax > reaction_channel.rate:
                        reaction_channel.rmax = rmax
            reaction_channel.temp_rmax = reaction_channel.rmax

    # ------------------------------------------------------------------ reactions
    def find_in_reactant_list(self, reactant_list, reactant_id):
        """Index of the reactant with the given id in a list of reactants (O(N)), or None."""
        for ri, reac in enumerate(reactant_list):
            if reac.id == reactant_id:
                return ri
        return None

    def get_random_element(self, a_huge_key_list, weights=None):
        """A uniformly (or weighted) random element of a list and its index."""
        L = len(a_huge_key_list)
        if weights is None:
            i = np.random.randint(0, L)
        else:
            weights = weights / np.sum(weights)
            i = np.random.choice(np.arange(L), p=weights)
        return i, a_huge_key_list[i]

    def perform_reaction(self, mu, reactant_id=None):
        """
        Attempt the event of channel ``mu``. With ``reactant_id`` the event comes from the
        delay queue and is certain; otherwise it is accepted or rejected (rejection step).
        """
        reaction_channel = self.reaction_channel_list[mu]
        if reaction_channel.use_custom_function:
            self._perform_custom_reaction(reaction_channel, reactant_id)
        else:
            self._perform_standard_reaction(reaction_channel, reactant_id)

    def _perform_custom_reaction(self, reaction_channel, reactant_id):
        """Delegate the bookkeeping to the user function and update clocks and delays."""
        force_reaction = reactant_id is not None
        consumed, created = reaction_channel.perform_reaction_custom(self, forced_ID=reactant_id)

        if len(consumed) == 0 and len(created) == 0:
            reaction_channel.number_of_rejected_reactions += 1
            if len(reaction_channel.reactants) == 1 and force_reaction == True:
                raise RuntimeError('A delayed reaction cannot be rejected: you need to adjust your custom function to always execute the reaction at the end of the delay')
            return

        reaction_channel.number_of_accepted_reactions += 1
        for (consumed_id, reactant_str) in consumed.items():
            Dt = self.t - self.reactant_times[reactant_str][reaction_channel.name][consumed_id]
            reaction_channel.wait_times.append(Dt)
            for channel in self.reaction_channel_list:
                self.reactant_times[reactant_str][channel.name].pop(consumed_id, None)
        for (consumed_id, reactant_str) in consumed.items():
            self._cancel_delays(consumed_id)
        for (product_id, product_str) in created.items():
            for channel in self.reaction_channel_list:
                self.reactant_times[product_str][channel.name][product_id] = self.t
        for (product_id, product_str) in created.items():
            self._schedule_delays(product_id, self.reactions_list_per_reactant[product_str])

    def _perform_standard_reaction(self, reaction_channel, reactant_id):
        """Default bookkeeping: rejection step, removal of the reactants, creation of the products."""
        force_reaction = reactant_id is not None
        reject_reaction = True

        if len(reaction_channel.reactants) != 1:
            # zero or several reactants: Poisson process (the waiting time of an individual is undefined)
            Dt = 0
            if reaction_channel.rate >= reaction_channel.temp_rmax * random.random():
                reject_reaction = False
        elif force_reaction:
            # event from the delay queue
            reject_reaction = False
            reactant_str = reaction_channel.reactants[0]
            if self.overlapping_reactants:
                index = self.find_in_reactant_list(self.reactant_list[reactant_str], reactant_id)
            else:
                index = 0  # without overlap the reactants of a species are interchangeable
            if reactant_id not in self.reactant_times[reactant_str][reaction_channel.name]:
                return  # already consumed by another channel: discard the delayed event
            Dt = self.t - self.reactant_times[reactant_str][reaction_channel.name][reactant_id]
        else:
            # rejection step on a random reactant of the species
            reactant_str = reaction_channel.reactants[0]
            index, reactant = self.get_random_element(self.reactant_list[reactant_str])
            reactant_id = reactant.id
            if reactant_id not in self.reactant_times[reactant_str][reaction_channel.name]:
                return
            Dt = self.t - self.reactant_times[reactant_str][reaction_channel.name][reactant_id]
            reactant_rate = reaction_channel.rate_function(Dt, reaction_channel.rate, reaction_channel.shape_param)
            if np.isnan(reactant_rate) or np.isinf(reactant_rate):
                if self.param.print_warnings:
                    print('Problem: computed rate is', reactant_rate, 'for time %.2e' % Dt, 'and reaction', reaction_channel.name)
                reactant_rate = reaction_channel.temp_rmax  # accept
            if reactant_rate >= reaction_channel.temp_rmax * random.random():
                reject_reaction = False

        if reject_reaction:
            reaction_channel.number_of_rejected_reactions += 1
            return

        reaction_channel.number_of_accepted_reactions += 1
        reaction_channel.wait_times.append(Dt)

        # consume the reactants
        for reactant_str in reaction_channel.reactants:
            if not force_reaction and len(reaction_channel.reactants) != 1:
                index, reactant = self.get_random_element(self.reactant_list[reactant_str])
                reactant_id = reactant.id
            keeps_identity = (reactant_str in reaction_channel.products) and reaction_channel.transfer_identity
            if not keeps_identity:
                self.reactant_list[reactant_str].pop(index)
                for channel_i in self.reaction_channel_list:
                    self.reactant_times[reactant_str][channel_i.name].pop(reactant_id, None)
                self.reactant_population[reactant_str] -= 1
            if force_reaction:
                if not self.overlapping_reactants or keeps_identity:
                    self.delayed_reactions.popitem(index=0)  # the event that just fired is the first of the queue
                else:
                    self._cancel_delays(reactant_id)
            elif not keeps_identity:
                self._cancel_delays(reactant_id)

        # create the products
        for product_str in reaction_channel.products:
            keeps_identity = (product_str in reaction_channel.reactants) and reaction_channel.transfer_identity
            product = Reactant(ID=reactant_id) if keeps_identity else Reactant()
            product_id = product.id
            if not keeps_identity:
                self.reactant_list[product_str].append(product)
                for channel_i in self.reaction_channel_list:
                    self.reactant_times[product_str][channel_i.name][product_id] = self.t
                self.reactant_population[product_str] += 1
            self.reactant_times[product_str][reaction_channel.name][product_id] = self.t  # this channel's clock restarts
            if keeps_identity:
                self._schedule_delays(product_id, [reaction_channel])  # the other clocks are untouched
            else:
                self._schedule_delays(product_id, self.reactions_list_per_reactant[product_str])

    def _schedule_delays(self, reactant_id, channels):
        """Draw and queue the waiting times of a new reactant for the channels using the delay queue."""
        for channel in channels:
            if channel.precompute_delay:
                t_delay = generate_delay_from_distribution(channel)
                self.delayed_reactions[self.t + t_delay] = [self.reaction_channel_index[channel.name], reactant_id]
                self.id_delay[channel.name][reactant_id] = self.t + t_delay

    def _cancel_delays(self, reactant_id):
        """Remove the queued events of a reactant that has been consumed."""
        for channel in self.reaction_channel_list:
            if channel.precompute_delay and reactant_id in self.id_delay[channel.name]:
                time_delay = self.id_delay[channel.name][reactant_id]
                self.delayed_reactions.pop(time_delay)
                self.id_delay[channel.name].pop(reactant_id)

    # ------------------------------------------------------------------ results
    def get_populations(self):
        return self.population_t

    def get_reactant_list(self):
        return self.reactant_list

    def get_reactant_IDs(self):
        return {reactant_str: [r.id for r in reactants] for reactant_str, reactants in self.reactant_list.items()}

    def get_channel_waiting_times(self):
        return [channel.wait_times for channel in self.reaction_channel_list]

    # ------------------------------------------------------------------ plots
    def plot_populations_single(self):
        """Populations of the last realisation."""
        timepoints = self.population_t.shape[0]
        time_points = np.linspace(0, self.Tend, timepoints)
        plt.figure(figsize=(7, 4))
        plt.rcParams.update({'font.size': 16})
        for ri, reactant in enumerate(self.reactant_population.keys()):
            plt.plot(time_points, self.population_t[:, ri], lw=2, alpha=1, color=sns.color_palette()[ri], label=reactant)
        plt.xlabel('Time [%s]' % self.param.unit)
        plt.ylabel('Population')
        plt.legend()
        plt.show()

    def plot_populations(self, reactant_list=None, log_scale=False, color_list=None, figsize=(6.2, 4.2)):
        """Mean population over the realisations (thick lines) and individual realisations (thin lines)."""
        color_list = list(color_list) if color_list is not None else []
        reactant_poplist = list(self.reactant_population_init.keys()) + ['Total']
        if reactant_list is None:
            reactant_list = reactant_poplist
        for i in range(len(color_list), len(reactant_list)):
            color_list.append(sns.color_palette()[i + 1 - len(color_list)])

        population = self.population_compiled
        N_simulations, timepoints, N_reactants = population.shape
        time_points = np.linspace(0, self.Tend, timepoints)
        plt.figure(figsize=figsize)
        plt.rcParams.update({'font.size': 16})
        rii = 0
        for ri in range(N_reactants):
            if reactant_poplist[ri] in reactant_list:
                for i in range(N_simulations):
                    plt.plot(time_points, population[i, :, ri], lw=0.3, alpha=0.1, color=sns.color_palette()[0])
                plt.plot(time_points, population[:, :, ri].mean(axis=0), lw=3, color=color_list[rii], label=reactant_poplist[ri])
                rii += 1
        plt.xlabel('Time [%s]' % self.param.unit)
        plt.ylabel('Population')
        plt.legend()
        if log_scale:
            plt.yscale('log')
        plt.show()

    def plot_inter_event_time_distribution(self, color_list=None, plot_fitted=True, plot_theory=True, theory_color='green',
                                           fitted_color='red', bins=None, figsize=(6.2, 4.2), fontsize=16, lw=2):
        """
        Histogram of the waiting times of every single-reactant channel, with the theoretical
        density of the channel (dashed) and a maximum-likelihood fit (solid), plus a summary
        of the accepted and rejected events.
        """
        plt.rcParams.update({'font.size': fontsize})
        for ri, reaction_channel in enumerate(self.reaction_channel_list):
            wait_times = np.array(reaction_channel.wait_times)
            print("\n Reaction channel %s:" % reaction_channel.name)
            if len(wait_times) == 0:
                print('   This reaction has never occured')
                continue
            if len(set(reaction_channel.reactants)) != 1:
                print('   This channel reacted %.2f times per simulation on average.' % (len(wait_times) / self.param.N_simulations))
                print('   No distribution can be plotted as the definition of time')
                print('   before reaction for non unique reactant is ambiguous.')
                continue

            colori = sns.color_palette()[(ri + 4) % 10] if color_list is None else color_list[ri]
            if bins is None:
                bins = 30
            plt.figure(figsize=figsize)
            plt.hist(wait_times, bins=bins, color=colori, density=True, edgecolor='black')
            plt.xlabel("Time before reaction [%s]" % self.param.unit)
            plt.ylabel("PDF")
            plt.title(reaction_channel.name)

            t = np.linspace(np.max(wait_times) / 100000, np.max(wait_times), 10000)
            shape_param, rate = reaction_channel.shape_param, reaction_channel.rate
            canonical = canonical_distribution(reaction_channel.distribution)
            spec = DISTRIBUTIONS[canonical]
            log_scale = canonical == 'Pareto' or (canonical in ('Weibull', 'Gamma') and shape_param < 0.99)

            if plot_theory and len(reaction_channel.reactants) == 1:
                plt.plot(t, spec['pdf'](t, rate, shape_param), theory_color, lw=lw, linestyle='--', label='Theory')
            P = None
            if plot_fitted:
                try:
                    law, P = (fit_exponential if (canonical == 'Weibull' and shape_param == 1) else spec['fit'])(wait_times)
                    plt.plot(t, law.pdf(t, *P), fitted_color, lw=lw, label='Fitted')
                except Exception as E:
                    print('   Fit failed: %s' % E)
            plt.legend()
            plt.xlim(0, (np.max(wait_times) if isinstance(bins, int) else np.max(bins)) * 1.03)
            if log_scale:
                plt.yscale('log')
            plt.show()

            print("   Obtained rate is %.3f vs %.3f" % (1 / np.mean(wait_times), reaction_channel.rate))
            print("   Corresponding to %.1f%s vs %.1f%s" % (np.mean(wait_times), self.param.unit, 1 / reaction_channel.rate, self.param.unit))
            if P is not None and reaction_channel.shape_param != 0:
                print("   Fitted params [%.2f,%.2f] vs shape_param %.2f" % (P[0], P[1], reaction_channel.shape_param))
            print("   std is %.1f%s" % (np.std(wait_times), self.param.unit))
            print("   Number of rejected reactions = %s" % reaction_channel.number_of_rejected_reactions)
            print("   Number of accepted reactions = %s" % reaction_channel.number_of_accepted_reactions)
            if reaction_channel.number_of_accepted_reactions > 0:
                print("   Rejected/accepted reactions ratio = %.2f" % (reaction_channel.number_of_rejected_reactions / reaction_channel.number_of_accepted_reactions))


# =============================================================================
# Helpers
# =============================================================================

def get_random_element(a_huge_key_list, weights=None):
    """Module-level alias of ``Gillespie_simulation.get_random_element`` kept for older scripts."""
    return Gillespie_simulation.get_random_element(None, a_huge_key_list, weights)


def interpolate_minusones(y):
    """
    Fill the unrecorded grid points (-1) of a (timepoints x n) array with the last
    recorded value (0 before the first record).
    """
    x = np.arange(y.shape[0])
    ynew = np.zeros(y.shape)
    for ni in range(y.shape[1]):
        idx = np.where(y[:, ni] != -1)[0]
        if len(idx) > 1:
            last_value = y[idx[-1], ni]
            interp = interp1d(x[idx], y[idx, ni], kind='previous', fill_value=(0, last_value), bounds_error=False)
            ynew[:, ni] = interp(x)
        elif len(idx) == 1:
            ynew[:, ni] = y[idx[-1], ni]
    return ynew
