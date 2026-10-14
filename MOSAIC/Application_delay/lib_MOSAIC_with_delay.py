"""
MOSAIC core library specialised for the delayed transcription application (Hes1 model).

This module exposes the full core library (``../MOSAIC.py``) and replaces
``Gillespie_simulation`` by a subclass whose rates are updated at every step from the
current protein level, as in the paper:

    initiation rate   0 -> N :  beta / (1 + (P / P0)**h)       (negative feedback)
    elongation rate   N -> M :  1 / tau                        (mean elongation time)

The simulation object must carry the attributes ``beta``, ``P0``, ``h`` and ``tau`` and
the channel list must start with the initiation channel followed by the elongation
channel (see ``Model_benchmark_accuracy.py`` and ``Model_benchmark_cost.py``).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from MOSAIC import *  # noqa: F401,F403  (Reaction_channel, Reactant, helpers)
import MOSAIC as _core


class Gillespie_simulation(_core.Gillespie_simulation):

    def RNA_rate_function(self, P):
        """Transcription initiation rate repressed by the protein level (Hill function)."""
        return self.beta / (1 + (P / self.P0) ** self.h)

    def RNA_delay_rate_function(self, N, M):
        """Mean elongation rate; constant in the model of the paper."""
        return 1 / self.tau

    def update_rates(self):
        super().update_rates()
        P = self.reactant_population['P']
        M = self.reactant_population['M']
        N = self.reactant_population['N']
        self.reaction_channel_list[0].rate = self.RNA_rate_function(P)
        self.reaction_channel_list[1].rate = self.RNA_delay_rate_function(N, M)
