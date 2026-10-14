"""
MOSAIC core library specialised for the germinal center application (B cell competition).

This module exposes the full core library (``../MOSAIC.py``) and replaces
``Gillespie_simulation`` by a subclass that manages the per-cell properties used by the
custom reactions of ``Simulating_immune_response.py``:

- ``Bcell_affinity``: affinity of every B cell, drawn at initialisation and inherited
  (with a random change) at division,
- ``Ancestor``: clone of origin of every B cell (ten founder clones),
- ``ancestor_dict``: snapshot of the clone and affinity of every B cell at each recorded
  timepoint, used for the clone dominance plots.

All the generic machinery (custom reaction functions, state-dependent rates, Pareto
waiting times, precomputed delays) lives in the core library.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from MOSAIC import *  # noqa: F401,F403  (Reaction_channel, Reactant, helpers)
import MOSAIC as _core

N_FOUNDER_CELLS_PER_CLONE = 10


def affinity_function(a0, Nstep = 10):
    """Random affinity change of a daughter cell: a0 moves a fraction 1/Nstep towards a uniform draw."""
    return a0 + (random.random() - a0) / Nstep


class Gillespie_simulation(_core.Gillespie_simulation):

    def on_reinitialize(self):
        """Draw the initial affinities and assign the founder clones (consecutive groups of ten cells)."""
        for ri, reactant in enumerate(self.reactant_list['B']):
            self.Bcell_affinity[reactant.id] = affinity_function(0)
            self.Ancestor[reactant.id] = int(ri / N_FOUNDER_CELLS_PER_CLONE)

    def record_extra(self, ti):
        """Store the clone and affinity of every B cell (dividing or not) at the recorded timepoint."""
        Bcells = self.reactant_list['Bdiv'] + self.reactant_list['B']
        affinities = [self.Bcell_affinity[cell.id] for cell in Bcells]
        ancestors = [self.Ancestor[cell.id] for cell in Bcells]
        self.ancestor_dict[ti] = [ancestors, affinities]
