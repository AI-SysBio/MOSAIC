"""
Activity-Driven Model (ADM) baseline for the temporal network benchmark.

The ADM simulations compared against MOSAIC-TN in the paper (``Sim_ADM_V9_*.csv``
in ``Data.zip`` and ``Data/sim_time_ADM.npy``) were generated with Didier Le
Bail's temporal network code, which is maintained in a separate repository:

    https://github.com/DidierLeBail/Temporal-networks-PhD-code

The runtime benchmark was obtained with that code as follows::

    import Librairies.Temp_net as tp
    adm_tij = tp.ADM_name(9, "conf17", N=N).load_TN()   # ADM version 9 tuned on conf17

This module is kept so that ``Benchmarking_time.py`` imports cleanly. To regenerate
the ADM results, clone the repository above and implement ``run_ADM_simulation``
on top of it.
"""


def run_ADM_simulation(N, Tend):
    raise NotImplementedError(
        "The ADM baseline requires the external code at "
        "https://github.com/DidierLeBail/Temporal-networks-PhD-code; "
        "precomputed results are provided in Data/sim_time_ADM.npy."
    )
