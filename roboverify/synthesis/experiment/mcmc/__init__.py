"""Instrumented MCMC program search.

Only the four functions that need to emit structured records are reimplemented
here (``MCMC``, ``score_candidate_program``, ``optimize_program`` and
``cem_optimize``). Everything else -- mutation, rollouts, BMC checks, environment
factories, video helpers -- is imported unchanged from ``synthesis.mcmc``, so there
is no second copy of that module to keep in sync.

``synthesis/mcmc/`` itself is not modified; ``test_mcmc_parity`` pins the two
implementations to the same accept/reject sequence.
"""
