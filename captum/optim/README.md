# Captum "optim" module

This is a WIP PR to integrate existing feature visualization code from the authors of `tensorflow/lucid` into captum.
It is also an opportunity to review which parts of such interpretability tools still feel rough to implement in a system like PyTorch, and to make suggestions to the core PyTorch team for how to improve these aspects.

## Roadmap

* unify API with Captum API: a single class that's callable per "technique"(? check for details before implementing)
* Consider if we need an abstraction around "an optimization process" (in terms of stopping criteria, reporting losses, etc) or if there are sufficiently strong conventions in PyTorch land for such tasks
* integrate Eli's FFT param changes (mostly for simplification) 
* make a table of PyTorch interpretability tools for readme?
* do we need image viewing helpers and io helpers or throw those out?
* can we integrate paper references closer with the code?
