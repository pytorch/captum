#!/usr/bin/env python3

# pyre-strict
import itertools
from typing import List
import torch
from collections import deque
import random
import math
from dataclasses import dataclass

from captum.attr._utils.attribution import PerturbationAttribution
from captum._utils.typing import BaselineType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.common import _format_input_baseline, _validate_input, _format_output
from captum.log.dummy_log import log_usage


class Partition:
    def __init__(self, borders: List[slice] = None, elements=None, size=None):
        self.borders = borders
        self.elements = elements
        self.size = size
        
        self._mask = None

    def generate_mask(self, shape):
        # generates a mask for a partition (False indicates membership)
        if self._mask is not None: return self._mask
        self._mask = torch.zeros(shape, dtype=torch.bool)

        # non-contiguous case
        if self.elements is not None:
            self._mask[tuple(self.elements.T)] = True
        
        # contiguous case
        elif self.borders is not None:
            slices = list(slice(lo, hi) for (lo, hi) in self.borders)
            self._mask[slices] = True
        
        return self._mask
    
    def __len__(self):
        return self.size

    def __str__(self):
        # unsafe
        return self.generate_mask(None).to(torch.int).__str__()
        

@dataclass(eq=False)
class Mutant:
    partitions: List[Partition]
    data: torch.Tensor

    # eagerly create the underlying mutant data
    def __init__(self, partitions: List[Partition], data: torch.Tensor, neutral, shape):
        
        # A bitmap in the shape of the input indicating membership to a partition in this mutant
        mask = torch.zeros(shape, dtype=torch.bool)
        for part in partitions: mask |= part.generate_mask(mask.shape)

        self.partitions = partitions
        self.data = torch.where(mask, data, neutral)

    def __len__(self):
        return len(self.partitions)


def _powerset(s):
    return (list(combo) for r in range(len(s)+1) 
            for combo in itertools.combinations(s, r))


def _apply_responsibility(fi, part, responsibility):
    distributed = responsibility / len(part)
    mask = part.generate_mask(fi.shape)

    return torch.where(mask, distributed, fi)


def _part_to_set(partition):
    return frozenset(frozenset(p) if isinstance(p, list) else p for p in partition)


def _calculate_responsibility(subject_partition: Partition,
                              mutants: List[Mutant],
                              recovery_mutants: List[Mutant]) -> float:


    recovery_set = {_part_to_set(m.partitions) for m in recovery_mutants}

    valid_witnesses = []
    for m in mutants:
        if subject_partition in m.partitions:
            continue
        W = m.partitions
        W_set = _part_to_set(W)
        W_plus_P_set = _part_to_set([subject_partition] + W)

        # W alone does NOT recover, but W ∪ {P} DOES recover.
        if (W_set not in recovery_set) and (W_plus_P_set in recovery_set):
            valid_witnesses.append(W)

    if not valid_witnesses:
        return 0.0

    k = min(len(w) for w in valid_witnesses)
    # Responsibility per your definition: 1 / (1 + k)
    return 1.0 / (1.0 + float(k))


def _generate_indices(ts):
    return torch.tensor(tuple(itertools.product(*(range(s) for s in ts.shape))), dtype=torch.long)


class ReX(PerturbationAttribution):
    """
    A perturbation-based approach to computing attribution, based on the
    Halpern-Pearl definition of Actual Causality[1]. 
    
    ReX works by partitioning the input space, and masking each partition with the baseline value. It is fully 
    model agnostic, and relies only on a 'forward_func' returning a scalar.

    Intuitively, if masking a partition changes the prediction of the model, then that partition has 
    some responsibility (attribution > 0). Such partially masked partitions are called
    mutants. The responsibility of a partition is defined as 1/(1+k) where
    k is a minimum number of occluded partitions in a mutant which make forward_func's 
    output dependednt on the subject partition.
    
    Partitions with nonzero responsibility are recusrively re-partitioned and masked in a search.
    The algorithm runs multiple such searches, where each subsequent search uses the previously 
    computed attribution map as a heuristic for partitioning.


    [1] - halpern 06
    [2] - rex paper
    """
    def __init__(self, forward_func):
        r"""
        Args:
            forward_func (Callable): The function to be explained. Must return
            a scalar for which the equality operator is defined.
        """
        PerturbationAttribution.__init__(self, forward_func)

    @log_usage(part_of_slo=True)
    def attribute(self,
                  inputs: TensorOrTupleOfTensorsGeneric,
                  baselines: BaselineType = 0,
                  search_depth: int = 10,
                  n_partitions: int = 4,
                  n_searches: int = 5,
                  contiguous_partitioning: bool = False,
                  merge: bool = True) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:
            inputs:
                An input or tuple of inputs whose corresponding output is to be explained. Each input
                must be of the shape expected by the forward_func. Where multiple examples are 
                provided, they must be listed in a tuple.
            
            baselines: 
                A neutral values to be used as occlusion values. Where a scalar is provided, it is used
                as the masking value at each index. Where a tensor is provided, values are masked at
                corresponding indices. Where a tuple of tensors is provided, it must be of the same length
                as inputs; then baseline and input tensors are matched element-wise and treated as before.

            search_depth (optional):
                The maximum depth to which ReX will search. Where one is not provided, the default is 4
            
            n_partitions (optional):
                The number of partitions to be made out of the input at each search step.
                This must be at most hte size of each input, and at least 1.
        """
        inputs, baselines = _format_input_baseline(inputs, baselines)
        _validate_input(inputs, baselines)

        self._n_partitions = n_partitions
        self._max_depth = search_depth
        self._n_searches = n_searches
        self._is_contiguous = contiguous_partitioning
        self._merge = merge

        is_input_tuple = isinstance(inputs, tuple)
        is_baseline_tuple = isinstance(baselines, tuple)

        attributions = []

        # match inputs and baselines, explain
        if is_input_tuple and is_baseline_tuple:
            for input, baseline in zip(inputs, baselines):
                attributions.append(self._explain(input, baseline))
        elif is_input_tuple and not is_baseline_tuple:
            for input in inputs:
                attributions.append(self._explain(input, baselines))
        else:
            attributions.append(self._explain(inputs, baselines))

        return _format_output(is_input_tuple, tuple(attributions))


    def _explain(self, input, baseline):
        self._shape = input.shape
        self._size = input.numel()

        initial_prediction = self.forward_func(input)

        prev_attribution = torch.full_like(input, 0.0, dtype=torch.float32)
        attribution = torch.full_like(input, 1.0/input.numel(), dtype=torch.float32)

        initial_partition = Partition(
            borders     = list((0, top) for top in self._shape),
            elements    = _generate_indices(input),
            size        = self._size
        )

        for i in range(1, self._n_searches + 1):
            Q = deque()
            Q.append((initial_partition, 0))

            while Q:
                prev_part, depth = Q.popleft()
                partitions = self._contiguous_partition(prev_part, depth) \
                    if self._is_contiguous else self._partition(prev_part, attribution)

                
                mutants = [Mutant(ps, input, baseline, self._shape) for ps in _powerset(partitions)]
                consistent_mutants = [mut for mut in mutants if self.forward_func(mut.data) == initial_prediction]

                prev_part.generate_mask(self._shape)
                for part in partitions:
                    resp        = _calculate_responsibility(part, mutants, consistent_mutants)
                    attribution = _apply_responsibility(attribution, part, resp)

                    if resp == 1 and len(part) > 1 and self._max_depth > depth:
                        Q.append((part, depth + 1))
                    else:
                        attribution = _apply_responsibility(attribution, part, resp)
                
            if self._merge: 
                prev_attribution += (1/i) * (attribution - prev_attribution)
                attribution = prev_attribution

        return attribution.clone().detach()


    def _partition(self, part: Partition, responsibility: torch.Tensor) -> List[Partition]:
        # shuffle candidate indices (randomize tiebreakers)
        perm = torch.randperm(len(part.elements))
        population = part.elements[perm]
        weights = responsibility[tuple(population.T)]
        
        if torch.sum(weights, dim=None) == 0: weights = torch.ones_like(weights) / len(weights)
        target_weight = torch.sum(weights) / self._n_partitions

        # sort for greedy selection
        idx = torch.argsort(weights, descending=False)
        weight_sorted, pop_sorted = weights[idx], population[idx]

        # cumulative sum of weights / weight per bucket rounded down gives us bucket ids
        eps = torch.finfo(weight_sorted.dtype).eps 
        c = weight_sorted.cumsum(0) + eps
        bin_id = torch.div(c, target_weight, rounding_mode='floor').clamp_min(0).long()
        
        # count elements in each bucket, and split input accordingly
        _, counts = torch.unique_consecutive(bin_id, return_counts=True)
        groups = torch.split(pop_sorted, counts.tolist())
        
        partitions = [Partition(elements=g, size=len(g)) for g in groups]
        return partitions


    def _contiguous_partition(self, part, depth):
        ndim = len(self._shape)
        split_dim = -1

        # find a dimension we can split 
        dmin, dmax = max(self._shape), 0
        for i in range(ndim):
            candidate_dim = (i + depth) % ndim
            dmin, dmax = tuple(part.borders[candidate_dim])

            if dmax - dmin > 1:
                split_dim = candidate_dim
                break

        if split_dim == -1: return [part]
        
        n_splits = min((dmax - dmin), self._n_partitions) - 1

        # drop splits randomly
        split_points = random.sample(range(dmin + 1, dmax), n_splits)
        split_borders = sorted(set([dmin, *split_points, dmax]))

        bins = []
        for i in range(len(split_borders) - 1):
            new_borders = list(part.borders)
            new_borders[split_dim] = (split_borders[i], split_borders[i+1])

            bins.append(Partition(
                borders = tuple(new_borders),
                size    = math.prod(hi - lo for (lo, hi) in new_borders)
            ))

        return bins


    def multiplies_by_inputs(self) -> bool:
        return False    

    def has_convergence_delta(self) -> bool:
        return True 