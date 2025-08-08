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
        # function to generate a mask for a partition, polymorphic over
        # splitting strategy

        if self._mask is None and self.elements is not None:
            self._mask = torch.ones(shape, dtype=torch.bool)
            self._mask[tuple(self.elements.T)] = False
        
        elif self._mask is None and self.borders is not None:
            self._mask = torch.ones(shape, dtype=torch.bool)

            slices = list(slice(lo, hi) for (lo, hi) in self.borders)
            self._mask[slices] = False
        
        return self._mask
    
    def __len__(self):
        return self.size
        
@dataclass(eq=False)
class Mutant:
    partitions: List[List[int]]
    data: List[int]

    # initialize a Mutant from some partitions
    # eagerly create the underlying mutant data from partition masks
    def __init__(self, data: torch.Tensor, partitions: List[Partition], neutral):
        self.partitions = partitions

        mask = torch.ones_like(data, dtype=torch.bool)
        for part in partitions: mask &= part.generate_mask(mask.shape)

        self.data = torch.where(mask, data, neutral)

    def __len__(self):
        return len(self.partitions)


def _powerset(s):
    return (list(combo) for r in range(len(s)+1) 
            for combo in itertools.combinations(s, r))


def _apply_responsibility(fi, part, responsibility):
    distributed = responsibility / len(part)
    mask = part.generate_mask(fi.shape)

    return torch.where(mask, fi, (fi * distributed))


def _part_to_set(partition):
    return frozenset(frozenset(p) if isinstance(p, list) else p for p in partition)


def _responsibility(subject_partition: List, consistent_partitions: List[List[int]]) -> float:
    witnesses = [mut.partitions for mut in consistent_partitions if subject_partition not in mut.partitions]
    consistent_set = set(_part_to_set(part.partitions) for part in consistent_partitions)
    
    # a witness is valid if perturbing it results in a counterfactual 
    # dependence on the subject partition
    valid_witnesses = []
    for witness in witnesses:
        counterfactual = _part_to_set([subject_partition] + witness)
        if not counterfactual in consistent_set:
            valid_witnesses.append(witness) 

    if len(valid_witnesses) == 0:
        return 0.0

    min_mutant = min(valid_witnesses, key=len)
    minpart = len(min_mutant)

    return 1.0 / (1.0 + float(minpart))


def _generate_indices(ts):
    return torch.tensor(tuple(itertools.product(*(range(s) for s in ts.shape))), dtype=torch.long)

class ReX(PerturbationAttribution):
    """
    A perturbation-based approach to computing attribution, based on the
    Halpern-Pearl definition of actual causality[1]. 
    
    The approach works by
    partitioning the input space, and masking each partition. Intuitively, if masking a 
    partition changes the prediction of the model, then that partition has 
    some responsibility (attribution > 0). Such partially masked partitions are called
    mutants. The responsibility of a subject partition is defined as 1/(1+k) where
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
                  *,
                  search_depth: int = 10,
                  n_partitions: int = 8,
                  n_searches: int = 5,
                  contiguous_partitions: bool = False) -> TensorOrTupleOfTensorsGeneric:
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
        self._search_depth = search_depth
        self._n_searches = n_searches

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
        self._original_shape = input.shape
        self._size = input.numel()

        initial_prediction = self.forward_func(input)
        feature_attribution = torch.full_like(input, 1.0/input.numel(), dtype=torch.float32)

        initial_partition = Partition(
            borders     = list((0, top) for top in self._original_shape),
            elements    = _generate_indices(input),
            size        = self._size
        )
        for _ in range(self._n_searches):
            # by definition, root partition contains all indices
            part_q = deque()
            part_q.append((
                initial_partition,
                0
            ))

            while part_q:
                prev_part, depth = part_q.popleft()
                partitions = self._fast_partition(feature_attribution, prev_part)

                consistent_set = set()
                for parts_combo in _powerset(partitions):
                    mut = Mutant(input, parts_combo, baseline)
                    if self.forward_func(mut.data) == initial_prediction:
                        consistent_set.add(mut)

                for part in partitions:
                    resp = _responsibility(part, consistent_set)
                    feature_attribution = _apply_responsibility(feature_attribution, part, resp)

                    if resp > 0 and \
                            len(part) > 1 and \
                                depth < self._search_depth:
                        part_q.append((part, depth + 1))

                asum = feature_attribution.abs().sum()
                feature_attribution /= asum if asum != 0 else 1

        return feature_attribution.clone().detach()


    def _fast_partition(self, responsibility: torch.Tensor, part: Partition) -> List[Partition]:
        perm = torch.randperm(len(part.elements))
        
        population = part.elements[perm]
        weights = responsibility[tuple(population.T)]
        
        if torch.sum(weights, dim=None) == 0: weights = torch.ones_like(weights) / len(weights)
        print(torch.sum(weights, dim=None))

        remaining_weight = torch.sum(weights, dim=None)
        target_weight = remaining_weight / self._n_partitions
        

        idx = torch.argsort(weights, descending=True)
        print("inb4", weights, population)
        print(part.elements, part.size)
        weight_sorted, pop_sorted = weights[idx], population[idx]

        eps = torch.finfo(weight_sorted.dtype).eps 
        c = weight_sorted.cumsum(0) - eps

        bin_id = torch.div(c, target_weight, rounding_mode='floor').clamp_min(0).long()

        _, counts = torch.unique_consecutive(bin_id, return_counts=True)
        groups = torch.split(pop_sorted, counts.tolist())
        
        print("--------------")
        print(c)
        print(weight_sorted)
        print(bin_id)
        print(counts)
        print(groups)
        print("--------------")

        partitions = [Partition(elements=g, size=len(g)) for g in groups]
        return partitions


    def _contiguous_partition(self, resposibility, part, depth):
        ndim = len(self._original_shape)
        split_dim = -1

        # find max and min values for dimension we are splitting
        dmin, dmax = max(self._original_shape), 0
        for i in range(ndim):
            candidate_dim = (i + depth) % ndim
            dmin, dmax = tuple(part.borders[candidate_dim])

            if dmax - dmin > 1:
                split_dim = candidate_dim
                break
        
        n_splits = min((dmax - dmin), self._n_partitions)

        split_points = random.sample(range(dmin, dmax), n_splits - 1)
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


    def _partition(self, responsibility: List[float], choices: List[int]) -> List[List[int]]:
        population = choices.copy()
        random.shuffle(population)
        
        weights = [responsibility[i] for i in population]
        if torch.sum(weights) == 0: weights = [1 for _ in choices]

        target_weight = sum(weights) / self._n_partitions
        partitions = []

        curr_weight = 0.0
        curr_partition = []
        
        while population:
            choice = random.choices(population, weights, k=1)[0]
            idx = population.index(choice)

            population.pop(idx)
            
            weights = [responsibility[i] for i in population]
            if sum(weights) == 0: weights = [1 for _ in population]

            curr_partition.append(choice)
            curr_weight += responsibility[choice]

            if curr_weight > target_weight:
                partitions.append(curr_partition)
                curr_partition, curr_weight = [], 0.0

        if curr_partition:
            partitions.append(Partition(
                    elements = set(curr_partition),
                    size     = len(curr_partition)
                ))
        
        return partitions



    def multiplies_by_inputs(self) -> bool:
        return False    

    def has_convergence_delta(self) -> bool:
        return True 