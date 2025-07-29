#!/usr/bin/env python3

# pyre-strict
import itertools
from typing import List
import torch
from collections import deque
import random
from dataclasses import dataclass

from captum.attr._utils.attribution import PerturbationAttribution
from captum._utils.typing import BaselineType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.common import _format_input_baseline, _validate_input, _format_output
from captum.log.dummy_log import log_usage

@dataclass
class Mutant:
    partitions: List[List[int]]
    data: List[int]

def _occlude_data(data: torch.Tensor, partitions: List[List[int]], neutral) -> torch.Tensor:
    mask = torch.ones_like(data, dtype=torch.bool)
    for part in partitions:
        mask[part] = False

    return torch.where(mask, data, neutral)

def _powerset(iterable):
    s = list(iterable)
    return (list(combo) for r in range(len(s)+1) 
            for combo in itertools.combinations(s, r))


def _partitions_combinations(partitions):
    return  list(filter(
        lambda x: len(x) <= len(partitions),
        _powerset(partitions)))


def _apply_responsibility(feature_importance, part, responsibility):
    distributed_resp = responsibility / len(part)
    for idx in part:
        feature_importance[idx] = distributed_resp

    return feature_importance


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


    def _flatten(self, data):
        self._original_shape = data.size()
        return data.reshape(-1)

    def _unflatten(self, data):
        return data.reshape(self._original_shape)


    def _fast_partition(self, responsibility: List[int], choices: List[int]) -> List[List[int]]:
        population = choices.copy()
        random.shuffle(population)

        weights = [responsibility[i] for i in population]
        if sum(weights) == 0: weights = [1/len(population) for _ in population]
        
        remaining_weight = sum(weights)
        target_weight = remaining_weight / self._n_partitions
        
        zip_sorted = zip(weights, population)
        weight_sorted, pop_sorted = zip(*sorted(zip_sorted, key=lambda x: x[0], reverse=True))
        
        partitions, lp, rp, cumsum = [], 0, 0, 0
        while rp < len(weight_sorted):
            cumsum += weight_sorted[rp]
            remaining_weight -= weight_sorted[rp]
            if cumsum >= target_weight or rp == len(weight_sorted) - 1:
                partitions.append(list(pop_sorted[lp:rp + 1]))
                lp = rp + 1
                cumsum = 0

            rp += 1

        return partitions


    def _partition(self, responsibility: List[float], choices: List[int]) -> List[List[int]]:
        population = choices.copy()
        random.shuffle(population)
        
        weights = [responsibility[i] for i in population]
        if sum(weights) == 0: weights = [1 for _ in choices]

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
            partitions.append(curr_partition)
        
        return partitions



    def _explain(self, input, baseline):
        initial_prediction = self.forward_func(input)
        flattened_input = self._flatten(input)

        n_features = flattened_input.numel()
        feature_attribution = [0.0 for _ in range(n_features)]

        for _ in range(self._n_searches):
            # by definition, root partition contains all indices
            part_q = deque()
            part_q.append((list(range(n_features)), 1.0, 0))

            while part_q:
                indices, parent_resp, depth = part_q.popleft()
                partitions = self._fast_partition(feature_attribution, indices)

                mutants = []
                for idx_combination in _partitions_combinations(partitions):
                    occluded_data = _occlude_data(flattened_input, idx_combination, baseline)
                    mut = Mutant(partitions=idx_combination, data=occluded_data)
                    mutants.append(mut)

                cst_set = list(filter(
                    lambda mut: self.forward_func(self._unflatten(mut.data)) == initial_prediction,
                    mutants
                ))

                for part in partitions:
                    resp = _responsibility(part, cst_set)
                    feature_attribution = _apply_responsibility(feature_attribution, part, resp * parent_resp)
                    if resp > 0 and \
                            len(part) > 1 and \
                                depth < self._search_depth:
                        part_q.append((part, resp, depth + 1))

        return self._unflatten(torch.tensor(feature_attribution))


    def multiplies_by_inputs(self) -> bool:
        return False    

    def has_convergence_delta(self) -> bool:
        return True 