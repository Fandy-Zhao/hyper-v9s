import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class CandidateSet:
    index: int
    expert_ids: Tuple[int, ...]
    gates: Tuple[float, ...]
    normalization: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "set_index": self.index,
            "expert_ids": list(self.expert_ids),
            "gates": list(self.gates),
            "normalization": self.normalization,
        }


def build_candidate_sets(expert_ids: Iterable[int]) -> List[CandidateSet]:
    ordered = sorted(set(int(value) for value in expert_ids))
    if not ordered:
        raise ValueError("candidate expert_ids must not be empty")
    candidates = [CandidateSet(0, (), (), "none")]
    for expert_id in ordered:
        candidates.append(CandidateSet(len(candidates), (expert_id,), (1.0,), "none"))
    pair_gate = 1.0 / math.sqrt(2.0)
    for pair in itertools.combinations(ordered, 2):
        candidates.append(
            CandidateSet(len(candidates), pair, (pair_gate, pair_gate), "l2")
        )
    return candidates


class CandidateSetIndex:
    def __init__(self, expert_ids: Sequence[int]) -> None:
        self.candidates = build_candidate_sets(expert_ids)
        self._by_ids = {candidate.expert_ids: candidate.index for candidate in self.candidates}

    def from_index(self, set_index: int) -> CandidateSet:
        try:
            return self.candidates[int(set_index)]
        except (IndexError, ValueError):
            raise KeyError("unknown candidate set index {}".format(set_index))

    def to_index(self, expert_ids: Iterable[int]) -> int:
        key = tuple(sorted(int(value) for value in expert_ids))
        try:
            return self._by_ids[key]
        except KeyError:
            raise KeyError("unknown candidate expert set {}".format(key))
