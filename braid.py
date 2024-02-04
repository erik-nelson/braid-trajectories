import braid_group
import numpy as np
from typing import List, Callable

class Strand:
  def __init__(self, f: Callable[float, np.ndarray]):
    # Sanity check that the provided function is a valid strand. Start and end
    # points' x values must be integer, y values must be zero.
    beg, end = f(0), f(1)
    assert (abs(beg[0] - round(beg[0])) < 1e-8) and abs(beg[1]) < 1e-8
    assert (abs(end[0] - round(end[0])) < 1e-8) and abs(end[1]) < 1e-8
    self.f = f

  @staticmethod
  def Over(start_idx: int, end_idx: int):
    assert abs(start_idx - end_idx) <= 1
    beg = np.array([start_idx, 0])
    end = np.array([end_idx, 0])
    return Strand(lambda t: ((1-t) * beg) + (t * end))

  @staticmethod
  def Under(start_idx: int, end_idx: int):
    assert abs(start_idx - end_idx) <= 1
    beg = np.array([start_idx, 0])
    mid = np.array([0.5 * (start_idx + end_idx), -1])
    end = np.array([end_idx, 0])
    def f(t: float) -> np.ndarray:
      if t < 0.5:
        return ((1 - 2 * t) * beg) + (t * 2 * mid)
      else:
        return ((1 - t) * 2 * mid) + ((2 * t - 1) * end)
    return Strand(f)

  @staticmethod
  def Straight(idx: int):
    return Strand.Over(idx, idx)

  @staticmethod
  def OverLeft(idx: int):
    return Strand.Over(idx, idx - 1)

  @staticmethod
  def OverRight(idx: int):
    return Strand.Over(idx, idx + 1)

  @staticmethod
  def UnderLeft(idx: int):
    return Strand.Under(idx, idx - 1)

  @staticmethod
  def UnderRight(idx: int):
    return Strand.Under(idx, idx + 1)

  def AtTime(self, time: float) -> np.ndarray:
    assert time >= 0.0 and time <= 1.0
    return self.f(time)

  def Compose(self, rhs):
    # Sanity check that the two strands are composable.
    assert np.allclose(self.f(1), rhs.f(0))

    def f(time: float):
      if time < 0.5:
        return self.f(2 * time)
      else:
        return rhs.f(2 * time - 1)

    return Strand(f)


class Braid:
  def __init__(self, strands: List[Strand]):
    assert strands
    self.strands = strands

  @staticmethod
  def Create(word: braid_group.Word, num_strands: int):
    strands = [Strand.Straight(idx) for idx in range(num_strands)]
    idx_to_strand = {idx : idx for idx in range(num_strands)}

    for character in word.characters:
      idxs_to_propagate = list(range(num_strands))
      if isinstance(character, braid_group.Identity):
        # Skip - identity element propogates each strand as it already is.
        pass

      elif isinstance(character, braid_group.Generator):
        # Perform a swap of the two relevant strands for this generator.
        idx = character.i
        strands[idx_to_strand[idx + 0]] = strands[idx_to_strand[idx + 0]].Compose(Strand.OverRight(idx))
        strands[idx_to_strand[idx + 1]] = strands[idx_to_strand[idx + 1]].Compose(Strand.UnderLeft(idx + 1))
        idxs_to_propagate = idxs_to_propagate[:idx] + idxs_to_propagate[idx+2:]

      elif isinstance(character, braid_group.InverseGenerator):
        # Perform a swap of the two relevant strands for this generator.
        idx = character.i
        strands[idx_to_strand[idx + 0]] = strands[idx_to_strand[idx + 0]].Compose(Strand.UnderRight(idx))
        strands[idx_to_strand[idx + 1]] = strands[idx_to_strand[idx + 1]].Compose(Strand.OverLeft(idx + 1))
        idxs_to_propagate = idxs_to_propagate[:idx] + idxs_to_propagate[idx+2:]

      else:
        raise TypeError("Invalid character type")

      # Propagate all other strands not touched by this character.
      for idx in idxs_to_propagate:
        strands[idx_to_strand[idx]] = strands[idx_to_strand[idx]].Compose(Strand.Straight(idx))

      # Reindex at latest timestamp.
      idx_to_strand = {int(strand.AtTime(1)[0]) : idx for idx, strand in enumerate(strands)}
 
    return Braid(strands)

  def Strand(self, idx: int) -> Strand:
    assert idx >= 0 and idx < len(self.strands)
    return self.strands[idx]

  def Compose(self, rhs):
    # Match up the end indices of our strands with the start indices of the
    # provided braid's strands.
    return Braid([s.Compose(rhs.Strand(s.end_idx)) for s in self.strands])