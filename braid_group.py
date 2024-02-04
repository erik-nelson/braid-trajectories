from abc import ABC, abstractmethod
from typing import List

# Astract base class for an element of the braid group.
class Element(ABC):
  @abstractmethod
  def Inverse():
    pass

  @abstractmethod
  def Compose(rhs):
    pass

  @abstractmethod
  def __str__(self):
    pass

# An element of the braid group representing a single character in a word.
# Characters can be one of three types of elements:
# - A generator element.
# - The inverse of a generator element.
# - The identity element.
class Character(Element):
  pass

# The identity element of the braid group.
class Identity(Character):
  def __init__(self):
    super().__init__()

  def Inverse(self) -> Element:
    return Identity()

  def Compose(self, rhs: Element) -> Element:
    return rhs

  # Debug printing.
  def __str__(self):
    return "id"

# A generator element of the braid group.
class Generator(Character):
  def __init__(self, i):
    super().__init__()
    self.i = i

  def Inverse(self) -> Element:
    return InverseGenerator(self.i)

  def Compose(self, rhs: Element) -> Element:
    return Word(self).Compose(rhs)

  # Debug printing.
  def __str__(self):
    return "g" + str(self.i)


# The inverse of a generator element of the braid group.
class InverseGenerator(Character):
  def __init__(self, i):
    super().__init__()
    self.i = i

  def Inverse(self) -> Element:
    return Generator(self.i)

  def Compose(self, rhs: Element) -> Element:
    return Word(self).Compose(rhs)

  # Debug printing.
  def __str__(self):
    return "inv(g" + str(self.i) + ")"


# A word in the braid group. A word is a composed sequence of characters.
class Word(Element):
  # Initialize from a single character or sequence of characters. Multiplication is applied 
  # in right-to-left order, e.g. if the sequence [a, b, c, d] is passed, this word corresponds
  # to the group element a * b * c * d, where group multiplication is read right to left.
  def __init__(self, characters):
    if isinstance(characters, list):
      assert characters
      self.characters = characters
    else:
      if isinstance(characters, Word):
        self.characters = characters.characters
      else:
        self.characters = [characters]

  # A generic group inverse reverses character order and inverts each character.
  def Inverse(self) -> Element:
    return Word([c.Inverse() for c in reversed(self.characters)])

  # Compose this word with the provided `rhs` group element.
  def Compose(self, rhs: Element) -> Element:
    lhs_characters = self.characters
    rhs_characters = rhs.characters if isinstance(rhs, Word) else Word(rhs).characters
    composed_characters = lhs_characters + rhs_characters
    return Word(composed_characters)

  # Debug printing.
  def __str__(self):
    return ' * '.join([c.__str__() for c in self.characters])