import sample
import braid_group

# Test unsortedness helper function -------------------------------------------
assert sample.unsortedness((0,), (0,)) == 0
assert sample.unsortedness((0, 1), (0, 1)) == 0
assert sample.unsortedness((0, 1), (1, 0)) == 2
assert sample.unsortedness((0, 1, 2), (0, 1, 2)) == 0
assert sample.unsortedness((0, 1, 2), (0, 2, 1)) == 2
assert sample.unsortedness((0, 1, 2), (1, 0, 2)) == 2
assert sample.unsortedness((0, 1, 2), (1, 2, 0)) == 4
assert sample.unsortedness((0, 1, 2), (2, 0, 1)) == 4
assert sample.unsortedness((0, 1, 2), (2, 1, 0)) == 4

# Test permutation for word helper function -----------------------------------
gens = [braid_group.Generator(i) for i in range(3)]
invs = [braid_group.InverseGenerator(i) for i in range(3)]

# Words on one strand.
assert sample.permutation_for_word(braid_group.Identity(), num_strands=1) == (0,)
# Words on two strands.
assert sample.permutation_for_word(braid_group.Identity(), num_strands=2) == (0, 1)
assert sample.permutation_for_word(gens[0], num_strands=2) == (1, 0)
assert sample.permutation_for_word(gens[0].Compose(invs[0]), num_strands=2) == (0, 1)
# Words on three strands.
assert sample.permutation_for_word(braid_group.Identity(), num_strands=3) == (0, 1, 2)
assert sample.permutation_for_word(gens[0], num_strands=3) == (1, 0, 2)
assert sample.permutation_for_word(invs[0], num_strands=3) == (1, 0, 2)
assert sample.permutation_for_word(gens[1], num_strands=3) == (0, 2, 1)
assert sample.permutation_for_word(invs[1], num_strands=3) == (0, 2, 1)
assert sample.permutation_for_word(gens[0].Compose(gens[1]), num_strands=3) == (1, 2, 0)
assert sample.permutation_for_word(invs[0].Compose(invs[1]), num_strands=3) == (1, 2, 0)
assert sample.permutation_for_word(gens[0].Compose(invs[1]), num_strands=3) == (1, 2, 0)
assert sample.permutation_for_word(invs[0].Compose(gens[1]), num_strands=3) == (1, 2, 0)


# Test braid sampling ---------------------------------------------------------

# Words on one strand ---------------------------
# There is only one valid word that achives the permutation (0,) - the identity braid.
words = sample.sample_braids(permutation=(0,))
assert len(words) == 1
assert len(words[0].characters) == 1
assert isinstance(words[0].characters[0], braid_group.Identity)

# Words on two strands --------------------------
# Find words for the identity permutation.
words = sample.sample_braids(permutation=(0, 1))
assert len(words) == 1
assert len(words[0].characters) == 1
assert isinstance(words[0].characters[0], braid_group.Identity)

# Find words for the (1, 0) permutation. We expect to find two - applying the generator,
# and applying its inverse.
words = sample.sample_braids(permutation=(1, 0))
assert len(words) == 2

# Found word that applies one generator.
assert len(words[0].characters) == 1
assert isinstance(words[0].characters[0], braid_group.Generator)
assert words[0].characters[0].i == 0

# Found word that applies one inverse generator.
assert len(words[1].characters) == 1
assert isinstance(words[1].characters[0], braid_group.InverseGenerator)
assert words[1].characters[0].i == 0
