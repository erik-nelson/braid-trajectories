import braid
import braid_group
import numpy as np

# Test basic braid group element composition and inverses ---------------------
e1 = braid_group.Identity()
e2 = braid_group.Generator(1)
e3 = braid_group.InverseGenerator(2)
e4 = braid_group.Word([e1, e2, e3])
e5 = e4.Compose(e4.Inverse())
e6 = e5.Inverse().Compose(braid_group.Generator(3))
assert e6.__str__() == "id * g1 * inv(g2) * g2 * inv(g1) * id * g3"

# Test strand constructors ----------------------------------------------------
s0 = braid.Strand.Straight(0)
assert np.allclose(s0.AtTime(0.0), np.array([0.0, 0.0]))
assert np.allclose(s0.AtTime(0.5), np.array([0.0, 0.0]))
assert np.allclose(s0.AtTime(1.0), np.array([0.0, 0.0]))

s1 = braid.Strand.OverLeft(1)
assert np.allclose(s1.AtTime(0.0), np.array([1.0, 0.0]))
assert np.allclose(s1.AtTime(0.5), np.array([0.5, 0.0]))
assert np.allclose(s1.AtTime(1.0), np.array([0.0, 0.0]))

s2 = braid.Strand.OverRight(2)
assert np.allclose(s2.AtTime(0.0), np.array([2.0, 0.0]))
assert np.allclose(s2.AtTime(0.5), np.array([2.5, 0.0]))
assert np.allclose(s2.AtTime(1.0), np.array([3.0, 0.0]))

s3 = braid.Strand.UnderLeft(3)
assert np.allclose(s3.AtTime(0.0), np.array([3.0,  0.0]))
assert np.allclose(s3.AtTime(0.5), np.array([2.5, -1.0]))
assert np.allclose(s3.AtTime(1.0), np.array([2.0,  0.0]))

s4 = braid.Strand.UnderRight(4)
assert np.allclose(s4.AtTime(0.0), np.array([4.0,  0.0]))
assert np.allclose(s4.AtTime(0.5), np.array([4.5, -1.0]))
assert np.allclose(s4.AtTime(1.0), np.array([5.0,  0.0]))

# Test strand composition -----------------------------------------------------
s5 = s1.Compose(s0)
assert np.allclose(s5.AtTime(0.00), np.array([1.0, 0.0]))
assert np.allclose(s5.AtTime(0.25), np.array([0.5, 0.0]))
assert np.allclose(s5.AtTime(0.50), np.array([0.0, 0.0]))
assert np.allclose(s5.AtTime(0.75), np.array([0.0, 0.0]))
assert np.allclose(s5.AtTime(1.00), np.array([0.0, 0.0]))

s6 = s3.Compose(s2)
assert np.allclose(s6.AtTime(0.00), np.array([3.0,  0.0]))
assert np.allclose(s6.AtTime(0.25), np.array([2.5, -1.0]))
assert np.allclose(s6.AtTime(0.50), np.array([2.0,  0.0]))
assert np.allclose(s6.AtTime(0.75), np.array([2.5,  0.0]))
assert np.allclose(s6.AtTime(1.00), np.array([3.0,  0.0]))

# Test braid construction from words ------------------------------------------
# Construct a 3-strand braid from the identity element.
w1 = braid_group.Word(braid_group.Identity())
b1 = braid.Braid.Create(word=w1, num_strands=3)
for s in range(3):
  assert np.allclose(b1.Strand(s).AtTime(0), np.array([s, 0.0]))
  assert np.allclose(b1.Strand(s).AtTime(1), np.array([s, 0.0]))

# Construct a 3-strand braid that first swaps the first two strands, then the
# second two strands, then the first two strands again.
#
# Looks like:
#
#  x   x   x  <-- t=1.0000
#   \ /    |
#    /     |  <-- t=0.7500
#   / \    |
#  x   x   x  <-- t=0.5000
#  |    \ /
#  |     \    <-- t=0.3750
#  |    / \
#  x   x   x  <-- t=0.2500
#   \ /    |
#    /     |  <-- t=0.1875
#   / \    |
#  x   x   x  <-- t=0.1250
#  |   |   |
#  |   |   |
#  |   |   |
#  x   x   x  <-- t=0.0000
# -----------
#  0   1   2
#
# Note that there is always a "delay", i.e. all strands begin as the identity word /
# pure braid, so there is a straight segment from t=0 to the next timestamp.
# Also note that composition "squishes" time, so [0, 1] --> [0, 0.5] --> [0, 0.25] etc.
# Because of this, the last character applied to the braid takes exponentially more time
# than the first character applied to the braid.
#
g0 = braid_group.Generator(0) 
i1 = braid_group.InverseGenerator(1)
w2 = braid_group.Word(g0.Compose(i1).Compose(g0))
b2 = braid.Braid.Create(word=w2, num_strands=3)

# Check zeroth strand.
assert np.allclose(b2.Strand(0).AtTime(0.0000), np.array([0.0,  0]))
assert np.allclose(b2.Strand(0).AtTime(0.1250), np.array([0.0,  0]))
assert np.allclose(b2.Strand(0).AtTime(0.1875), np.array([0.5,  0]))
assert np.allclose(b2.Strand(0).AtTime(0.2500), np.array([1.0,  0]))
assert np.allclose(b2.Strand(0).AtTime(0.3750), np.array([1.5, -1]))
assert np.allclose(b2.Strand(0).AtTime(0.5000), np.array([2.0,  0]))
assert np.allclose(b2.Strand(0).AtTime(0.7500), np.array([2.0,  0]))
assert np.allclose(b2.Strand(0).AtTime(1.0000), np.array([2.0,  0]))

# Check first strand.
assert np.allclose(b2.Strand(1).AtTime(0.0000), np.array([1.0,  0]))
assert np.allclose(b2.Strand(1).AtTime(0.1250), np.array([1.0,  0]))
assert np.allclose(b2.Strand(1).AtTime(0.1875), np.array([0.5, -1]))
assert np.allclose(b2.Strand(1).AtTime(0.2500), np.array([0.0,  0]))
assert np.allclose(b2.Strand(1).AtTime(0.3750), np.array([0.0,  0]))
assert np.allclose(b2.Strand(1).AtTime(0.5000), np.array([0.0,  0]))
assert np.allclose(b2.Strand(1).AtTime(0.7500), np.array([0.5,  0]))
assert np.allclose(b2.Strand(1).AtTime(1.0000), np.array([1.0,  0]))

# Check second strand.
assert np.allclose(b2.Strand(2).AtTime(0.0000), np.array([2.0,  0]))
assert np.allclose(b2.Strand(2).AtTime(0.1250), np.array([2.0,  0]))
assert np.allclose(b2.Strand(2).AtTime(0.1875), np.array([2.0,  0]))
assert np.allclose(b2.Strand(2).AtTime(0.2500), np.array([2.0,  0]))
assert np.allclose(b2.Strand(2).AtTime(0.3750), np.array([1.5,  0]))
assert np.allclose(b2.Strand(2).AtTime(0.5000), np.array([1.0,  0]))
assert np.allclose(b2.Strand(2).AtTime(0.7500), np.array([0.5, -1]))
assert np.allclose(b2.Strand(2).AtTime(1.0000), np.array([0.0,  0]))