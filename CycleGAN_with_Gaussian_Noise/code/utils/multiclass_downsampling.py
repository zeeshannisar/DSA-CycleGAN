import numpy

'''
    Taken from https://github.com/william-silversmith/countless
'''


def countless(data):
    """
    Vectorized implementation of downsampling a 2D
    image by 2 on each side using the COUNTLESS algorithm.

    data is a 2D numpy array with even dimensions.
    """
    # allows us to prevent losing 1/2 a bit of information
    # at the top end by using a bigger type. Without this 255 is handled incorrectly.
    data, upgraded = upgrade_type(data)

    data = data + 1  # don't use +=, it will affect the original data.

    sections = []

    # This loop splits the 2D array apart into four arrays that are
    # all the result of striding by 2 and offset by (0,0), (0,1), (1,0),
    # and (1,1) representing the A, B, C, and D positions from Figure 1.
    factor = (2, 2)
    for offset in numpy.ndindex(factor):
        part = data[tuple(numpy.s_[o::f] for o, f in zip(offset, factor))]
        sections.append(part)

    a, b, c, d = sections

    ab_ac = a * ((a == b) | (a == c))  # PICK(A,B) || PICK(A,C) w/ optimization
    bc = b * (b == c)  # PICK(B,C)

    a = ab_ac | bc  # (PICK(A,B) || PICK(A,C)) or PICK(B,C)
    result = a + (a == 0) * d - 1  # (matches or d) - 1

    if upgraded:
        return downgrade_type(result)

    return result


def upgrade_type(arr):
    dtype = arr.dtype

    if dtype == numpy.uint8:
        return arr.astype(numpy.uint16), True
    elif dtype == numpy.uint16:
        return arr.astype(numpy.uint32), True
    elif dtype == numpy.uint32:
        return arr.astype(numpy.uint64), True

    return arr, False


def downgrade_type(arr):
    dtype = arr.dtype

    if dtype == numpy.uint64:
        return arr.astype(numpy.uint32)
    elif dtype == numpy.uint32:
        return arr.astype(numpy.uint16)
    elif dtype == numpy.uint16:
        return arr.astype(numpy.uint8)

    return arr
