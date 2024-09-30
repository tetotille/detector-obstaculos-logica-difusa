import cupy as cp

def _nearest(x, y0):
    d = cp.abs(x - y0)
    idx0 = cp.nonzero(d == d.min())[0][0]
    return idx0, x[idx0]

def dsigmf(x, b1, c1, b2, c2):
    return sigmf(x, b1, c=c1) - sigmf(x, b2, c=c2)

def gaussmf(x, mean, sigma):
    return cp.exp(-((x - mean)**2.) / (2 * sigma**2.))

def gauss2mf(x, mean1, sigma1, mean2, sigma2):
    assert mean1 <= mean2, 'mean1 <= mean2 is required.  See docstring.'
    y = cp.ones(len(x))
    idx1 = x <= mean1
    idx2 = x > mean2
    y[idx1] = gaussmf(x[idx1], mean1, sigma1)
    y[idx2] = gaussmf(x[idx2], mean2, sigma2)
    return y

def gbellmf(x, a, b, c):
    return 1. / (1. + cp.abs((x - c) / a) ** (2 * b))

def piecemf(x, abc):
    a, b, c = abc
    if c != x.max():
        c = x.max()

    assert a <= b and b <= c, '`abc` requires a <= b <= c.'

    n = len(x)
    y = cp.zeros(n)

    idx0 = _nearest(x, 0)[0]
    idxa = _nearest(x, a)[0]
    idxb = _nearest(x, b)[0]

    n = cp.r_[0:n - idx0]
    y[idx0 + n] = n / float(c)
    y[idx0:idxa] = 0
    m = cp.r_[0:idxb - idxa]
    y[idxa:idxb] = b * m / (float(c) * (b - a))

    return y / y.max()

def pimf(x, a, b, c, d):
    y = cp.ones(len(x))
    assert a <= b and b <= c and c <= d, 'a <= b <= c <= d is required.'

    idx = x <= a
    y[idx] = 0

    idx = cp.logical_and(a <= x, x <= (a + b) / 2.)
    y[idx] = 2. * ((x[idx] - a) / (b - a)) ** 2.

    idx = cp.logical_and((a + b) / 2. < x, x <= b)
    y[idx] = 1 - 2. * ((x[idx] - b) / (b - a)) ** 2.

    idx = cp.logical_and(c <= x, x < (c + d) / 2.)
    y[idx] = 1 - 2. * ((x[idx] - c) / (d - c)) ** 2.

    idx = cp.logical_and((c + d) / 2. <= x, x <= d)
    y[idx] = 2. * ((x[idx] - d) / (d - c)) ** 2.

    idx = x >= d
    y[idx] = 0

    return y

def psigmf(x, b1, c1, b2, c2):
    return sigmf(x, b1, c1) * sigmf(x, b2, c2)

def sigmoid(wx, b):
    return 1. / (1. + cp.exp(-(wx + cp.dot(cp.atleast_2d(b).T,
                                           cp.ones((1, wx.shape[1]))))))

def sigmf(x, b, c):
    return 1. / (1. + cp.exp(- c * (x - b)))

def smf(x, a, b):
    assert a <= b, 'a <= b is required.'
    y = cp.ones(len(x))
    idx = x <= a
    y[idx] = 0

    idx = cp.logical_and(a <= x, x <= (a + b) / 2.)
    y[idx] = 2. * ((x[idx] - a) / (b - a)) ** 2.

    idx = cp.logical_and((a + b) / 2. <= x, x <= b)
    y[idx] = 1 - 2. * ((x[idx] - b) / (b - a)) ** 2.

    return y

def trapmf(x, abcd):
    assert len(abcd) == 4, 'abcd parameter must have exactly four elements.'
    a, b, c, d = cp.r_[abcd]
    assert a <= b and b <= c and c <= d, 'abcd requires the four elements \
                                          a <= b <= c <= d.'
    y = cp.ones(len(x))

    idx = cp.nonzero(x <= b)[0]
    y[idx] = trimf(x[idx], cp.r_[a, b, b])

    idx = cp.nonzero(x >= c)[0]
    y[idx] = trimf(x[idx], cp.r_[c, c, d])

    idx = cp.nonzero(x < a)[0]
    y[idx] = cp.zeros(len(idx))

    idx = cp.nonzero(x > d)[0]
    y[idx] = cp.zeros(len(idx))

    return y

def trimf(x, abc):
    assert len(abc) == 3, 'abc parameter must have exactly three elements.'
    a, b, c = cp.r_[abc]     # Zero-indexing in Python
    assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'

    y = cp.zeros(len(x))

    # Left side
    if a != b:
        idx = cp.nonzero(cp.logical_and(a < x, x < b))[0]
        y[idx] = (x[idx] - a) / float(b - a)

    # Right side
    if b != c:
        idx = cp.nonzero(cp.logical_and(b < x, x < c))[0]
        y[idx] = (c - x[idx]) / float(c - b)

    idx = cp.nonzero(x == b)
    y[idx] = 1
    return y

def zmf(x, a, b):
    assert a <= b, 'a <= b is required.'

    y = cp.ones(len(x))

    idx = cp.logical_and(a <= x, x < (a + b) / 2.)
    y[idx] = 1 - 2. * ((x[idx] - a) / (b - a)) ** 2.

    idx = cp.logical_and((a + b) / 2. <= x, x <= b)
    y[idx] = 2. * ((x[idx] - b) / (b - a)) ** 2.

    idx = x >= b
    y[idx] = 0

    return y
