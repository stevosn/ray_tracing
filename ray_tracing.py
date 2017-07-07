from numpy import array, matrix, inf, nan, ones, zeros, arctan, isinf

import matplotlib

matplotlib.use('nbAgg')

import matplotlib.pyplot as plt


class OPE(object):
    def __init__(self, a=inf, d=None, f=None, name=''):
        """
        Optical path element used for ray tracing.
    
        Arguments
        ---------
        a : inf or float
            Aperture of the element.
        d : float or None
            Distance travelled through optical element.
        f : flaot or None
            Focal length of optical element.
        name : str
            Name of the element.
        """
        self.aperture = a
        self.name = name

        if f:
            self.c = -1 / f
        else:
            self.c = 0

        if d:
            self.b = d
        else:
            self.b = 0

    def is_lens(self):
        if self.get_matrix()[1, 0] != 0:
            return True
        else:
            return False

    def get_matrix(self):
        return matrix([[1, self.b], [self.c, 1]])

    def passes_aperture(self, r, verbose=False):
        """
        Does an input ray r pass the optical element.
        """
        h = r.flatten()[0]
        if abs(h) > self.aperture:
            if verbose:
                print('{0:s}: stopped at aperture with height {1:1.3f}'.format(self.name, h))
            return False
        else:
            if verbose:
                print('{0:s}: passed aperture with height {1:1.3f}'.format(self.name, h))
            return True

    def pass_aperture(self, r, verbose=False):
        """ Return nan vector if ray cannot pass aperture."""
        if self.passes_aperture(r, verbose=verbose):
            return r
        else:
            return matrix(ones(r.shape) * nan)

    def transmit(self, r, verbose=False):
        """ Calculate output ray. """
        M = self.get_matrix()
        r_ = M * matrix(self.pass_aperture(r, verbose=verbose)).T
        return array(r_).flatten()

    def get_travel_length(self):
        """ Length the optical element takes. """
        return self.b


def trace_ray(r, sequence):
    """
    Take r as input vector and trace the ray through the sequence of
    optical path elements OLEs.
    """
    ray0 = zeros((2, len(sequence) + 1))
    ray0[:, 0] = r

    #Do the calculations
    distances = [0.0]
    for idx, el in enumerate(sequence):
        ray0[:, idx + 1] = el.transmit(ray0[:, idx])
        distances.append(distances[idx] + el.get_travel_length())

    dist = array(distances)

    return (dist, ray0)


def get_first_aperture(sequence):
    """
    get aperture of first lens in path and calculate distance to it.

    Arguments
    ---------
    sequence : list of OPE
        List of optical path elements

    Returns
    -------
    tuple(idx, distance, aperture)
    """
    d = 0.0
    a = inf
    for idx, m in enumerate(sequence):
        if isinf(m.aperture):
            d += m.get_travel_length()
        else:
            a = m.aperture
            break
    return (idx, d, a)


def get_lens_pos(sequence):
    """
    Calculate positions of lenses.
    """
    d = 0.0
    d_ = []
    for m in sequence:
        if m.is_lens():
            d_.append(d)
        else:
            d += m.get_travel_length()

    return d_


def get_angle_lim(h, d, a):
    """
    Calculate the upper and lower angles of a source that is at
    distance d to an aperture a.
    """
    #return (arctan((a - h) / d), arctan(-(a + h) / d))
    return ((a - h) / d, -(a + h) / d)


def trace_parser(s):
    '''
    Convert a string into a seqeunce of optical path elements.

    Example
    -------
    s = 'd15 | l15/5.5 | d15'
    would result in a seqeunce of a distance element of 15 lu
    (lu=length units), followed by a Lens with focal length 15 lu
    with aperture 5.5 lu and another distance of 15 lu.
    '''
    sequence = []

    for idx, si in enumerate(s.lower().replace(' ', '').split('|')):
        if si.startswith('d'):
            s_ = si.split('/')
            d = float(s_[0][1:])
            if len(s_) == 2:
                a = float(s_[1])
                ope = OPE(d=d, a=a, name='D' + str(idx))
            else:
                ope = OPE(d=d, name='D' + str(idx))
        elif si.startswith('l'):
            s_ = si.split('/')
            f = float(s_[0][1:])
            if len(s_) == 2:
                a = float(s_[1])
                ope = OPE(f=f, a=a, name='L' + str(idx))
            else:
                ope = OPE(f=f, name='L' + str(idx))
        else:
            print('unknown element {}'.format(si))
        sequence.append(ope)

    return sequence


def plot_ray(h, sequence, parallel=False, d=None,
             axis=None, label=None, **pltkws):
    """
    Plot the ray trace through the sequence of OPEs.

    Arguments
    ---------
    h : float
        height of ray.
    sequence : list of OPE
        sequence of optical path elements.
    parallel : bool
        Whether the source ray is a parallel beam.
    d : float
        Diameter of an incoming parallel beam.
    axis : matplotlib.Axis
    label : str
        Label of the plotted line

    Keyword Arguments
    -----------------
    kws passed to plt.plot(**pltkws)

    Returns
    -------
    maplotlib.figure
    """
    if axis:
        fig = axis.figure
        ax = axis
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if not any([a in pltkws for a in ['c', 'col', 'color']]):
        pltkws['color'] = next(ax._get_lines.prop_cycler)['color']

    ax.axhline(color='k', linewidth=0.5, linestyle='--')

    if parallel:
        d = d or 1.0
        rin_0 = [h + d/2, 0.0]
        rin_1 = [h - d/2, 0.0]
    else:
        # get distance and aperture of first aperture
        _, d, aperture = get_first_aperture(sequence)

        a1, a2 = get_angle_lim(h, d, aperture)
        rin_0 = [h, a1]
        rin_1 = [h, a2]

    dist, r0 = trace_ray(rin_0, sequence)
    dist, r1 = trace_ray(rin_1, sequence)

    #pltkws = {'linewidth': 0.5}.update(pltkws)

    ax.plot(dist, r0[0, :], label=label or 'h={:1.2f}'.format(h), **pltkws)
    ax.plot(dist, r1[0, :], **pltkws)

    for x in get_lens_pos(sequence):
        ax.axvline(x=x, ymin=0.02, ymax=0.98, linewidth=0.5, linestyle='--')

    return fig
