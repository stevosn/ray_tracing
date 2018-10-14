from numpy import array, matrix, inf, nan, ones, zeros, isinf, isfinite
from numpy import linspace

import matplotlib

matplotlib.use('nbAgg')

import matplotlib.pyplot as plt


def get_image_pos(object_distance, focal_length):
    s = object_distance
    f = focal_length
    return 1 / (1 / f - 1 / s)


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
        self.aperture = abs(a)
        self.name = name
        self.focal_length = f if f else inf
        self.distance = d if d else 0

        if f:
            self.c = -1 / f
        else:
            self.c = 0

        if d:
            self.b = d
        else:
            self.b = 0

    def copy(self, name=None):
        """
        Returns a copy of the OPE.
        """
        a = self.aperture
        name = name if name else f'{self.name}_copy'
        f = self.focal_length
        d = self.distance

        return OPE(a=a, d=d, f=f, name=name)

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
                print('{0:s}: stopped at aperture with height {1:1.3f}'.format(
                    self.name, h))
            return False
        else:
            if verbose:
                print('{0:s}: passed aperture with height {1:1.3f}'.format(
                    self.name, h))
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
    """Take r as input vector and trace the ray through the sequence of
    optical path elements OLEs.

    Arguments
    ---------
    r : 2-tuple
        Vector with the first element beeing the ray height and the
        second being the angle
    sequence : list of optical path elements (OPE)

    Returns
    -------
    (dist, ray)
    dist : array
        Array representing the position of the ray through the trace.
    ray : array
        Array representing the ray vectors at the corresponding
        distances.
    """
    ray0 = zeros((2, len(sequence) + 1))
    ray0[:, 0] = r

    # Do the calculations
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

    Returns
    -------
    List of tuples with index and position of OPE in sequence.
    """
    d = 0.0
    d_ = []
    for idx, m in enumerate(sequence):
        if m.is_lens():
            d_.append((idx, d))
        else:
            d += m.get_travel_length()

    return d_


def get_angle_lim(h, d, a):
    """
    Calculate the upper and lower angles of a source that is at
    distance d to an aperture a.
    """
    # return (arctan((a - h) / d), arctan(-(a + h) / d))
    return (-(a - h) / d, (a + h) / d)


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


def plot_ray(h, sequence, parallel=False, d=None, plot_fan=False,
             axis=None, label=None, plot_statics=True, **pltkws):
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
    plot_fan : bool or int
        Plot a fan of rays.
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

    if plot_statics:
        # draw optical axis
        ax.axhline(color='k', linewidth=0.5, linestyle='--')
        # draw lenses
        for idx, x in get_lens_pos(sequence):
            ax.axvline(x=x, ymin=0.02, ymax=0.98,
                       linewidth=0.5, linestyle='--')
        # draw apertures
        draw_apertures(sequence, axis=ax)

    n = 1 * plot_fan
    
    if parallel:
        d = d or 1.0

        heights = linspace(h - d/2, h + d/2, n)
        while True:
            try:
                dist, r1 = trace_ray((heights.pop(0), 0), sequence)
                ax.plot(dist, r1[0, :], label=label or 'h={:1.2f}'.format(h), **pltkws)
            except:
                # even
                break
            try:
                dist, r2 = trace_ray((heights.pop(-1), 0), sequence)
                ax.plot(dist, r2[0, :], **pltkws)
            except:
                # odd
                break        
    else:
        # get distance and aperture of first aperture
        _, d, aperture = get_first_aperture(sequence)

        a1, a2 = get_angle_lim(h, d, aperture)
        
        if n > 0:
            angles = linspace(a1, a2, n)
        else:
            angles = [a1, a2]

        while True:
            try:
                ray1 = (h, angles.pop(0))
                dist, r1 = trace_ray(ray1, sequence)
                ax.plot(dist, r1[0, :], label=label or 'h={:1.2f}'.format(h), **pltkws)
            except:
                # even
                break
            try:
                ray2 = (h, angles.pop(-1))
                dist, r2 = trace_ray(ray2, sequence)
                ax.plot(dist, r2[0, :], **pltkws)
            except:
                # odd
                break

    return fig


def draw_apertures(sequence, axis=None):
    """
    Draw the apertures in the sequence
    """
    if not axis:
        ax = plt.gca()
    else:
        ax = axis

    plt_kws = dict(linewidth=2, linestyle='-',
                   color='darkslategrey')

    a_max = get_max_aperture(sequence)

    if a_max:
        for idx, x in get_lens_pos(sequence):
            a = sequence[idx].aperture
            ax.plot([x, x], [-2*a_max, -a], **plt_kws)
            ax.plot([x, x], [a, 2*a_max], **plt_kws)


def get_max_aperture(sequence):
    """
    Return largest aperture in sequence.

    Returns None if no apertures are defined.
    """
    apertures = [abs(ope.aperture) for ope in sequence if ope.aperture != inf]

    if not apertures:
        out = None
    else:
        out = max(apertures)

    return out


class Trace(object):
    """
    Manages a tracing.
    """

    def __init__(self, sequence):
        """
        """
        self.sequence = sequence
        self.max_y = 0.0
        self._plot_axis = None
        self._statics_drawn = None

    def _init_axis(self):
        self._statics_drawn = False
        axis = plt.figure().add_subplot(111)
        self._plot_axis = axis

    @property
    def plot_axis(self):
        if not self._plot_axis:
            self._init_axis()
        return self._plot_axis

    @plot_axis.setter
    def plot_axis(self, axis):
        self._plot_axis = axis

    def set_max_y(self, y):
        if abs(y) > self.max_y:
            self.max_y = abs(y)

    def plot_statics(self, axis=None):
        if axis:
            self.plot_axis = axis
        else:
            axis = self.plot_axis

        if not self._statics_drawn:
            # draw optical axis
            axis.axhline(color='k', linewidth=0.5, linestyle='--')
            # draw lenses
            for idx, x in get_lens_pos(self.sequence):
                axis.axvline(x=x, ymin=0.02, ymax=0.98,
                             linewidth=0.5, linestyle='--')
                # draw apertures
            draw_apertures(self.sequence, axis=axis)

            self._statics_drawn = True

        return axis

    def reset_plot(self):
        """ Reset internal plot axis."""
        self._statics_drawn = False

    def plot_rays(self, h, parallel=False, d=None, plot_fan=False,
                  axis=None, label=None, plot_statics=True, **pltkws):
        """
        Plot the ray trace through the sequence of OPEs.

        Arguments
        ---------
        h : float or iterable
            Height(s) of ray .
        sequence : list of OPE
            sequence of optical path elements.
        parallel : bool
            Whether the source ray is a parallel beam.
        d : float
            Diameter of an incoming parallel beam.
        plot_fan : bool or int
            Plot a fan of rays.
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
        try:
            heights = iter(h)
        except TypeError:
            heights = [h]

        if plot_statics:
            ax = self.plot_statics(axis=axis)
        else:
            ax = self.plot_axis

        if not any([a in pltkws for a in ['c', 'col', 'color']]):
            cycle_colors = True

        n = 1 * plot_fan

        for h in heights:
            

            if parallel:
                d = d or 1.0
                hs = list(linspace(h - d/2, h + d/2, n if n > 1 else 2))
                while True:
                    pltkws['color'] = next(ax._get_lines.prop_cycler)['color']
                    try:
                        hin = hs.pop(0)
                        dist, r1 = trace_ray((hin, 0), self.sequence)
                        ax.plot(dist, r1[0, :], label=label or f'r = ({hin:1.2f}, 0)', **pltkws)
                        self.set_max_y(max(abs(r1[0, :])))
                    except:
                        # even
                        break
                    try:
                        dist, r2 = trace_ray((hs.pop(-1), 0), self.sequence)
                        ax.plot(dist, r2[0, :], **pltkws)
                        self.set_max_y(max(abs(r2[0, :])))
                    except:
                        # odd
                        break 
            else:
                if self.has_aperture():
                    # get distance and aperture of first aperture
                    _, d, aperture = get_first_aperture(self.sequence)
                    a1, a2 = get_angle_lim(h, d, aperture)
                else:
                    try:
                        _, d = get_lens_pos(self.sequence)[0]
                    except:
                        d = inf
                    a1 = -h/d
                    a2 = 0

                if n > 0:
                    angles = list(linspace(a1, a2, n))
                else:
                    angles = [a1, a2]

                while True:
                    pltkws['color'] = next(ax._get_lines.prop_cycler)['color']
                    try:
                        ang = angles.pop(0)
                        ray1 = (h, ang)
                        dist, r1 = trace_ray(ray1, self.sequence)
                        ax.plot(dist, r1[0, :], label=label or f'r =({h:1.2f}, {ang:1.2f})', **pltkws)
                        self.set_max_y(max(abs(r1[0, :])))
                    except:
                        # even
                        break
                    try:
                        ray2 = (h, angles.pop(-1))
                        dist, r2 = trace_ray(ray2, self.sequence)
                        ax.plot(dist, r2[0, :], **pltkws)
                        self.set_max_y(max(abs(r2[0, :])))
                    except:
                        # odd
                        break

        return ax

    def adjust_ylims(self, axis):
        """
        Adjusts the y limits of the plot according to the apertures
        and rays.
        """
        max_y = self.max_y
        max_a = get_max_aperture(self.sequence)

        y = max([max_y, max_a])
        axis.set_ylim([-1.1*y, 1.1*y])

    def get_idx_aperture_stop(self):
        """
        Find the optical element the defines the aperture stop of the
        system.

        Returns the index of the OPE in the sequence.
        """
        if get_max_aperture(self.sequence) is None:
            print('No apertures found.')
            out = None
        else:
            angle = 1e-10
            while True:
                _, rays = trace_ray((0, angle), self.sequence)
                if all(isfinite(rays[0])):
                    break
                else:
                    angle = angle / 2
            ratio = 0.0
            for idx, ope in enumerate(self.sequence):
                ratio_ = abs(rays[0, idx+1]) / ope.aperture
                if ratio_ > ratio:
                    ratio = ratio_
                    out = idx

        return out

    def has_aperture(self):
        if get_max_aperture(self.sequence):
            return True
        else:
            return False

    def get_aperture_stop_position(self, verbose=False):
        """
        Reduce sequence upto aperture and get distance from lens pos
        function.
        """
        _, d = get_lens_pos(
            self.sequence[:self.get_idx_aperture_stop() + 1])[-1]

        if verbose:
            print(f'aperture stop position = {d:1.2f}')

        return d

    def get_aperture_stop_size(self, verbose=False):
        """
        Return the (half) aperture stop size.
        """
        a = self.sequence[self.get_idx_aperture_stop()].aperture
        if verbose:
            print(f'aperture half-diameter = {a:1.2f}')
        return a

    def indicate_aperture_stop(self, axis=None, color='orangered', verbose=False):
        """
        Draw the aperture stop in the ray tracing diagram.
        """
        if axis:
            ax = axis
        else:
            ax = self.plot_axis
        plt_kws = dict(linewidth=2, linestyle='-', color=color)

        x = self.get_aperture_stop_position(verbose=verbose)
        a = self.get_aperture_stop_size(verbose=verbose)
        y_max = self._get_y_max(verbose=verbose)

        ax.plot([x, x], [a, y_max], **plt_kws)
        ax.plot([x, x], [-y_max, -a], **plt_kws)
        
    def calc_entrance_pupil_position(self, verbose=False):
        """ sequence of OPEs preceeding the aperture stop """
        sequence_prec = self.sequence[:self.get_idx_aperture_stop()]
        d_ap = self.get_aperture_stop_position(verbose=verbose)
        x = d_ap
        mag = 1.0
        for idx, lens_pos in get_lens_pos(sequence_prec)[::-1]:
            # object distance
            d_obj = x - lens_pos
            # image distance
            d_img = get_image_pos(d_obj, sequence_prec[idx].focal_length)
            if verbose > 1:
                print(f'imaging lens position = {lens_pos:1.2f}')
                print(f'x_before = {x:1.2f}')
                print(f'd_obj = {d_obj:1.2f}')
                print(f'd_img = {d_img:1.2f}')

            x = lens_pos - d_img
            mag = mag * d_img / d_obj
            if verbose > 1:
                print(f'x_after = {x:1.2f}')
        return x

    def calc_entrance_pupil_size(self, verbose=False):
        """
        Return the size of the entrance pupil.
        """
        sequence_prec = self.sequence[:self.get_idx_aperture_stop()]
        d_ap = self.get_aperture_stop_position(verbose=verbose)
        x = d_ap
        mag = 1.0
        for idx, lens_pos in get_lens_pos(sequence_prec)[::-1]:
            # object distance
            d_obj = x - lens_pos
            # image distance
            d_img = get_image_pos(d_obj, sequence_prec[idx].focal_length)
            if verbose > 1:
                print(f'imaging lens position = {lens_pos:1.2f}')
                print(f'magnification = {mag:1.2f}')
                print(f'd_obj = {d_obj:1.2f}')
                print(f'd_img = {d_img:1.2f}')

            x = lens_pos - d_img
            mag = mag * d_img / d_obj
            if verbose > 1:
                print(f'magnification_after = {mag:1.2f}')

        en_pupil = self.get_aperture_stop_size() * abs(mag)

        return en_pupil

    def draw_entrance_pupil(self, axis=None, color='orangered', verbose=False):
        """
        Draw the apparent entrance pupil.
        """
        if axis:
            ax = axis
        else:
            ax = self.plot_axis
        x = self.calc_entrance_pupil_position(verbose=verbose)
        a = self.calc_entrance_pupil_size(verbose=verbose)
        y_max = self.max_y

        plt_kws = dict(linewidth=2, linestyle='-', color=color)
        ax.plot([x, x], [a, y_max], **plt_kws)
        ax.plot([x, x], [-y_max, -a], **plt_kws)

    def _get_y_max(self, verbose=False):
        """
        Return maximum y_value in plot.
        """
        if get_max_aperture(self.sequence):
            y_max = 2 * get_max_aperture(self.sequence)
        else:
            y_max = 2 * self.max_y

        if verbose > 1:
            print(f'y_max = {y_max:1.2f}')

        return y_max

    def get_NA(self):
        """
        Return the NA of the optical system.

        This is the entrance pupil size divided by the entrance pupil
        distance.
        """
        d = self.calc_entrance_pupil_position()
        a = self.calc_entrance_pupil_size()

        return a/d
        
