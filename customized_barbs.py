# -*- coding: utf-8 -*-
# @Author: wqshen
# @Email: wqshen91@gmail.com
# @Date: 2020/12/27 19:12
# @Last Modified by: wqshen

import numpy as np
from numpy import ma
from matplotlib.patches import CirclePolygon
import matplotlib.transforms as transforms
from matplotlib.quiver import Barbs
from matplotlib import _preprocess_data


class CustomizedBarbs(Barbs):

    def _find_tails(self, mag, rounding=True, calm=1, half=5, full=10, flag=50):
        """
        Find how many of each of the tail pieces is necessary.

        Parameters
        ----------
        mag : `~numpy.ndarray`
            Vector magnitudes; must be non-negative (and an actual ndarray).
        rounding : bool, default: True
            Whether to round or to truncate to the nearest half-barb.
        half, full, flag : float, defaults: 5, 10, 50
            Increments for a half-barb, a barb, and a flag.

        Returns
        -------
        n_flags, n_barbs : int array
            For each entry in *mag*, the number of flags and barbs.
        half_flag : bool array
            For each entry in *mag*, whether a half-barb is needed.
        empty_flag : bool array
            For each entry in *mag*, whether nothing is drawn.
        """
        # If rounding, round to the nearest multiple of half, the smallest
        # increment
        mag_init = mag.copy()  # save initial wind magnitude to check for calm wind
        if rounding:
            mag = half * np.around(mag / half)
        n_flags, mag = divmod(mag, flag)
        n_barb, mag = divmod(mag, full)
        half_flag = mag >= half
        calm_ = (mag_init < calm & ~half_flag)  # check if wind is rather calm, adjusted by WB
        empty_flag = ~(half_flag | (n_flags > 0) | (n_barb > 0))
        empty_flag = empty_flag & calm_  # reajust if wind is really calm, adjusted by WB
        return n_flags.astype(int), n_barb.astype(int), half_flag, empty_flag

    def _make_barbs(self, u, v, nflags, nbarbs, half_barb, empty_flag, length,
                    pivot, sizes, fill_empty, flip):
        """
        Create the wind barbs.

        Parameters
        ----------
        u, v
            Components of the vector in the x and y directions, respectively.

        nflags, nbarbs, half_barb, empty_flag
            Respectively, the number of flags, number of barbs, flag for
            half a barb, and flag for empty barb, ostensibly obtained from
            :meth:`_find_tails`.

        length
            The length of the barb staff in points.

        pivot : {"tip", "middle"} or number
            The point on the barb around which the entire barb should be
            rotated.  If a number, the start of the barb is shifted by that
            many points from the origin.

        sizes : dict
            Coefficients specifying the ratio of a given feature to the length
            of the barb. These features include:

            - *spacing*: space between features (flags, full/half barbs).
            - *height*: distance from shaft of top of a flag or full barb.
            - *width*: width of a flag, twice the width of a full barb.
            - *emptybarb*: radius of the circle used for low magnitudes.

        fill_empty : bool
            Whether the circle representing an empty barb should be filled or
            not (this changes the drawing of the polygon).

        flip : list of bool
            Whether the features should be flipped to the other side of the
            barb (useful for winds in the southern hemisphere).

        Returns
        -------
        list of arrays of vertices
            Polygon vertices for each of the wind barbs.  These polygons have
            been rotated to properly align with the vector direction.
        """

        # These control the spacing and size of barb elements relative to the
        # length of the shaft
        spacing = length * sizes.get('spacing', 0.125)
        full_height = length * sizes.get('height', 0.4)
        full_width = length * sizes.get('width', 0.25)
        empty_rad = length * sizes.get('emptybarb', 0.15)

        # Controls y point where to pivot the barb.
        pivot_points = dict(tip=0.0, middle=-length / 2.)

        endx = 0.0
        try:
            endy = float(pivot)
        except ValueError:
            endy = pivot_points[pivot.lower()]

        # Get the appropriate angle for the vector components.  The offset is
        # due to the way the barb is initially drawn, going down the y-axis.
        # This makes sense in a meteorological mode of thinking since there 0
        # degrees corresponds to north (the y-axis traditionally)
        angles = -(ma.arctan2(v, u) + np.pi / 2)

        # Used for low magnitude.  We just get the vertices, so if we make it
        # out here, it can be reused.  The center set here should put the
        # center of the circle at the location(offset), rather than at the
        # same point as the barb pivot; this seems more sensible.
        circ = CirclePolygon((0, 0), radius=empty_rad).get_verts()

        if fill_empty:
            empty_barb = circ
        else:
            # If we don't want the empty one filled, we make a degenerate
            # polygon that wraps back over itself
            empty_barb = np.concatenate((circ, circ[::-1]))

        barb_list = []
        for index, angle in np.ndenumerate(angles):
            # If the vector magnitude is too weak to draw anything, plot an
            # empty circle instead
            if empty_flag[index]:  # adjusted by WB
                # We can skip the transform since the circle has no preferred
                # orientation
                barb_list.append(empty_barb)
                continue

            poly_verts = [(endx, endy)]
            offset = length

            # Handle if this barb should be flipped
            barb_height = -full_height if flip[index] else full_height

            # Add vertices for each flag
            for i in range(nflags[index]):
                # The spacing that works for the barbs is a little to much for
                # the flags, but this only occurs when we have more than 1
                # flag.
                if offset != length:
                    offset += spacing / 2.
                poly_verts.extend(
                    [[endx, endy + offset],
                     [endx + barb_height, endy - full_width / 2 + offset],
                     [endx, endy - full_width + offset]])

                offset -= full_width + spacing

            # Add vertices for each barb.  These really are lines, but works
            # great adding 3 vertices that basically pull the polygon out and
            # back down the line
            for i in range(nbarbs[index]):
                poly_verts.extend(
                    [(endx, endy + offset),
                     (endx + barb_height, endy + offset + full_width / 2),
                     (endx, endy + offset)])

                offset -= spacing

            # Add the vertices for half a barb, if needed
            if half_barb[index]:
                # If the half barb is the first on the staff, traditionally it
                # is offset from the end to make it easy to distinguish from a
                # barb with a full one
                if offset == length:
                    poly_verts.append((endx, endy + offset))
                    offset -= 1.5 * spacing
                poly_verts.extend(
                    [(endx, endy + offset),
                     (endx + barb_height / 2, endy + offset + full_width / 4),
                     (endx, endy + offset)])

            # If the vector magnitude is too weak to draw anything, plot an
            # empty barb instead
            if not empty_flag[index] and not half_barb[index]:  # adjusted by WB
                poly_verts.append((endx, endy + offset))

            # Rotate the barb according the angle. Making the barb first and
            # then rotating it made the math for drawing the barb really easy.
            # Also, the transform framework makes doing the rotation simple.
            poly_verts = transforms.Affine2D().rotate(-angle).transform(
                poly_verts)
            barb_list.append(poly_verts)

        return barb_list


@_preprocess_data()
def barbs(ax, *args, **kw):
    # Make sure units are handled for x and y values
    args = ax._quiver_units(args, kw)

    b = CustomizedBarbs(ax, *args, **kw)
    ax.add_collection(b, autolim=True)
    ax._request_autoscale_view()
    return b


def test_empty_barb():
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis([0, 1, 0, 1])
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    u = np.array([0, 0.9, 3.5, 7.5, 12.6, 20, 25, 50, 57.4])
    v = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    plt.plot(x, y, 'ko', markersize=2)
    # Here we set parameter fill_empty='barb' to enable the weak wind barb function
    barbs(ax, x, y, u, v, rounding=True, length=8, linewidth=0.5, fill_empty=False,
          barb_increments={'calm': 1, 'half': 5, 'full': 10, 'flag': 50})

    plt.title("Bald barbs for weak wind speed")
    plt.show()


if __name__ == "__main__":
    test_empty_barb()
