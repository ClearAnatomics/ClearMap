# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Modified by C. Kirst to allow arbitrary rotations

from __future__ import division

import numpy as np
from vispy.scene import TurntableCamera

from vispy.util import keys

MOUSE_LEFT = 1
MOUSE_RIGHT = 2


class ArbitraryRotationCamera(TurntableCamera):
    """ 3D camera class that orbits around a center point while
    maintaining a view on a center point.

    For this camera, the ``scale_factor`` indicates the zoom level, and
    the ``center`` indicates the position to put at the center of the
    view.

    Parameters
    ----------
    fov : float
        Field of view. Zero (default) means orthographic projection.
    elevation : float
        Elevation angle in degrees. Positive angles place the camera
        above the center point, negative angles place the camera below
        the center point.
    azimuth : float
        Azimuth angle in degrees. Zero degrees places the camera on the
        positive x-axis, pointing in the negative x direction.
    roll : float
        Roll angle in degrees
    distance : float | None
        The distance of the camera from the rotation point (only makes sense
        if fov > 0). If None (default) the distance is determined from the
        scale_factor and fov.
    **kwargs : dict
        Keyword arguments to pass to `BaseCamera`.

    Notes
    -----
    Interaction:

        * LMB: orbits the view around its center point.
        * RMB or scroll: change scale_factor (i.e. zoom level)
        * SHIFT + LMB: translate the center point
        * SHIFT + RMB: change FOV

    """

    def __init__(self, fov=45.0, elevation=30.0, azimuth=30.0, roll=0.0, distance=None, translate_speed=1.0, **kwargs):  # REFACTOR: theoretically useless
        super().__init__(fov=fov, elevation=elevation, azimuth=azimuth, roll=roll,
                         distance=distance, translate_speed=translate_speed, **kwargs)

    @property
    def elevation(self):
        """ The angle of the camera in degrees above the horizontal (x, z)
        plane.
        """
        return self._elevation

    @elevation.setter
    def elevation(self, elev):
        elev = float(elev)
        while elev < -180:
            elev += 360
        while elev > 180:
            elev -= 360        
        self._elevation = elev
        self.view_changed()

    def orbit(self, azim, elev):
        """ Orbits the camera around the center position.

        Parameters
        ----------
        azim : float
            Angle in degrees to rotate horizontally around the center point.
        elev : float
            Angle in degrees to rotate vertically around the center point.
        """
        self.azimuth += azim
        self.elevation = self.elevation + elev  # np.clip(self.elevation + elev, -90, 90)
        self.view_changed()

    def _update_roll(self, event):
        """Update roll parameter based on mouse movement"""
        _, _, d = self._get_mouse_points(event)
        self._event_value = self.roll if self._event_value is None else self._event_value
        self.roll = self._event_value + d[0]

    def _update_elevation(self, event):
        """Update elevation parameter based on mouse movement"""
        _, _, d = self._get_mouse_points(event)
        self._event_value = self.elevation if self._event_value is None else self._event_value
        self.elevation = self._event_value + d[0]
        
    def _update_azimuth(self, event):
        """Update azimuth parameter based on mouse movement"""
        _, _, d = self._get_mouse_points(event)
        self._event_value = self.azimuth if self._event_value is None else self._event_value
        self.azimuth = self._event_value + d[0]
    
    def _update_zoom(self, event):
        """Update zoom based on mouse movement"""
        _, _, d = self._get_mouse_points(event)
        if self._event_value is None:
            self._event_value = (self._scale_factor, self._distance)
        zoom_y = (1 + self.zoom_factor) ** d[1]
        
        self.scale_factor = self._event_value[0] * zoom_y
        # Modify distance if given
        if self._distance is not None:
            self._distance = self._event_value[1] * zoom_y
        self.view_changed()

    def _get_mouse_points(self, event):
        p1 = event.mouse_event.press_event.pos
        p2 = event.mouse_event.pos
        d = p2 - p1
        return p1,  p2, d

    def _update_translate(self, event):
        """Update translate parameter based on mouse movement"""
        p1, p2, _ = self._get_mouse_points(event)
        
        norm = np.mean(self._viewbox.size)
        if self._event_value is None or len(self._event_value) == 2:
            self._event_value = self.center
        dist = (p1 - p2) / norm * self._scale_factor  # FIXME: why invert d
        dist[1] *= -1
        # Black magic part 1: turn 2D into 3D translations
        dx, dy, dz = self._dist_to_trans(dist)
        # Black magic part 2: take up-vector and flipping into account
        ff = self._flip_factors
        up, forward, right = self._get_dim_vectors()
        dx, dy, dz = right * dx + forward * dy + up * dz
        dx, dy, dz = ff[0] * dx, ff[1] * dy, ff[2] * dz
        c = self._event_value
        self.center = c[0] + dx, c[1] + dy, c[2] + dz
    
    def _update_fov(self, event):
        """Update fov parameter based on mouse movement"""
        p1, p2, d = self._get_mouse_points(event)
        self._event_value = self._fov if self._event_value is None else self._event_value
        fov = self._event_value - d[1] / 5.0
        self.fov = min(180.0, max(0.0, fov))
    
    def _rotate_tr(self):
        """Rotate the transformation matrix based on camera parameters"""
        up, forward, right = self._get_dim_vectors()
        super()._rotate_tr()
        self.transform.rotate(self.roll, forward)

    def viewbox_mouse_event(self, event):
        """
        The viewbox received a mouse event; update transform
        accordingly.

        Parameters
        ----------
        event : instance of Event
            The event.
        """
        if event.handled or not self.interactive:
            return

        # PerspectiveCamera.viewbox_mouse_event(self, event)
        if event.type == 'mouse_wheel':
            s = 1.1 ** - event.delta[1]
            self._scale_factor *= s
            if self._distance is not None:
                self._distance *= s
            self.view_changed()
        elif event.type == 'mouse_release':
            self._event_value = None  # Reset
        elif event.type == 'mouse_press':
            event.handled = True
        elif event.type == 'mouse_move':
            if event.press_event is None:
                return

            modifiers = event.mouse_event.modifiers
            if MOUSE_LEFT in event.buttons:
                if not modifiers:  # Rotate
                    self._update_rotation(event)
                elif keys.CONTROL in modifiers:  # roll only
                    self._update_roll(event)
                elif keys.META in modifiers:  # elevation only
                    self._update_elevation(event)
                elif keys.SHIFT in modifiers:  # azimuth only
                    self._update_azimuth(event)
            elif MOUSE_RIGHT in event.buttons:
                if not modifiers:  # zoom
                    self._update_zoom(event)
                elif keys.SHIFT in modifiers:  # Translate
                    self._update_translate(event)
                elif keys.META in modifiers:  # Change fov
                    self._update_fov(event)
