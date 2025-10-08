
"""Utility functions for loading track data and coordinate transformations (Frenet <-> global)."""
import numpy as np
import os
from pathlib import Path
from leap_c.examples.race_car.config import CarParams, get_default_car_params


def get_track(car_params: CarParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads the track data and scales it with the car size."""
    track = "track.txt"  # not likely to change, so hardcoded here
    track_file = os.path.join(str(Path(__file__).parent), track)
    array = np.loadtxt(track_file)
    sref = array[:,0]  # [m]
    xref = array[:,1]  # [m]
    yref = array[:,2]  # [m]
    psiref = array[:,3]  # [rad]
    kapparef = array[:,4]  # [1/m]

    # the track is defined for the small car
    # we need to scale it with the given car size
    default_car_length = get_default_car_params().l.item()
    current_car_length = car_params.l.item()
    scaling_factor = current_car_length / default_car_length
    sref *= scaling_factor
    xref *= scaling_factor
    yref *= scaling_factor
    kapparef /= scaling_factor

    return sref, xref, yref, psiref, kapparef


def frenet2global(s, n, alpha, v, car_params: CarParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transforms the projected coordinates back to the global frame."""
    [sref, xref, yref, psiref, _] = get_track(car_params=car_params)
    track_length = sref[-1]
    si = s % track_length
    idxmindist = find_closest_s(si, sref)
    idxmindist2 = find_second_closest_s(si, sref, idxmindist)
    t = (si - sref[idxmindist]) / (sref[idxmindist2] - sref[idxmindist])
    x0 = (1 - t) * xref[idxmindist] + t * xref[idxmindist2]
    y0 = (1 - t) * yref[idxmindist] + t * yref[idxmindist2]
    psi0 = (1 - t) * psiref[idxmindist] + t * psiref[idxmindist2]

    x = x0 - n * np.sin(psi0)
    y = y0 + n * np.cos(psi0)
    psi = psi0 + alpha
    v = v
    return x, y, psi, v


def find_closest_s(si, sref):
    """Find indices of closest points in sref for each s in si."""
    si = np.atleast_1d(si)
    # Compute absolute differences between each si and all sref points
    diffs = np.abs(si[:, None] - sref[None, :])  # shape (len(si), len(sref))
    idx = np.argmin(diffs, axis=1)
    
    # If input was scalar, return scalar index
    return idx.item() if idx.size == 1 else idx


def find_second_closest_s(si, sref, idxmindist):
    """Find indices of second closest points in sref for each s in si."""
    d_prev = abs(si - sref[(idxmindist - 1) % sref.size])  # distance to previous node (wrap-around)
    d_next = abs(si - sref[(idxmindist + 1) % sref.size])  # distance to next node (wrap-around)
    
    # Choose the neighbor with the smaller distance
    idx2 = np.where(d_prev > d_next,
                    (idxmindist + 1) % sref.size,
                    (idxmindist - 1) % sref.size)
    return idx2


def global2frenet(x, y, psi, v, car_params: CarParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    [sref, xref, yref, psiref, _] = get_track(car_params=car_params)
    idxmindist = find_closest_point(x, y, xref, yref)
    idxmindist2 = find_closest_neighbour(x, y, xref, yref, idxmindist)
    t = find_projection(x, y, xref, yref, sref, idxmindist, idxmindist2)
    s0 = (1 - t) * sref[idxmindist] + t * sref[idxmindist2]
    x0 = (1 - t) * xref[idxmindist] + t * xref[idxmindist2]
    y0 = (1 - t) * yref[idxmindist] + t * yref[idxmindist2]
    psi0 = (1 - t) * psiref[idxmindist] + t * psiref[idxmindist2]

    s = s0
    n = np.cos(psi0)*(y-y0)-np.sin(psi0)*(x-x0)
    alpha = psi-psi0
    v = v
    return s,n,alpha,v


def find_projection(x, y, xref, yref, sref, idxmindist, idxmindist2):
    vabs = abs(sref[idxmindist]-sref[idxmindist2])
    vl = np.empty(2)
    u = np.empty(2)
    vl[0] = xref[idxmindist2]-xref[idxmindist]
    vl[1] = yref[idxmindist2]-yref[idxmindist]
    u[0] = x-xref[idxmindist]
    u[1] = y-yref[idxmindist]
    t = (vl[0]*u[0]+vl[1]*u[1])/vabs/vabs
    return t


def find_closest_point(x, y, xref, yref):
    mindist = 1
    idxmindist = 0
    for i in range(xref.size):
        dist = dist2D(x, xref[i], y, yref[i])
        if dist < mindist:
            mindist = dist
            idxmindist = i
    return idxmindist


def find_closest_neighbour(x, y, xref, yref, idxmindist):
    dist_before = dist2D(x, xref[idxmindist-1], y, yref[idxmindist-1])
    dist_after = dist2D(x, xref[idxmindist+1], y, yref[idxmindist+1])
    if dist_before < dist_after:
        idxmindist2 = idxmindist-1
    else:
        idxmindist2 = idxmindist+1
    if (idxmindist2 < 0):
        idxmindist2 = xref.size-1
    elif (idxmindist == xref.size):
        idxmindist2 = 0
    return idxmindist2


def dist2D(x1, x2, y1, y2):
    return np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))


if __name__ == "__main__":
    car_params = get_default_car_params()
    s = 4.5
    n = 0.2
    alpha = 0.1
    v = 10.0
    s2, n2, alpha2, v2 = global2frenet(*frenet2global(s, n, alpha, v, car_params), car_params)
    assert np.allclose(s, s2)
    assert np.allclose(n, n2)
    assert np.allclose(alpha, alpha2)
    assert np.allclose(v, v2)

    print("ok")



