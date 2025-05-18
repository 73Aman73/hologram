"""
Minimal core for bacterial-hologram initial-guess pipeline
"""

import cv2
import numpy as np
from scipy import ndimage

# -----------------------------------------------------------
# 1. (X, Y) centre via gradient-voting
# -----------------------------------------------------------
def gradient_vote_center(img_gray: np.ndarray,
                         mag_thresh: float = 10,
                         gauss_sigma: float = 3) -> tuple[int, int]:
    """
    Vote along gradient directions to locate the planar centre.
    Returns (cx, cy) in image coordinates.
    """
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)

    acc = np.zeros_like(img_gray, dtype=np.float64)
    h, w = img_gray.shape
    for y in range(h):
        for x in range(w):
            if mag[y, x] > mag_thresh:
                vx, vy = gx[y, x], gy[y, x]
                norm = np.hypot(vx, vy)
                vx, vy = vx / norm, vy / norm
                for d in (-1, 1):          # forward & backward
                    for step in range(1, min(h, w)//2):
                        xx = int(x + d*step*vx)
                        yy = int(y + d*step*vy)
                        if 0 <= xx < w and 0 <= yy < h:
                            acc[yy, xx] += 1
                        else:
                            break
    acc = ndimage.gaussian_filter(acc, sigma=gauss_sigma)
    cy, cx = np.unravel_index(acc.argmax(), acc.shape)
    return int(cx), int(cy)

# -----------------------------------------------------------
# 2. Azimuth angle via 1-D FFT-energy difference
# -----------------------------------------------------------
def line_profile(img: np.ndarray, centre: tuple[int,int],
                 angle_deg: float, radius: int) -> np.ndarray:
    """Sample intensities from centre outward along given angle."""
    theta = np.deg2rad(angle_deg)
    dx, dy = np.sin(theta), -np.cos(theta)
    cx, cy = centre
    vals = []
    for r in range(radius+1):
        x = int(round(cx + r*dx))
        y = int(round(cy + r*dy))
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            vals.append(img[y, x])
        else:
            break
    return np.array(vals)

def highfreq_energy(sig: np.ndarray) -> float:
    if sig.size < 2:
        return 0.0
    sig = sig - sig.mean()
    mag = np.abs(np.fft.fft(sig))
    return float(np.sum(mag[1:]**2))

def fft_orientation(img_gray: np.ndarray,
                    step_deg: int = 1) -> float:
    centre = (img_gray.shape[1]//2, img_gray.shape[0]//2)
    radius = min(img_gray.shape)//2 - 1
    best_ang, best_diff = None, -1.0
    for ang in range(0, 180, step_deg):
        e_main = highfreq_energy(line_profile(img_gray, centre, ang, radius))
        e_perp = highfreq_energy(line_profile(img_gray, centre, ang+90, radius))
        diff = abs(e_main - e_perp)
        if diff > best_diff:
            best_diff  = diff
            best_ang   = ang if e_main > e_perp else ang+90
    fringe_dir = best_ang % 180
    bact_axis  = (fringe_dir + 90) % 180   # bacterial long axis ⟂ fringes
    return float(bact_axis)

# -----------------------------------------------------------
# 3. Z & β joint score
#    (radial minima ΔV  +  directional contrast ΔC)
# -----------------------------------------------------------
def radial_minima(img2d: np.ndarray, n: int = 3,
                  centre: tuple[int,int]|None=None) -> np.ndarray:
    """First n radial minima distances (pixels)."""
    if centre is None:
        cy, cx = img2d.shape[0]//2, img2d.shape[1]//2
    else:
        cx, cy = centre
    y, x = np.indices(img2d.shape)
    r = np.sqrt((x-cx)**2 + (y-cy)**2).astype(int)
    prof = np.bincount(r.ravel(), img2d.ravel()) / np.bincount(r.ravel())
    mins = [k for k in range(1, len(prof)-1)
            if prof[k] < prof[k-1] and prof[k] < prof[k+1]]
    return np.array(mins[:n], dtype=int)

def directional_contrast(img2d: np.ndarray, axis_angle: float,
                         radius: int|None=None) -> float:
    perp_angle = (axis_angle + 90) % 180
    prof_axis  = line_profile(img2d, (img2d.shape[1]//2,
                                      img2d.shape[0]//2),
                              axis_angle, radius or min(img2d.shape)//2-1)
    prof_perp  = line_profile(img2d, (img2d.shape[1]//2,
                                      img2d.shape[0]//2),
                              perp_angle, radius or min(img2d.shape)//2-1)
    return np.std(prof_axis) / (np.std(prof_perp) + 1e-9)

def delta_V(exp_img2d: np.ndarray, sim_img2d: np.ndarray,
            weights=(1.0, 0.5, 0.25)) -> float:
    r_exp = radial_minima(exp_img2d)
    r_sim = radial_minima(sim_img2d)
    alpha = np.array(weights)[:len(r_exp)]
    return float(np.sum(alpha * np.abs(r_exp - r_sim)))

def delta_C(exp_img2d: np.ndarray, sim_img2d: np.ndarray,
            exp_axis: float, sim_axis: float) -> float:
    C_exp = directional_contrast(exp_img2d, exp_axis)
    C_sim = directional_contrast(sim_img2d, sim_axis)
    return abs(C_exp - C_sim)

# -----------------------------------------------------------
# 4. Overall score S = w1·ΔV + w2·ΔC
# -----------------------------------------------------------
def S_score(exp_img2d: np.ndarray, sim_img2d: np.ndarray,
            exp_axis: float, sim_axis: float,
            w1: float = 0.6, w2: float = 0.4) -> float:
    return w1 * delta_V(exp_img2d, sim_img2d) + \
           w2 * delta_C(exp_img2d, sim_img2d, exp_axis, sim_axis)