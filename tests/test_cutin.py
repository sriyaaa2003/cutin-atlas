import numpy as np

from cutin import param
from cutin.config import load
from cutin.extract import lane_centres
from cutin.replay import replay
from cutin.study_cluster import stratified_variances

CFG = load()
P = CFG["replay"]
T, C, FPS = 151, 75, 25.0


def smooth_offset(width_s=1.0, n=3):
    t = np.arange(T) / FPS
    return np.tile(0.5 * (1 - np.tanh((t - t[C]) / width_s)), (n, 1))


KW = dict(aeb_ttc=1.2, sensor_delay=0.1, max_brake=9.0)


def run(gap0, vc, vf0, off, seed=0):
    n = len(off)
    return replay(off, np.full(n, gap0), np.tile(vc[:, None], (1, T)) if np.ndim(vc) == 1 else vc, np.full(n, vf0),
                  np.full(n, 9.0), p=P, rng=np.random.default_rng(seed), **KW)


def test_cutter_in_other_lane_never_collides():
    off = np.full((4, T), 1.0)
    o = run(5.0, np.full(4, 20.0), 30.0, off)
    assert not o["crash"].any() and np.isnan(o["t_seen"]).all()


def test_cut_in_ahead_of_a_faster_follower_collides():
    # the follower is 10 m/s faster and only 4 m behind when the cutter arrives: no ACC + AEB system can avoid it
    o = run(34.0, np.full(3, 25.0), 35.0, smooth_offset())
    assert o["crash"].all() and (o["impact_dv"] > 0).all()


def test_distant_equal_speed_cut_in_is_safe():
    o = run(60.0, np.full(3, 30.0), 30.0, smooth_offset())
    assert not o["crash"].any() and np.isfinite(o["min_ttc"]).sum() == 0 or (o["min_ttc"] > 5).all()


def test_replay_is_deterministic_for_a_seed():
    a, b = run(15.0, np.full(3, 25.0), 30.0, smooth_offset(), 4), run(15.0, np.full(3, 25.0), 30.0, smooth_offset(), 4)
    assert (a["peak_decel"] == b["peak_decel"]).all()


def test_minjerk_shape_endpoints():
    assert abs(param.minjerk(np.array([0.0]))[0] - 1) < 1e-12 and abs(param.minjerk(np.array([1.0]))[0]) < 1e-12


def test_fit_recovers_a_sigmoid_lane_change():
    off = smooth_offset(n=2)
    lw = np.array([3.7, 3.7])
    recon, seg, valid = param.fit_all(off, lw, FPS, CFG, C)
    assert param.rmse_cm(off, recon["tanh sigmoid"], lw).max() < 0.5
    assert param.rmse_cm(off, recon["polynomial degree 7 (window)"], lw).max() < 2.0


def test_lane_centres_follow_highd_numbering():
    class RM:
        upperLaneMarkings = "8.51;12.59;16.43"
        lowerLaneMarkings = "21.00;24.96;28.80"
    lc = lane_centres(RM())
    assert sorted(lc) == [2, 3, 5, 6] and abs(lc[2][0] - 10.55) < 1e-9


def test_stratified_variance_never_exceeds_srs_for_proportional_allocation_and_matches_known_case():
    y = np.array([0.0] * 50 + [1.0] * 50)
    strata = np.array([0] * 50 + [1] * 50)
    v_srs, v_prop, v_ney, eta = stratified_variances(y, strata, 20)
    assert v_prop < 1e-12 and eta > 0.999 and v_srs > 0
    shuffled = np.random.default_rng(0).permutation(strata)
    assert stratified_variances(y, shuffled, 20)[1] <= v_srs * 1.05
