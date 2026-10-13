from solution.analysis import paired_ci


def test_same_predictions_have_zero_paired_uncertainty():
    entries = [dict(session="a"), dict(session="a"), dict(session="b")]
    rows = [(1, 0, 0), (0, 2, 1), (2, 0, 3)]
    for by_session in (False, True):
        assert paired_ci(entries, rows, rows, n_boot=100, by_session=by_session) == [0.0, 0.0]


def test_uniform_improvement_has_known_paired_difference():
    entries = [dict(session="a"), dict(session="a"), dict(session="b")]
    bad, good = [(0, 1, 1)]*3, [(1, 0, 0)]*3
    assert paired_ci(entries, bad, good, n_boot=100, by_session=True) == [1.0, 1.0]
