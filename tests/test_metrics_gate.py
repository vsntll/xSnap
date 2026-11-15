from check_model_metrics import LOG_LOSS_RISE, compare


def _m(elo, v1=None):
    m = {"elo": {"log_loss": elo, "margin_mae": 10.0}}
    if v1 is not None:
        m["outcome_v1"] = {"log_loss": v1, "margin_mae": 10.0}
    return m


def test_small_moves_and_improvements_pass():
    assert compare(_m(0.636, 0.626), _m(0.638, 0.620)) == []


def test_log_loss_rise_past_tolerance_fails():
    fails = compare(_m(0.636, 0.626), _m(0.636, 0.626 + LOG_LOSS_RISE + 0.001))
    assert len(fails) == 1 and fails[0].startswith("outcome_v1 log_loss")


def test_value_r2_drop_past_tolerance_fails():
    old, new = _m(0.636), _m(0.636)
    old["value"], new["value"] = {"model": {"r2": 0.446}}, {"model": {"r2": 0.42}}
    fails = compare(old, new)
    assert len(fails) == 1 and fails[0].startswith("value r2")
    new["value"]["model"]["r2"] = 0.44
    assert compare(old, new) == []


def test_model_missing_in_baseline_is_skipped():
    assert compare(_m(0.636), _m(0.636, 0.9)) == []
