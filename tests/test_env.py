"""Environment smoke tests. The model forward passes are marked slow:
    pytest -m "not slow"   # versions + mmcv ops only
"""

import pytest

from sf6mr import envcheck


def _assert_ok(report):
    failures = [r for r in report.results if r.required and not r.ok]
    assert not failures, "\n" + report.format()


def test_versions_and_ops():
    report = envcheck.Report()
    envcheck.check_versions(report)
    envcheck.check_mmcv_ops(report, envcheck.available_devices())
    _assert_ok(report)


@pytest.mark.slow
@pytest.mark.parametrize("check", [envcheck.check_rtmdet, envcheck.check_rtmpose, envcheck.check_stgcnpp])
def test_model_forward_cpu(check):
    report = envcheck.Report()
    check(report, "cpu", required=True)
    _assert_ok(report)
