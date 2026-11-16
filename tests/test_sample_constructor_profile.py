"""A diagnostic that records failed workers must also fail its CLI status."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import profile_sample_constructor as profile


@pytest.mark.parametrize('status,expected',[('failed',1),('passed',0)])
def test_coordinator_cli_status_exposes_failed_diagnostic(monkeypatch,tmp_path,status,expected):
    monkeypatch.setattr(profile,'coordinator',lambda args:{'status':status})
    monkeypatch.setattr(sys,'argv',['profile_sample_constructor.py',
        '--baseline-python','before-python','--candidate-python','after-python',
        '--baseline-wheel','before.whl','--candidate-wheel','after.whl',
        '--fixtures',str(tmp_path),'--request','request.json',
        '--build-manifest','build.json','--output',str(tmp_path/'new'),
        '--acknowledge-license'])
    assert profile.main()==expected
