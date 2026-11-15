"""RSS evidence checks use actual worker reports, never launcher sampling."""
import importlib.util
import os
from pathlib import Path
import sys
import pytest

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
import profile_native_memory as memory


def test_worker_self_report_peak_and_phase_pid_are_consistent():
    pid=os.getpid()+1000
    worker={'pid':pid,'full_source_nifti_loaded':False,
        'snapshots':[{'pid':pid,'rss_bytes':20,'lifetime_peak_rss_bytes':30}]}
    footer={'pid':pid,'rss_bytes':25,'lifetime_peak_rss_bytes':35}
    assert memory.validate_worker(worker,footer)==35
    footer['pid']=pid+1
    with pytest.raises(ValueError,match='PID mismatch'):
        memory.validate_worker(worker,footer)


def test_setup_or_invalid_peak_cannot_be_misrepresented_as_phase_evidence():
    pid=os.getpid()+1000
    worker={'pid':pid,'full_source_nifti_loaded':True,
        'snapshots':[{'pid':pid,'rss_bytes':20,'lifetime_peak_rss_bytes':30}]}
    footer={'pid':pid,'rss_bytes':25,'lifetime_peak_rss_bytes':35}
    with pytest.raises(ValueError,match='Full-volume'):
        memory.validate_worker(worker,footer)
    worker['full_source_nifti_loaded']=False
    footer['lifetime_peak_rss_bytes']=10
    with pytest.raises(ValueError,match='Invalid own-process'):
        memory.validate_worker(worker,footer)


def test_live_memory_snapshot_is_this_process():
    snapshot=memory.own_memory('test')
    assert snapshot['pid']==os.getpid()
    assert snapshot['rss_bytes']>0
    assert snapshot['lifetime_peak_rss_bytes']>=snapshot['rss_bytes']
