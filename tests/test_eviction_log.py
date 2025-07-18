import pickle
import time

import lmdb

from datacreek.analysis import mapper
from datacreek.utils.evict_log import clear_eviction_logs, evict_logs, log_eviction


def test_eviction_log_ring_buffer():
    clear_eviction_logs()
    for i in range(105_000):
        log_eviction(f"k{i}", time.time(), "ttl")
    assert len(evict_logs) == 100_000
    assert evict_logs[0].key == "k5000"


def test_manual_eviction(tmp_path):
    env = lmdb.open(str(tmp_path / "db"), map_size=1 << 20)
    with env.begin(write=True) as txn:
        txn.put(b"k0", pickle.dumps((time.time(), b"d")))
    clear_eviction_logs()
    mapper.delete_l2_entry(env, "k0")
    assert any(log.cause == "manual" and log.key == "k0" for log in evict_logs)
    env.close()
