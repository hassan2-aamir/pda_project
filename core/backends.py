"""
AHA Communication Backends
---------------------------
Parameter Server (PS) and Ring AllReduce (RA) implementations
using Python multiprocessing primitives (queues, pipes).

In a real cluster these would be gRPC + NCCL.  Here we model the
communication cost explicitly via time.sleep() proportional to
gradient size and simulated bandwidth.
"""

import time
import threading
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


# ════════════════════════════════════════════════════════════════════════ #
# Network simulation helpers                                               #
# ════════════════════════════════════════════════════════════════════════ #

class NetworkLink:
    """
    Models a single network link with configurable bandwidth and packet loss.
    Call simulate_transfer(bytes) to block for the realistic transfer time.
    """

    def __init__(self, bandwidth_gbps: float = 10.0, base_latency_ms: float = 0.1,
                 pkt_loss_rate: float = 0.0, jitter_ms: float = 0.5):
        self.bandwidth_gbps = bandwidth_gbps
        self.base_latency_ms = base_latency_ms
        self.pkt_loss_rate = pkt_loss_rate
        self.jitter_ms = jitter_ms

    def simulate_transfer(self, n_bytes: int) -> float:
        """
        Simulate transferring n_bytes. Returns actual transfer time in ms.
        Includes: propagation latency + transmission time + jitter.
        Retransmissions are modeled if packet loss > 0.
        """
        bandwidth_bps = self.bandwidth_gbps * 1e9
        tx_time_ms = (n_bytes * 8 / bandwidth_bps) * 1000
        jitter = np.random.uniform(0, self.jitter_ms)
        retransmit_factor = 1.0
        if self.pkt_loss_rate > 0:
            # Geometric retransmit model
            n_retx = np.random.geometric(1 - self.pkt_loss_rate) - 1
            retransmit_factor = 1 + 0.5 * n_retx
        total_ms = (self.base_latency_ms + tx_time_ms) * retransmit_factor + jitter
        time.sleep(total_ms / 1000.0)
        return total_ms

    def set_congestion(self, factor: float):
        """Scale latency by factor (1.0 = normal, 5.0 = heavy congestion)."""
        self.base_latency_ms *= factor
        self.jitter_ms *= factor

    def inject_loss(self, rate: float):
        self.pkt_loss_rate = min(max(rate, 0.0), 1.0)


# ════════════════════════════════════════════════════════════════════════ #
# Gradient tensor (simulated)                                              #
# ════════════════════════════════════════════════════════════════════════ #

@dataclass
class GradientTensor:
    """Simulated gradient. In real code this would be a torch.Tensor."""
    values: np.ndarray
    version: int
    worker_id: int

    @property
    def n_bytes(self) -> int:
        return self.values.nbytes

    @classmethod
    def random(cls, size_mb: float, worker_id: int, version: int, seed: int = 0):
        rng = np.random.default_rng(seed + version * 100 + worker_id)
        n_elements = int(size_mb * 1e6 / 4)  # float32: 4 bytes
        return cls(values=rng.standard_normal(n_elements).astype(np.float32),
                   version=version, worker_id=worker_id)


# ════════════════════════════════════════════════════════════════════════ #
# Parameter Server                                                         #
# ════════════════════════════════════════════════════════════════════════ #

class ParameterServer:
    """
    Sharded Parameter Server with bounded-staleness (τ-BSP).
    Each shard holds a partition of the model parameters.
    Workers push gradient slices and pull updated parameters.
    """

    def __init__(self, n_shards: int = 4, tau: int = 2,
                 link: Optional[NetworkLink] = None):
        self.n_shards = n_shards
        self.tau = tau
        self.link = link or NetworkLink()
        self._lock = threading.Lock()
        # Shard state: accumulated gradient sum and version counter
        self._shards: Dict[int, np.ndarray] = {}
        self._versions: Dict[int, int] = {i: 0 for i in range(n_shards)}
        self._worker_versions: Dict[int, int] = {}   # latest version per worker
        self._global_version: int = 0
        self._staleness_violations: int = 0
        self._total_pushes: int = 0
        self._queue_depth: int = 0    # outstanding push count proxy

    def _shard_for(self, grad: GradientTensor) -> List[Tuple[int, np.ndarray]]:
        """Partition gradient tensor across shards by equal slicing."""
        chunk = len(grad.values) // self.n_shards
        slices = []
        for i in range(self.n_shards):
            start = i * chunk
            end = (i + 1) * chunk if i < self.n_shards - 1 else len(grad.values)
            slices.append((i, grad.values[start:end]))
        return slices

    def push(self, grad: GradientTensor) -> float:
        """
        Worker pushes gradient. Returns transfer time in ms.
        Gradient is rejected if it exceeds the staleness bound.
        """
        slice_bytes = grad.n_bytes // self.n_shards
        total_time = 0.0
        # Simulate push of each shard slice
        for _ in range(self.n_shards):
            total_time += self.link.simulate_transfer(slice_bytes)

        with self._lock:
            self._total_pushes += 1
            self._queue_depth += 1
            worker_ver = self._worker_versions.get(grad.worker_id, 0)
            lag = self._global_version - grad.version
            if lag > self.tau:
                self._staleness_violations += 1
                self._queue_depth -= 1
                return total_time   # drop stale update

            slices = self._shard_for(grad)
            for shard_id, slice_vals in slices:
                if shard_id not in self._shards:
                    self._shards[shard_id] = slice_vals.copy()
                else:
                    self._shards[shard_id] = (self._shards[shard_id] + slice_vals) / 2.0
                self._versions[shard_id] += 1

            self._global_version += 1
            self._worker_versions[grad.worker_id] = self._global_version
            self._queue_depth -= 1

        return total_time

    def pull(self, grad_size_bytes: int) -> Tuple[np.ndarray, float]:
        """
        Worker pulls updated parameters. Returns (params, transfer_time_ms).
        """
        slice_bytes = grad_size_bytes // self.n_shards
        total_time = 0.0
        for _ in range(self.n_shards):
            total_time += self.link.simulate_transfer(slice_bytes)

        with self._lock:
            if self._shards:
                params = np.concatenate(list(self._shards.values()))
            else:
                params = np.zeros(grad_size_bytes // 4, dtype=np.float32)

        return params, total_time

    def stats(self) -> dict:
        with self._lock:
            return {
                "global_version": self._global_version,
                "staleness_violations": self._staleness_violations,
                "total_pushes": self._total_pushes,
                "violation_rate": self._staleness_violations / max(self._total_pushes, 1),
                "queue_depth": self._queue_depth,
            }

    def reset(self):
        with self._lock:
            self._shards = {}
            self._versions = {i: 0 for i in range(self.n_shards)}
            self._global_version = 0
            self._worker_versions = {}
            self._staleness_violations = 0
            self._total_pushes = 0


# ════════════════════════════════════════════════════════════════════════ #
# Ring AllReduce                                                            #
# ════════════════════════════════════════════════════════════════════════ #

class RingAllReduce:
    """
    Synchronous Ring AllReduce.
    Workers scatter-reduce then all-gather their gradient chunks.
    The ring is rebuilt if a worker is detected as failed.
    """

    def __init__(self, worker_ids: List[int], link: Optional[NetworkLink] = None):
        self.worker_ids = list(worker_ids)
        self.link = link or NetworkLink()
        self._lock = threading.Lock()
        self._barrier = threading.Barrier(len(worker_ids))
        self._chunks: Dict[int, np.ndarray] = {}   # worker_id → aggregated chunk
        self._scatter_ready: Dict[int, bool] = {w: False for w in worker_ids}
        self._n_workers = len(worker_ids)
        self._ring_rebuild_count = 0

    def remove_worker(self, worker_id: int):
        """Remove a failed/crashed worker and rebuild the ring."""
        with self._lock:
            if worker_id in self.worker_ids:
                self.worker_ids.remove(worker_id)
                self._n_workers = len(self.worker_ids)
                self._ring_rebuild_count += 1
                old_barrier = self._barrier
                self._barrier = threading.Barrier(self._n_workers)
        return self._n_workers

    def allreduce(self, grad: GradientTensor, worker_id: int) -> Tuple[np.ndarray, float]:
        """
        Perform ring allreduce for one gradient tensor.
        Returns (reduced_gradient, comm_time_ms).

        Communication cost per worker:
            2 × (N-1)/N × grad_bytes over the ring.
        """
        N = self._n_workers
        if N <= 1:
            return grad.values.copy(), 0.0

        chunk_bytes = grad.n_bytes // N
        total_time = 0.0

        # ----- Scatter-reduce phase: N-1 hops -----
        for _ in range(N - 1):
            total_time += self.link.simulate_transfer(chunk_bytes)

        # ----- All-gather phase: N-1 hops -----
        for _ in range(N - 1):
            total_time += self.link.simulate_transfer(chunk_bytes)

        with self._lock:
            self._chunks[worker_id] = grad.values.copy() / N  # averaged

        # Barrier: all workers must finish before any advances
        try:
            self._barrier.wait(timeout=10.0)
        except threading.BrokenBarrierError:
            # Ring is broken (straggler or failure) — return partial result
            return grad.values.copy(), total_time

        with self._lock:
            if self._chunks:
                reduced = np.mean(list(self._chunks.values()), axis=0)
            else:
                reduced = grad.values.copy()

        return reduced, total_time

    def stats(self) -> dict:
        return {
            "n_workers": self._n_workers,
            "ring_rebuild_count": self._ring_rebuild_count,
        }
