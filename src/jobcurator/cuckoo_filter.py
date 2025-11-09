from __future__ import annotations

import hashlib
import os
import random
from typing import Any


class CuckooFilter:
    """
    Simple Cuckoo Filter implementation.

    This is used as an approximate "seen set" for job exact hashes
    in incremental pipelines.

    Core ideas:
      - Fixed number of buckets
      - Each bucket holds up to `bucket_size` small fingerprints
      - Insert:
          * compute fingerprint fp and two candidate buckets i1, i2
          * try to place fp in either bucket
          * if both full, randomly evict a fingerprint and relocate it
            (up to max_kicks times)
      - Lookup:
          * check if fp is present in either i1 or i2

    API:
      - cf = CuckooFilter(capacity=1_000_000)
      - cf.add(item)
      - item in cf          # uses __contains__
      - cf.contains(item)
      - cf.remove(item)     # best-effort delete

    Notes:
      - This is probabilistic: false positives are possible,
        false negatives are rare (only if removal is used heavily).
      - Designed to be picklable (used with SQL/local backends).
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        bucket_size: int = 4,
        fingerprint_size: int = 1,
        max_kicks: int = 500,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")

        if bucket_size <= 0:
            raise ValueError("bucket_size must be > 0")

        if fingerprint_size <= 0:
            raise ValueError("fingerprint_size must be > 0")

        self.bucket_size = int(bucket_size)
        self.fingerprint_size = int(fingerprint_size)
        self.max_kicks = int(max_kicks)

        # Number of buckets ≈ capacity / bucket_size, padded a bit
        # to reduce load factor.
        num_buckets = capacity // bucket_size
        if capacity % bucket_size != 0:
            num_buckets += 1

        # For simplicity, enforce at least a small minimum.
        num_buckets = max(num_buckets, 1024)

        self.num_buckets = num_buckets

        # Each bucket is a list of fingerprints (bytes objects or None).
        self.buckets = [[None] * self.bucket_size for _ in range(self.num_buckets)]

        # random source for kicks
        self._rand = random.Random(os.urandom(16))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_bytes(item: Any) -> bytes:
        if isinstance(item, bytes):
            return item
        if isinstance(item, str):
            return item.encode("utf-8", errors="ignore")
        # ints, etc.
        return str(item).encode("utf-8", errors="ignore")

    def _hash_item(self, item: Any) -> int:
        data = self._to_bytes(item)
        h = hashlib.blake2b(data, digest_size=8)  # 64-bit
        return int.from_bytes(h.digest(), byteorder="big", signed=False)

    def _fingerprint(self, h: int) -> bytes:
        # Derive fingerprint bytes from hash.
        fp = h.to_bytes(8, byteorder="big", signed=False)[: self.fingerprint_size]
        # Avoid all-zero fingerprint (would be indistinguishable from empty)
        if all(b == 0 for b in fp):
            # flip last bit
            last = fp[-1] | 0x01
            fp = fp[:-1] + bytes([last])
        return fp

    def _index1(self, h: int) -> int:
        return h % self.num_buckets

    def _index2(self, i1: int, fp: bytes) -> int:
        # Alternate index derived from fingerprint
        h_fp = int.from_bytes(hashlib.blake2b(fp, digest_size=8).digest(), "big")
        return (i1 ^ h_fp) % self.num_buckets

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def contains(self, item: Any) -> bool:
        """
        Check if the item is (probably) in the filter.
        False positives possible, false negatives unlikely.
        """
        h = self._hash_item(item)
        fp = self._fingerprint(h)
        i1 = self._index1(h)
        i2 = self._index2(i1, fp)

        bucket1 = self.buckets[i1]
        bucket2 = self.buckets[i2]

        return fp in bucket1 or fp in bucket2

    def __contains__(self, item: Any) -> bool:
        return self.contains(item)

    def add(self, item: Any) -> bool:
        """
        Insert an item into the filter.
        Returns True if inserted (or already present), False if insertion failed.
        """
        h = self._hash_item(item)
        fp = self._fingerprint(h)
        i1 = self._index1(h)
        i2 = self._index2(i1, fp)

        # If already present, treat as success.
        if fp in self.buckets[i1] or fp in self.buckets[i2]:
            return True

        # Try to place in bucket i1
        bucket1 = self.buckets[i1]
        for idx in range(self.bucket_size):
            if bucket1[idx] is None:
                bucket1[idx] = fp
                return True

        # Try to place in bucket i2
        bucket2 = self.buckets[i2]
        for idx in range(self.bucket_size):
            if bucket2[idx] is None:
                bucket2[idx] = fp
                return True

        # Need to kick out existing fingerprints
        i = i1 if self._rand.random() < 0.5 else i2
        cur_fp = fp

        for _ in range(self.max_kicks):
            bucket = self.buckets[i]
            evict_idx = self._rand.randrange(self.bucket_size)
            # Swap fingerprint
            bucket[evict_idx], cur_fp = cur_fp, bucket[evict_idx]

            # Recompute alternate index for evicted fingerprint
            i = self._index2(i, cur_fp)

            # Try to insert evicted fingerprint into its alternate bucket
            bucket = self.buckets[i]
            for slot in range(self.bucket_size):
                if bucket[slot] is None:
                    bucket[slot] = cur_fp
                    return True

        # If we reach here, insertion failed (filter too full).
        return False

    def insert(self, item: Any) -> bool:
        """
        Alias for add(item) for convenience.
        """
        return self.add(item)

    def remove(self, item: Any) -> bool:
        """
        Best-effort deletion. Returns True if a matching fingerprint was found
        and removed, False otherwise.

        Note: Cuckoo Filters support deletion, but excessive use may impact
        false-negative behaviour. For the typical 'seen set' use case, you
        generally don't need to remove items.
        """
        h = self._hash_item(item)
        fp = self._fingerprint(h)
        i1 = self._index1(h)
        i2 = self._index2(i1, fp)

        bucket1 = self.buckets[i1]
        for idx in range(self.bucket_size):
            if bucket1[idx] == fp:
                bucket1[idx] = None
                return True

        bucket2 = self.buckets[i2]
        for idx in range(self.bucket_size):
            if bucket2[idx] == fp:
                bucket2[idx] = None
                return True

        return False

    def approximate_size(self) -> int:
        """
        Rough estimate of how many items are stored.
        """
        count = 0
        for bucket in self.buckets:
            for fp in bucket:
                if fp is not None:
                    count += 1
        return count
