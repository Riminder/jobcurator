from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Literal
import math
from .hash_utils import flatten_category_tokens


@dataclass
class Category:
    """
    Hierarchical category node.
    """
    id: str
    label: str
    level: int
    parent_id: Optional[str] = None
    level_path: List[str] = field(default_factory=list)


@dataclass
class SalaryField:
    """
    Structured salary information.
    """
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    currency: str = "EUR"
    period: Literal["year", "month", "day", "hour"] = "year"


@dataclass
class Location3DField:
    """
    Geographical point with 3D coordinates:
    - lat, lon: degrees
    - alt_m: altitude in meters
    - x, y, z: computed Earth-centered Cartesian coordinates (meters)
    """
    lat: float
    lon: float
    alt_m: float = 0.0
    city: Optional[str] = None
    country_code: Optional[str] = None

    x: float = field(default=0.0, init=False)
    y: float = field(default=0.0, init=False)
    z: float = field(default=0.0, init=False)

    def compute_xyz(self, earth_radius_m: float = 6_371_000.0) -> None:
        phi = math.radians(self.lat)
        lam = math.radians(self.lon)
        r = earth_radius_m + self.alt_m

        self.x = r * math.cos(phi) * math.cos(lam)
        self.y = r * math.cos(phi) * math.sin(lam)
        self.z = r * math.sin(phi)


@dataclass
class Job:
    """
    Canonical Job schema.

    categories:
      Dict[dimension_name, List[Category]]
    """
    id: str
    reference: str
    title: str
    text: str
    categories: Dict[str, List[Category]]
    location: Location3DField
    salary: Optional[SalaryField] = None
    company: Optional[str] = None
    contract_type: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[datetime] = None

    # internal/computed fields
    length_tokens: int = field(default=0, init=False)
    length_score: float = field(default=0.0, init=False)
    completion_score_val: float = field(default=0.0, init=False)
    quality: float = field(default=0.0, init=False)
    exact_hash: int = field(default=0, init=False)
    signature: int = field(default=0, init=False)
    diversity_score: float = field(default=0.0, init=False)
    selection_score: float = field(default=0.0, init=False)
    # MinHash fields
    minhash_sig: Optional[List[int]] = field(default=None, repr=False)
    minhash_len: Optional[int] = field(default=None, repr=False)
    # SKlearn fields
    sklearn_hashvector : Optional[List[float]] = field(default=None, repr=False)
    faiss_dim_sig: Optional[int] = field(default=None, repr=False)           # the dim_sig used
    # FAISS fields
    faiss_hashvector : Optional[List[float]] = field(default=None, repr=False)
    # optional fields for selection explanation / clustering
    simhash_bucket: int = field(default=0, init=False)
    lsh_bucket: int = field(default=0, init=False)
    is_selected: bool = field(default=False, init=False)
    selection_reason: Optional[str] = field(default=None, init=False)
    cluster_id: Optional[int] = field(default=None, init=False)
    cluster_size: Optional[int] = field(default=None, init=False)
    cluster_distance: Optional[float] = field(default=None, init=False)
    vector: Optional[List[float]] = field(default=None, init=False)
    # ids
    @property
    def canonical_id(self) -> Optional[str]:
        if self.reference and self.company:
            ref = " ".join(self.reference.split()).lower()
            comp = " ".join(self.company.split()).lower()
            return f"{ref}::{comp}"
        if getattr(self, "id", None):
            return str(self.id)
        return str(id(self))
    
    @property
    def numerical_vector(self) -> List[float]:
        """
        Numeric feature vector for ML-based filters / stats:

        - length_tokens
        - completion_score_val
        - quality
        - avg salary
        - 3D location (x,y,z)
        - category richness (count of flattened tokens)
        """
        if self.location is not None:
            self.location.compute_xyz()
            x = self.location.x
            y = self.location.y
            z = self.location.z
        else:
            x = y = z = 0.0

        sal = 0.0
        if self.salary is not None:
            vals = []
            if self.salary.min_value is not None:
                vals.append(self.salary.min_value)
            if self.salary.max_value is not None:
                vals.append(self.salary.max_value)
            if vals:
                sal = sum(vals) / len(vals)

        cat_tokens = flatten_category_tokens(self)
        cat_count = float(len(cat_tokens))

        return [
            float(self.length_tokens),
            float(self.completion_score_val),
            float(self.quality),
            float(sal),
            float(x),
            float(y),
            float(z),
            cat_count,
        ]
    
    def canonical_hash(self, maxlen: int = 16) -> Optional[str]:
        """
        Return a short printable string for this job's exact_hash if it's set.
        If exact_hash is None, return None (do not compute anything).
        """
        h = self.exact_hash
        if h is None:
            return None

        if isinstance(h, int):
            s = f"{h:x}".rjust(16, "0")  # hex for ints, 0-padded to 16
        elif isinstance(h, (bytes, bytearray)):
            s = h.hex()
        else:
            s = str(h)

        if maxlen and maxlen > 0 and len(s) > maxlen:
            return s[:maxlen] + "…"
        return s

