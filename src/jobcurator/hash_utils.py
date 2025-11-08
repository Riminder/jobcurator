from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Literal
import math


@dataclass
class Category:
    """
    Hierarchical category node.

    Example:
      dim = "job_function"
      level_path = ["Engineering", "Software", "Backend"]
    """
    id: str                        # unique id in your taxonomy
    label: str                     # human label, e.g. "Backend"
    level: int                     # 0 = root, 1 = child, ...
    parent_id: Optional[str] = None
    level_path: List[str] = field(default_factory=list)  # from root to this node


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
    - alt_m: altitude in meters above sea level
    - x, y, z: Earth-centered Cartesian coordinates in meters (computed)
    """
    lat: float
    lon: float
    alt_m: float = 0.0
    city: Optional[str] = None
    country_code: Optional[str] = None

    # internal / computed Cartesian coords (Earth-centered)
    x: float = field(default=0.0, init=False)
    y: float = field(default=0.0, init=False)
    z: float = field(default=0.0, init=False)

    def compute_xyz(self, earth_radius_m: float = 6_371_000.0) -> None:
        """
        Convert (lat, lon, alt_m) to 3D Cartesian (x, y, z) in meters.
        Simple spherical Earth model.
        """
        phi = math.radians(self.lat)    # latitude
        lam = math.radians(self.lon)    # longitude
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
    title: str
    text: str
    categories: Dict[str, List[Category]]
    location: Location3DField
    salary: Optional[SalaryField] = None
    company: Optional[str] = None
    contract_type: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[datetime] = None

    # internal/computed fields (not part of external schema)
    length_tokens: int = field(default=0, init=False)
    length_score: float = field(default=0.0, init=False)
    completion_score_val: float = field(default=0.0, init=False)
    quality: float = field(default=0.0, init=False)
    exact_hash: int = field(default=0, init=False)
    signature: int = field(default=0, init=False)
