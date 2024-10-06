from dataclasses import dataclass, field
from typing import List


@dataclass
class ArtistData:
    name: str    
    bio: str
    addresses: List[str] = field(default_factory=list)
    alias: str = None
