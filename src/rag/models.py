from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path


class ImageCategory(Enum):
    VALID_METER = "valid_meter"
    NOT_A_METER = "not_a_meter"


@dataclass
class ImageMetadata:
    category: ImageCategory
    confidence_score: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    

@dataclass
class ReferenceImage:
    id: str
    file_path: Path
    metadata: ImageMetadata
    base64_data: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'file_path': str(self.file_path),
            'metadata': {
                'category': self.metadata.category.value,
                'confidence_score': self.metadata.confidence_score,
                'created_at': self.metadata.created_at.isoformat()
            },
            'base64_data': self.base64_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReferenceImage':
        metadata = ImageMetadata(
            category=ImageCategory(data['metadata']['category']),
            confidence_score=data['metadata'].get('confidence_score', 1.0),
            created_at=datetime.fromisoformat(data['metadata']['created_at'])
        )
        return cls(
            id=data['id'],
            file_path=Path(data['file_path']),
            metadata=metadata,
            base64_data=data.get('base64_data')
        )