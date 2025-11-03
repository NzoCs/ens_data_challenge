import re
from typing import Optional
from pydantic import BaseModel

class ParsedProteinChange(BaseModel):
    prefix: str
    ref_aa: Optional[str] = None
    position: Optional[int] = None
    alt_aa: Optional[str] = None
    suffix: Optional[str] = None

class ProteinChangeParser:

    def __init__(self, protein_change: str) -> None:
        self.protein_change = protein_change
    
    def parse(self) -> ParsedProteinChange:
        pc = self.protein_change.strip()
        
        # Try to match standard p. format
        match = re.match(r'^p\.([A-Z]|\?)?(\d+)?([A-Z]|\?|\*|=)?$', pc)
        
        if match:
            ref_aa, pos_str, alt_or_suffix = match.groups()
            position = int(pos_str) if pos_str else None
            
            alt_aa = None
            suffix = None
            if alt_or_suffix:
                if alt_or_suffix in ['*']:
                    suffix = alt_or_suffix
                else:
                    alt_aa = alt_or_suffix
            
            # Special case for 'p.?'
            if ref_aa == '?' and position is None and alt_aa is None:
                ref_aa = None
                alt_aa = '?'
            
            return ParsedProteinChange(
                prefix='p',
                ref_aa=ref_aa,
                position=position,
                alt_aa=alt_aa,
                suffix=suffix
            )
        else:
            # For other formats like 'MLL_PTD'
            return ParsedProteinChange(
                prefix=pc,
                ref_aa=None,
                position=None,
                alt_aa=None,
                suffix=None
            )

    