"""Contract-month codes and the spread / butterfly notation.

    CL[J][K0]        long 1 CL April, short 1 CL May, same contract year
    GC[X][Z1]        long 1 GC November, short 1 GC December of the NEXT year
    NG[J][X0][Z0]    long 1 April, short 2 November, long 1 December

The trailing digit on a leg is the YEAR OFFSET.  Offset 0 means the first
occurrence of that month strictly after the PREVIOUS leg; offset n adds n
further years.  Resolving against the previous leg rather than the front is
what keeps a butterfly monotonic: NG[F][Z0][H0] is Jan Y, Dec Y, Mar Y+1,
whereas resolving every leg against the front would put the last leg in Mar Y
and produce legs that are out of order.  For a two-leg spread the previous leg
IS the front, so this reduces to the plain rule.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MONTH_CODE_TO_NUM = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                     'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
NUM_TO_MONTH_CODE = {v: k for k, v in MONTH_CODE_TO_NUM.items()}
MONTH_CODES = ''.join(MONTH_CODE_TO_NUM)

MAX_YEAR_OFFSET = 2          # the study is restricted to offsets 0, 1, 2

_COMBO_RE = re.compile(
    r'^(?P<ticker>[A-Za-z0-9._]+)'
    r'(?P<legs>(?:\[[' + MONTH_CODES + r'][0-9]?\]){2,3})$')
_LEG_RE = re.compile(r'\[([' + MONTH_CODES + r'])([0-9]?)\]')


@dataclass(frozen=True)
class Leg:
    month: str      # single-letter contract month code
    offset: int     # year offset relative to the previous leg

    @property
    def month_num(self) -> int:
        return MONTH_CODE_TO_NUM[self.month]


@dataclass(frozen=True)
class ComboSpec:
    """A spread or butterfly, independent of any particular year."""
    ticker: str
    legs: tuple

    def __post_init__(self):
        if len(self.legs) not in (2, 3):
            raise ValueError('a combo has 2 legs (spread) or 3 (butterfly), '
                             f'got {len(self.legs)}')
        if self.legs[0].offset != 0:
            raise ValueError('the front leg anchors the combo; its offset '
                             'must be 0')
        for leg in self.legs[1:]:
            if not 0 <= leg.offset <= MAX_YEAR_OFFSET:
                raise ValueError(f'year offset {leg.offset} outside '
                                 f'0..{MAX_YEAR_OFFSET}')

    @property
    def kind(self) -> str:
        return 'spread' if len(self.legs) == 2 else 'butterfly'

    @property
    def weights(self) -> tuple:
        return (1, -1) if len(self.legs) == 2 else (1, -2, 1)

    @property
    def code(self) -> str:
        body = ''.join(f'[{lg.month}{lg.offset}]' for lg in self.legs[1:])
        return f'{self.ticker}[{self.legs[0].month}]{body}'

    @property
    def safe_code(self) -> str:
        """Filesystem-safe form of `code`: CL[J][K0] -> CL_J_K0."""
        return (self.code.replace('][', '_').replace('[', '_')
                          .replace(']', ''))

    def __str__(self) -> str:
        return self.code


def parse_combo(text: str) -> ComboSpec:
    """'CL[J][K0]' or 'NG[J][X0][Z0]' -> ComboSpec."""
    m = _COMBO_RE.match(text.strip())
    if not m:
        raise ValueError(f'not a combo code: {text!r}')
    legs = tuple(Leg(mo, int(off) if off else 0)
                 for mo, off in _LEG_RE.findall(m.group('legs')))
    return ComboSpec(m.group('ticker').upper(), legs)


def parse_safe_code(name: str) -> ComboSpec:
    """Filename form back to a spec: 'GC_X_Z1' -> GC[X][Z1]."""
    parts = name.split('_')
    if len(parts) < 3:
        raise ValueError(f'not a combo filename: {name!r}')
    ticker, legs = parts[0], parts[1:]
    code = ticker + ''.join(f'[{lg}]' for lg in legs)
    return parse_combo(code)


def resolve_leg_years(front_year: int, legs) -> list:
    """Contract year of every leg, given the front leg's contract year.

    Each leg lands on the first occurrence of its month strictly after the
    previous leg, plus its own year offset.
    """
    years = [front_year]
    for prev, leg in zip(legs, legs[1:]):
        roll = 0 if leg.month_num > prev.month_num else 1
        years.append(years[-1] + roll + leg.offset)
    return years
