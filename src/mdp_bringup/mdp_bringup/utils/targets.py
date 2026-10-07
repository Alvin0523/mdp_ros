"""Official MDP Target IDs -> readable names, for logs and Foxglove only (the
tablet gets the bare ID). The same table as mdp_vision's MDP_TARGET_IDS, the
other way round: 11-19 Number 1-9, 20-27 Letter A-H, 28-35 Letter S-Z,
36-39 arrows, 40 Stop, 99 Bullseye."""

NAMES = {
    **{10 + n: f'Number {n}' for n in range(1, 10)},
    **{20 + i: f'Letter {c}' for i, c in enumerate('ABCDEFGH')},
    **{28 + i: f'Letter {c}' for i, c in enumerate('STUVWXYZ')},
    36: 'Arrow Up', 37: 'Arrow Down', 38: 'Arrow Right', 39: 'Arrow Left', 40: 'Stop', 99: 'Bullseye',
}


SYMBOLS = {**{10 + n: str(n) for n in range(1, 10)},
           **{20 + i: c for i, c in enumerate('ABCDEFGH')},
           **{28 + i: c for i, c in enumerate('STUVWXYZ')},
           36: '\u2191', 37: '\u2193', 38: '\u2192', 39: '\u2190', 40: 'Stop', 99: 'Bullseye'}


def short(target_id) -> str:
    """'32' -> 'W / 32', '36' -> '↑ / 36' (the 3D view); unknown ids as they are."""
    try:
        return f'{SYMBOLS[int(target_id)]} / {int(target_id)}'
    except (KeyError, TypeError, ValueError):
        return str(target_id or '')


def label(target_id) -> str:
    """'11' -> 'Number 1 (11)'; an unknown or empty id as it is ('' -> '')."""
    try:
        return f'{NAMES[int(target_id)]} ({int(target_id)})'
    except (KeyError, TypeError, ValueError):
        return str(target_id or '')
