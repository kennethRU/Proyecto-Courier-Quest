# sorting.py
from typing import Any, Callable, List


def merge_sort(items: List[Any], key: Callable[[Any], Any], reverse: bool = False) -> List[Any]:
    n = len(items)
    if n <= 1:
        return items[:]

    mid = n // 2
    left = merge_sort(items[:mid], key, reverse)
    right = merge_sort(items[mid:], key, reverse)

    return _merge(left, right, key, reverse)


def _merge(left: List[Any], right: List[Any], key, reverse: bool) -> List[Any]:
    i = j = 0
    out: List[Any] = []
    cmp = (lambda a, b: key(a) >= key(b)) if reverse else (lambda a, b: key(a) <= key(b))
    while i < len(left) and j < len(right):
        if cmp(left[i], right[j]):
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out
