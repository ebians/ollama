"""シンプルなインメモリ レート制限"""
import time
from collections import defaultdict

from app.config import RATE_LIMIT_PER_MINUTE

# {user_id: [timestamp, ...]}
_requests: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(user_id: str) -> bool:
    """レート制限内ならTrue、超過ならFalse"""
    now = time.time()
    window = now - 60
    # 古いエントリを除去
    _requests[user_id] = [t for t in _requests[user_id] if t > window]
    if len(_requests[user_id]) >= RATE_LIMIT_PER_MINUTE:
        return False
    _requests[user_id].append(now)
    return True


def reset_rate_limit():
    """テスト用: リセット"""
    _requests.clear()
