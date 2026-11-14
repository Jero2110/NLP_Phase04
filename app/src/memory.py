from collections import deque
from dataclasses import dataclass


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str


class ShortTermMemory:
    """Rolling window of recent exchanges, limited by turns and by an approximate character budget.
    Oldest exchanges are dropped first, always in user/assistant pairs."""

    def __init__(self, max_turns: int = 6, max_chars: int = 6000):
        self.max_turns = max_turns
        self.max_chars = max_chars
        self._turns: deque[Turn] = deque()

    def add_exchange(self, user_text: str, assistant_text: str) -> None:
        self._turns.append(Turn("user", user_text))
        self._turns.append(Turn("assistant", assistant_text))
        self._trim()

    def _trim(self) -> None:
        while len(self._turns) > 2 * self.max_turns:
            self._pop_pair()
        while self.total_chars() > self.max_chars and len(self._turns) > 2:
            self._pop_pair()

    def _pop_pair(self) -> None:
        self._turns.popleft()
        self._turns.popleft()

    def total_chars(self) -> int:
        return sum(len(t.content) for t in self._turns)

    def as_messages(self) -> list[dict]:
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def clear(self) -> None:
        self._turns.clear()