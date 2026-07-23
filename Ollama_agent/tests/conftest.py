# Shared pytest fixtures and path setup.

# sys.path is patched here so every test file can import from the project root
# without installing the package.

import sys
import os
import pytest

# Add the project root (one level up from tests/) to the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# -- SHARED MESSAGE FIXTURES --

@pytest.fixture
def user_message():
    return {"role": "user", "content": "Hello"}

@pytest.fixture
def assistant_message():
    return {"role": "assistant", "content": "Hi there!", "thinking": ""}

@pytest.fixture
def assistant_message_with_thinking():
    return {
        "role": "assistant",
        "content": "The answer is 42.",
        "thinking": "Let me think about this carefully...",
    }

@pytest.fixture
def short_history(user_message, assistant_message):
    return [user_message, assistant_message]

@pytest.fixture
def long_history():
    """50 messages — exceeds MAX_HISTORY_MESSAGES (40)."""
    msgs = []
    for i in range(25):
        msgs.append({"role": "user", "content": f"Message {i}"})
        msgs.append({"role": "assistant", "content": f"Reply {i}", "thinking": ""})
    return msgs

@pytest.fixture
def mock_placeholder():
    """A simple object that records markdown calls, standing in for st.empty()."""
    class FakePlaceholder:
        def __init__(self):
            self.calls: list[str] = []

        def markdown(self, text: str) -> None:
            self.calls.append(text)

        @property
        def last(self) -> str:
            return self.calls[-1] if self.calls else ""

    return FakePlaceholder()