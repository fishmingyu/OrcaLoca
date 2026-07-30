from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from Orcar.search_agent import SearchAgent, SearchWorker


@pytest.fixture()
def llm():
    return SimpleNamespace(callback_manager=None)


@patch("Orcar.search_agent.AgentRunner.__init__", return_value=None)
@patch.object(SearchWorker, "from_tools", return_value=object())
@patch.object(SearchAgent, "_setup_tools", return_value=[])
def test_search_agent_accepts_injected_manager(
    _setup_tools,
    _from_tools,
    _runner_init,
    llm,
):
    manager = object()

    agent = SearchAgent(llm=llm, repo_path="/repo", search_manager=manager)

    assert agent._search_manager is manager
    _from_tools.assert_called_once()


@patch("Orcar.search_agent.AgentRunner.__init__", return_value=None)
@patch.object(SearchWorker, "from_tools", return_value=object())
@patch.object(SearchAgent, "_setup_tools", return_value=[])
def test_search_agent_accepts_manager_factory(
    _setup_tools,
    _from_tools,
    _runner_init,
    llm,
):
    manager = object()
    factory = Mock(return_value=manager)

    agent = SearchAgent(
        llm=llm,
        repo_path="/repo",
        search_manager_factory=factory,
    )

    assert agent._search_manager is manager
    factory.assert_called_once_with("/repo")


def test_search_agent_rejects_manager_and_factory(llm):
    with pytest.raises(
        ValueError,
        match="both search_manager and search_manager_factory",
    ):
        SearchAgent(
            llm=llm,
            search_manager=object(),
            search_manager_factory=Mock(),
        )
