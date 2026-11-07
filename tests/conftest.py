import pytest

from budget_agent.workspace import Workspace


@pytest.fixture(scope="session")
def demo_ws(tmp_path_factory):
    ws = Workspace(tmp_path_factory.mktemp("ws"), mmm_maxiter=25)
    ws.init_demo(seed=7)
    ws.fit_model()
    return ws
