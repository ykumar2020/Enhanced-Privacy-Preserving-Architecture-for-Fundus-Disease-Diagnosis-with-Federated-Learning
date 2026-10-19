from __future__ import annotations

import torch

from retina_fl.federated import LocalUpdate, aggregate_sample_weighted, select_clients, train_epoch


def _update(client: str, examples: int, value: float) -> LocalUpdate:
    return LocalUpdate(client, examples, 0.0, 0.0, 0.0, {"weight": torch.tensor([value])})


def test_fedavg_uses_only_selected_client_sample_counts():
    state, weights = aggregate_sample_weighted([
        _update("client_00", 10, 1.0),
        _update("client_03", 30, 5.0),
    ])
    assert weights == {"client_00": 0.25, "client_03": 0.75}
    assert torch.allclose(state["weight"], torch.tensor([4.0]))
    assert sum(weights.values()) == 1.0


def test_client_selection_is_reproducible_and_partial():
    clients = [f"client_{index:02d}" for index in range(4)]
    first = select_clients(clients, 2, seed=2025, round_index=1)
    second = select_clients(clients, 2, seed=2025, round_index=1)
    assert first == second
    assert len(first) == 2


def test_multilabel_epoch_counts_images_not_label_elements():
    model = torch.nn.Linear(2, 3)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    loader = [{"image": torch.zeros(2, 2), "target": torch.zeros(2, 3)}]
    metrics = train_epoch(model, loader, optimizer, torch.device("cpu"), "multilabel")
    assert metrics["examples"] == 2
    assert metrics["accuracy"] == 1.0

