"""Three-filing GRU with filing-age and padding indicators."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .experiment import NUMERIC


def _train_one(train, validation, test, history_panel, seed, epochs=80):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    history = {c: g.sort_values('known_at') for c, g in history_panel.groupby('cik')}
    fill = SimpleImputer(strategy='median', keep_empty_features=True).fit(train[NUMERIC])
    scale = StandardScaler().fit(fill.transform(train[NUMERIC]))
    index = {}
    for _, row in pd.concat([train, validation, test]).iterrows():
        available = history[row.cik]
        available = available[available.known_at <= row.known_at].tail(3)
        values = scale.transform(fill.transform(available[NUMERIC])).astype('float32')
        ages = (pd.to_datetime(row.known_at) - pd.to_datetime(available.known_at)).dt.days.to_numpy() / 365.25
        seq = np.zeros((3, len(NUMERIC) + 2), dtype='float32')
        seq[-len(values):, :len(NUMERIC)] = values
        seq[-len(values):, -2] = ages
        seq[-len(values):, -1] = 1.0
        index[row.id] = seq

    def tensor(rows):
        return (torch.tensor(np.stack([index[key] for key in rows.id])),
                torch.tensor((rows.target_capex_rate - rows.capex_rate).to_numpy(dtype='float32')))

    x, y = tensor(train)
    xv, yv = tensor(validation)
    xt, _ = tensor(test)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(len(NUMERIC) + 2, 16, batch_first=True)
            self.head = nn.Sequential(nn.Linear(16, 12), nn.ReLU(), nn.Linear(12, 1))

        def forward(self, z):
            return self.head(self.rnn(z)[1][-1]).squeeze(-1)

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.01)
    best, best_epoch, state, patience = float('inf'), -1, None, 0
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        net.train()
        for ids in np.array_split(rng.permutation(len(train)), max(1, int(np.ceil(len(train)/64)))):
            opt.zero_grad()
            loss = ((net(x[ids]) - y[ids])**2).mean()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            error = float((net(xv) - yv).abs().mean())
        if error < best - 1e-6:
            best, best_epoch, state, patience = error, epoch, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            patience += 1
        if patience >= 12:
            break
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        pv = validation.capex_rate.to_numpy() + net(xv).numpy()
        pt = test.capex_rate.to_numpy() + net(xt).numpy()
    return pv, pt, {'seed': seed, 'early_stop_epoch': best_epoch, 'max_epochs': epochs,
                    'hidden': 16, 'max_history_filings': 3}


def train_gru(train, validation, test, history_panel):
    """Equal average over three fixed seeds; no test-based seed selection."""
    runs = [_train_one(train, validation, test, history_panel, seed) for seed in (17, 41, 73)]
    return np.mean([r[0] for r in runs], axis=0), np.mean([r[1] for r in runs], axis=0), {
        'ensemble': 'equal mean', 'runs': [r[2] for r in runs]}
