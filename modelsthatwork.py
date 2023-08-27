model = nn.Sequential(
    nn.Conv1d(1, 16, 4, stride=2),
    nn.LeakyReLU(),
    nn.Conv1d(16, 32, 4, stride=2),
    nn.LeakyReLU(),
    nn.Conv1d(32, 64, 4, stride=2),
    nn.LeakyReLU(),
    nn.Conv1d(64, 32, 4, stride=2),
    nn.LeakyReLU(),
    nn.Conv1d(32, 16, 4, stride=2),
    nn.Flatten(),
    nn.Linear(32, 9)
)