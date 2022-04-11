import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2,hidden_dim3,hidden_dim4,hidden_dim5):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim1, hidden_dim1),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim1, hidden_dim1),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim2, hidden_dim3),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim3, hidden_dim3),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim3, hidden_dim3),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim3, hidden_dim4),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim4, hidden_dim4),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim4, hidden_dim4),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim4, hidden_dim5),
            nn.LeakyReLU(),

        )

        self.decoder = nn.Sequential(

            nn.Linear(hidden_dim5, hidden_dim4),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim4, hidden_dim4),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim4, hidden_dim4),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim4, hidden_dim3),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim3, hidden_dim3),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim3, hidden_dim3),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim3, hidden_dim2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim2, hidden_dim2),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim2, hidden_dim1),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim1, hidden_dim1),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim1, hidden_dim1),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim1, input_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        out = x.view(x.size(0),-1)
        out = self.encoder(out)
        out = self.decoder(out)
        out = out.view(x.size())


        return out

    def get_codes(self, x):
        return self.encoder(x)