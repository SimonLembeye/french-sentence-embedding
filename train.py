import torch
from torch import nn

from models.siamese_camembert import SiameseCamemBERT
from trainer import SiameseTrainer

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    saved_model_path = None

    if saved_model_path:
        model = torch.load("saved_models/3_0_siamese.pth").to(device)
    else:
        model = SiameseCamemBERT("camembert-base").to(device)

    loss = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    trainer = SiameseTrainer(device, model, loss, optimizer, train_log_frequency=100)
    trainer.train()
