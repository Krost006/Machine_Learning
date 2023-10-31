from imports import *
from model import *
from dataset_prepare import *
def train:
    d = Data()
    m = Model()

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
    )

    trainer.fit(m,d)        
