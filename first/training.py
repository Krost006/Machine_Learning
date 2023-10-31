from imports import *
from model import *
from dataset_prepare import *

d = Data()
m = Model()

def train():

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
    )

    trainer.fit(m,d)        
