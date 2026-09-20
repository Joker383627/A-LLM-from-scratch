import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from Brain.preprocess import TextDataset
from Brain.transformer import Transformer

from config import VOCAB_SIZE,LEARNING_RATE as LR,CONTEXT_LENGTH,BATCH_SIZE


def train(model:nn.Module,
          epochs:int,
          train_loader:DataLoader,
          valid_loader:DataLoader,
          optimizer:torch.optim,
          device = "cpu"):

    model.train()

    for epoch in range(epochs):
        avg_train_loss = train_one_epoch(model,train_loader,optimizer,device)
        avg_validation_loss = evaluate(model,valid_loader,device)
        
        print( f"\nEpoch {epoch + 1} complete | " 
              f"train loss: {avg_train_loss:.4f} | " 
              f"valid_loss: {avg_validation_loss:.4f}\n" )

        return avg_train_loss

@torch.no_grad()
def evaluate(model:nn.Module,dataloader:DataLoader,device = "cpu"):

    model.eval()

    total_loss = 0.0
    for x,y in dataloader:
        x = x.to(device)         
        y = y.to(device)

        logits = model(x)[0]

        loss = F.cross_entropy(logits.reshape(-1,VOCAB_SIZE),y.reshape(-1))

        total_loss += loss.item()

    return total_loss/len(dataloader)

def train_one_epoch(model:nn.Module,dataloader:DataLoader,optimizer: torch.optim,device = "cpu"):
    model.train()
    total_loss = 0.0
    for batch_index,(x,y) in enumerate(dataloader):
       x = x.to(device)
       y = y.to(device)

       optimizer.zero_grad()

       logits = model(x)[0]

       loss = F.cross_entropy(logits.reshape(-1,VOCAB_SIZE),y.reshape(-1))

       loss.backward()
       optimizer.step()

       total_loss += loss.item()
       if batch_index % 100 == 0: 
           print(f"Batch {batch_index} | " f"Loss {loss.item():.4f}" )
    avg_loss = total_loss/len(dataloader)

    return avg_loss

def main():
    TRAIN_PATH = Path(__file__).resolve().parent.parent /"data" /"WikipediaCorpus"/ "train_corpus.txt"
    VALID_PATH = Path(__file__).resolve().parent.parent /"data" /"WikipediaCorpus"/ "valid_corpus.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = TextDataset(
        path = TRAIN_PATH,
        context_length=CONTEXT_LENGTH,
        stride = 128)

    validation_dataset = TextDataset(
        path = VALID_PATH,
        context_length=CONTEXT_LENGTH,
        stride = 128)


    train_loader = DataLoader(train_dataset,BATCH_SIZE,True)
    valid_loader = DataLoader(validation_dataset,BATCH_SIZE,False)

    model = Transformer(
        emb_dim=384,
        num_heads=8,
        dropout=0.2,
        num_layers=6
    ).to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr = LR
        )

    train(
        model = model,
        epochs = 4,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        device = device
        )

    torch.save(
        model.state_dict(),
        "model_checkpoint.pt"
    )

    print("\nModel Saved")

if __name__ == "__main__":
    main()


