import os

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


from Brain.transformer import Transformer
from Brain.preprocess import FastBinaryTextDataSet
from Brain.train import train_one_epoch,evaluate
from Brain.checkpoints import load_checkpoint,save_checkpoint

def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device("cuda", local_rank)
    )
    return local_rank

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():

    VOCAB_SIZE = 50257
    CONTEXT_LENGTH = 768
    STRIDE = 768
    BATCH_SIZE = 8
    EMBEDDED_DIM = 768
    NUM_LAYERS = 8
    NUM_HEADS = 8
    DROPOUT = 0.1
    EPOCHS = 1
    ACCUMULATION_STEPS = 4
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.1

    CHECKPOINT_EVERY = 20000

    TRAIN_PATH = "/kaggle/input/datasets/tuhinbakuli/tiktoken-fineweb-1b/train.bin"
    VALID_PATH = "/kaggle/input/datasets/tuhinbakuli/tiktoken-fineweb-1b/valid.bin"
    CHECKPOINT_PATH = "/kaggle/input/datasets/tuhinbakuli/model-param-checkpt/latest_checkpoint_2.pt"
    BEST_PATH = "/kaggle/working/best_model_checkpoint.pt"
    SAVE_PATH = "/kaggle/working/latest_checkpoint.pt"

    local_rank = setup()
    device = torch.device("cuda", local_rank)

    train_dataset = FastBinaryTextDataSet(
        TRAIN_PATH, CONTEXT_LENGTH, STRIDE
    )
    valid_dataset = FastBinaryTextDataSet(
        VALID_PATH, CONTEXT_LENGTH, STRIDE
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        sampler=valid_sampler,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2
    )

    if local_rank == 0:
        print(f"Training samples: {len(train_dataset):,}")
        print(f"Batches per GPU: {len(train_loader):,}")
        print(f"Global microbatch: {BATCH_SIZE * 2:,}")
        print(
            f"Effective global batch: "
            f"{BATCH_SIZE * 2 * ACCUMULATION_STEPS:,}"
        )

    model = Transformer(
        emb_dim=EMBEDDED_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        num_layers=NUM_LAYERS,
        is_causal=True,
        context_length=CONTEXT_LENGTH,
        vocab_size=VOCAB_SIZE
    ).to(device)

    model = DDP(
        model,
        device_ids=[local_rank],
        gradient_as_bucket_view=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    model.compile()

    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    start_batch = 0
    best_valid_loss = float("inf")

    if os.path.exists(CHECKPOINT_PATH):
        start_epoch, start_batch, best_valid_loss = load_checkpoint(
            CHECKPOINT_PATH,
            model,
            optimizer,
            scaler,
            device
        )

        if local_rank == 0:
            print(
                f"Resuming: epoch={start_epoch}, "
                f"batch={start_batch}, "
                f"best_loss={best_valid_loss:.4f}"
            )

    try:
        for epoch in range(start_epoch, EPOCHS):

            train_sampler.set_epoch(epoch)

            if epoch != start_epoch:
                start_batch = 0

            avg_train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                scaler,
                ACCUMULATION_STEPS,
                start_batch,
                CHECKPOINT_EVERY,
                SAVE_PATH,
                epoch,
                best_valid_loss
            )

            avg_valid_loss = evaluate(
                model,
                valid_loader,
                device
            )

            if local_rank == 0:
                print(
                    f"\nEpoch {epoch + 1} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Valid Loss: {avg_valid_loss:.4f}"
                )

            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss

                save_checkpoint(
                    BEST_PATH,
                    model,
                    optimizer,
                    scaler,
                    epoch + 1,
                    0,
                    best_valid_loss
                )

            save_checkpoint(
                SAVE_PATH,
                model,
                optimizer,
                scaler,
                epoch + 1,
                0,
                best_valid_loss
            )

    finally:
        cleanup()


if __name__ == "__main__":
    main()