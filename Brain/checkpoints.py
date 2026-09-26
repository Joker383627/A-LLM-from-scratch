import torch
import torch.distributed as dist



def save_checkpoint(path, model, optimizer, scaler, epoch, batch_index,
                    best_valid_loss):
    if dist.get_rank() == 0:
        torch.save({
            "epoch": epoch,
            "batch_index": batch_index,
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_valid_loss": best_valid_loss
        }, path)
        print(f"Checkpoint saved at batch {batch_index}", flush=True)


def load_checkpoint(path, model, optimizer, scaler, device):

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    model.module.load_state_dict(checkpoint["model"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    scaler.load_state_dict(checkpoint["scaler"])

    return (
        checkpoint["epoch"],
        checkpoint["batch_index"],
        checkpoint["best_valid_loss"]
    )