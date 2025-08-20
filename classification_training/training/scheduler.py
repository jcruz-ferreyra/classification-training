import torch.optim.lr_scheduler as lr_scheduler
import logging

logger = logging.getLogger(__name__)


def create_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Create learning rate scheduler with warmup support."""

    logger.info("Creating cosine scheduler")
    logger.info(f"Warmup epochs: {warmup_epochs}, Total epochs: {total_epochs}")

    if warmup_epochs > 0:
        # Cosine annealing with linear warmup
        main_epochs = total_epochs - warmup_epochs

        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,  # Start at 10% of base LR
            end_factor=1.0,  # Ramp up to full LR
            total_iters=warmup_epochs,
        )

        cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_epochs)

        scheduler = lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )

        logger.debug(
            f"Created sequential scheduler: {warmup_epochs} warmup + {main_epochs} cosine epochs"
        )

    else:
        # Simple cosine annealing without warmup
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        logger.debug("Created simple cosine annealing scheduler")

    return scheduler
