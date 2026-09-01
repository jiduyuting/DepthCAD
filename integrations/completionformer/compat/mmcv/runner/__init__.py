import torch


def load_checkpoint(model, filename, map_location="cpu", strict=False, logger=None):
    checkpoint = torch.load(filename, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_state = model.state_dict()
    cleaned = {}
    skipped = []
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key in model_state and model_state[clean_key].shape != value.shape:
            skipped.append(clean_key)
            continue
        cleaned[clean_key] = value
    incompatible = model.load_state_dict(cleaned, strict=strict)
    if logger is not None:
        logger.info(
            "Loaded checkpoint %s (missing=%d, unexpected=%d)",
            filename,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
        if skipped:
            logger.info("Skipped %d shape-mismatched tensors", len(skipped))
    return checkpoint
