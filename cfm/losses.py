# cfm/losses.py
def cfm_loss(model, x_t, t, target_v):
    """
    Simple MSE between predicted and target velocity.
    """
    pred_v = model(x_t, t)     # (B, x_dim)
    return ((pred_v - target_v) ** 2).sum(dim=-1).mean()