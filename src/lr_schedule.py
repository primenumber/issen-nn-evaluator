def cosine_with_warmup(epoch):
    return 0.95 ** epoch * min(1.0, (epoch + 1) * 0.2)
