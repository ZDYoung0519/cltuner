for n, p in self.named_parameters():
    if p.requires_grad_:
        print(n, p.shape)