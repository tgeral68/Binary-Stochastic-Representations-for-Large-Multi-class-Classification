
def hamming(x, B):
    return (x.unsqueeze(0).expand_as(B) - B).abs().sum(1)