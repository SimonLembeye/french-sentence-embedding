def get_acc(out, targets):
    s = 0
    for i in range(targets.size()[0]):
        _, argmax = out[i].max(0)
        if argmax == targets[i]:
            s += 1
    return s