import torch

def od_collate_fn(batch):
    """
    fetch (n, 5)
    n: the number of objects
    """

    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0]) # sample[0] is img
        targets.append(torch.FloatTensor(sample[1])) # sample[1] is annotation
    #sample[0]:[C][H][W], imgs[img, img, img, ...]
    #[torch.Size([3, 300, 300]), torch.Size([3, 300, 300]), ...] => torch.Size([batch_num, 3, 300, 300])
    imgs = torch.stack(imgs, dim=0)

    #targes:[n, 5], n:the number of objects
    #[xmin, ymin, xmax, ymax, class_idex]
    return imgs, targets

