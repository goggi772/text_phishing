import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def make_loader(dataset,batch_size=16, seed=42, istest=False):
    generator = torch.Generator()
    generator.manual_seed(seed)
    if istest:
        return DataLoader(dataset, 
                      sampler=SequentialSampler(dataset),
                      collate_fn = lambda s: dataset.collate_fn(s),
                      batch_size=batch_size,
                      shuffle=False
                      )
    else:
        return DataLoader(dataset, 
                        sampler=RandomSampler(dataset, generator=generator),
                        collate_fn = lambda s: dataset.collate_fn(s),
                        batch_size=batch_size,
                         )