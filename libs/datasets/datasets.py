import torch
from .data_utils import trivial_batch_collator, worker_init_reset_seed

datasets = {}
def register_dataset(name):
   def decorator(cls):
       datasets[name] = cls
       return cls
   return decorator

# bộ sinh vị trí (điểm, đoạn, ...)
generators = {}
def register_generator(name):
    def decorator(cls):
        generators[name] = cls
        return cls
    return decorator

def make_dataset(name, is_training, split, **kwargs):
   """
       Tạo dataset từ tên đã đăng ký
   """
   dataset = datasets[name](is_training, split, **kwargs)
   return dataset

def make_data_loader(dataset, is_training, generator, batch_size, num_workers, sampler):
    """
        Tạo DataLoader. Lúc train bỏ batch cuối nếu thiếu, và đặt lại seed cho từng worker
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        drop_last=is_training,
        generator=generator,
        persistent_workers=True
    )
    return loader

def make_generator(name, **kwargs):
    generator = generators[name](**kwargs)
    return generator
