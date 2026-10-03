"""Name -> class registry for backbones (copied from libs/modeling/models.py)."""
BACKBONES = {}


def register_backbone(name):
    def decorator(cls):
        BACKBONES[name] = cls
        return cls
    return decorator


def make_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
