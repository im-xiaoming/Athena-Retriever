"""Interface every feature encoder implements, so the pipeline can swap ONE-PEACE for InternVideo2."""


class FeatureEncoder:
    name = 'base'
    dim_visual = dim_audio = 0
    # time grid: one feature every feat_stride frames at fps, each covering num_frames frames
    feat_stride, num_frames, fps = 1, 1, 1

    def encode(self, path):
        """Return {'visual': (T, dim_visual), 'audio': (T, dim_audio), 'duration': seconds, 'has_audio': bool}."""
        raise NotImplementedError
