class N2PConfig:
    def __init__(self, subpatch_shape, random = False, just_color=False, color_sigma=None, pos_sigma=None) -> None:
        self.subpatch_shape = subpatch_shape
        self.random = random
        self.just_color = just_color
        self.color_sigma = color_sigma
        self.pos_sigma = pos_sigma