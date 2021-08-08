import enum

from homework.model.efficient_net.efficient_net import EfficientNet


class EfficientNetType(enum.Enum):
    B0 = 1
    B1 = 2
    B2 = 3
    B3 = 4
    B4 = 5
    B5 = 6
    B6 = 7
    B7 = 8


class EfficientNetFactory:
    @staticmethod
    def get_efficient_net(efficient_net_type: EfficientNetType,
                          output_size: int):
        w_factor = None
        d_factor = None
        if efficient_net_type is EfficientNetType.B0:
            w_factor = 1
            d_factor = 1
        elif efficient_net_type is EfficientNetType.B1:
            w_factor = 1
            d_factor = 1.1
        elif efficient_net_type is EfficientNetType.B2:
            w_factor = 1.1
            d_factor = 1.2
        elif efficient_net_type is EfficientNetType.B3:
            w_factor = 1.2
            d_factor = 1.4
        elif efficient_net_type is EfficientNetType.B4:
            w_factor = 1.4
            d_factor = 1.8
        elif efficient_net_type is EfficientNetType.B5:
            w_factor = 1.6
            d_factor = 2.2
        elif efficient_net_type is EfficientNetType.B6:
            w_factor = 1.8
            d_factor = 2.6
        elif efficient_net_type is EfficientNetType.B7:
            w_factor = 2
            d_factor = 3.1
        return EfficientNet(w_factor=w_factor,
                            d_factor=d_factor,
                            output_size=output_size)
