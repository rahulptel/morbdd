import sys


def selector_factory(cfg):
    if cfg.selector.type == "width":
        from .selector import WidthBasedNodeSelector
        return WidthBasedNodeSelector(cfg)
    elif cfg.selector.type == "threshold":
        from .selector import ThresholdBasedNodeSelector
        return ThresholdBasedNodeSelector(cfg)
    else:
        print("Invalid node selector!")
        sys.exit()
