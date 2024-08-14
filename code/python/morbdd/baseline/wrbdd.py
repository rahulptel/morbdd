from morbdd.baseline.baseline import Baseline


class WidthRestrictedBDD(Baseline):
    def __init__(self, cfg):
        super().__init__(cfg)

    def node_select_random(self):
        pass

    def select_nodes(self):
        pass


class KnapsackWidthRestrictedBDD(WidthRestrictedBDD):
    def __init__(self, cfg):
        super().__init__(cfg)

    def select_nodes(self):
        if self.cfg.node_selection == 'random':
            self.node_select_random()


class IndepsetWidthRestrictedBDD(WidthRestrictedBDD):
    def __init__(self, cfg):
        super().__init__(cfg)

    def select_nodes(self):
        if self.cfg.node_selection == 'random':
            self.node_select_random()
