from morbdd.deployer.deployer import Deployer


class LayerStitcher:
    def __init__(self, strategy="select_all"):
        self.strategy = strategy

    def __call__(self, scores):
        if self.strategy == "select_all":
            return []


def get_instance_data():
    pass


def get_sparsifier():
    pass


def preprocess_data():
    pass


@hydra.main(config_path="../configs", config_name="05_deploy.yaml", version_base="1.2")
def main(cfg):
    # Load instance data
    data = get_instance_data(cfg.prob.name, cfg.size, cfg.deploy.split, cfg.deploy.pid)
    node_selector = LayerNodeSelector(cfg.deploy.node_select.strategy,
                                      width=cfg.deploy.node_select.width,
                                      threshold=cfg.deploy.node_select.threshold)
    layer_stitcher = LayerStitcher()

    # Load model
    helper = MISTrainingHelper(cfg)
    model_path = helper.get_checkpoint_path() / "best_model.pt"
    model = load_model(cfg, model_path)
    model.eval()

    start = time.time()
    # Preprocess data for ML model
    obj, adj, pos = preprocess_data(data, top_k=cfg.top_k)
    data_preprocess_time = time.time() - start

    # Obtain variable and instance embedding
    # Same for all the nodes of the BDD
    start = time.time()
    v_emb = get_var_embedding(model, obj, adj, pos)
    node_emb_time = time.time() - start

    start = time.time()
    inst_emb = get_inst_embedding(model, v_emb)
    inst_emb_time = time.time() - start

    # Set BDD Manager
    start = time.time()
    order = []
    env = get_env(cfg.prob.n_objs)
    env.reset(cfg.problem_type,
              cfg.preprocess,
              cfg.method,
              cfg.maximization,
              cfg.dominance,
              cfg.bdd_type,
              cfg.maxwidth,
              order)
    env.set_inst(cfg.prob.n_vars, data["n_cons"], cfg.prob.n_objs, data["obj_coeffs"], data["cons_coeffs"],
                 data["rhs"])
    # Initializes BDD with the root node
    env.initialize_dd_constructor()
    # Set the variable used to generate the next layer
    env.set_var_layer(-1)
    lid = 0

    # Restrict and build
    while lid < data["n_vars"] - 1:
        env.generate_next_layer()
        env.set_var_layer(-1)

        layer = env.get_layer(lid + 1)
        states = get_state_tensor(layer, cfg.prob.n_vars)
        vid = torch.tensor(env.get_var_layer()[lid + 1]).int()

        scores = get_node_scores(model, v_emb, inst_emb, lid, vid, states, threshold=0.5)
        lid += 1

        if lid >= cfg.deploy.node_select.prune_from_lid:
            selection, selected_idx, removed_idx = node_selector(lid, scores)
            # Stitch in case of a disconnected BDD
            if len(removed_idx) == len(scores):
                removed_idx = layer_stitcher(scores)
                print("Disconnected at layer: ", {lid})
            # Restrict if necessary
            if len(removed_idx):
                env.approximate_layer(lid, RESTRICT, 1, removed_idx)

    # Generate terminal layer
    env.generate_next_layer()
    build_time = time.time() - start

    start = time.time()
    # Compute pareto frontier
    env.compute_pareto_frontier()
    pareto_time = time.time() - start

    orig_dd = load_orig_dd(cfg)
    restricted_dd = env.get_dd()
    orig_size = compute_dd_size(orig_dd)
    rest_size = compute_dd_size(restricted_dd)
    size_ratio = rest_size / orig_size

    true_pf = load_pf(cfg)
    try:
        pred_pf = env.get_frontier()["z"]
    except:
        pred_pf = None

    cardinality_raw, cardinality = -1, -1
    if true_pf is not None and pred_pf is not None:
        cardinality_raw = compute_cardinality(true_pf=true_pf, pred_pf=pred_pf)
        cardinality = cardinality_raw / len(true_pf)

    run_path = get_run_path(cfg, helper.get_checkpoint_path().stem)
    run_path.mkdir(parents=True, exist_ok=True)

    total_time = data_preprocess_time + node_emb_time + inst_emb_time + build_time + pareto_time
    df = pd.DataFrame([[cfg.size, cfg.deploy.split, cfg.deploy.pid, total_time, size_ratio, cardinality,
                        rest_size, orig_size, cardinality_raw, data_preprocess_time, node_emb_time, inst_emb_time,
                        build_time, pareto_time]],
                      columns=["size", "split", "pid", "total_time", "size", "cardinality", "rest_size",
                               "orig_size",
                               "cardinality_raw", "data_preprocess_time", "node_emb_time", "inst_emb_time",
                               "build_time", "pareto_time"])
    print(df)

    pid = str(cfg.deploy.pid) + ".csv"
    result_path = run_path / pid
    df.to_csv(result_path)


class IndepsetDeployer(Deployer):
    def __init__(self, cfg):
        Deployer.__init__(self, cfg)

    def worker(self):
        pass
