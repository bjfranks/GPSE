from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('posenc')
def set_cfg_posenc(cfg):
    """Extend configuration with positional encoding options.
    """
    # Argument group for each Positional Encoding class.
    cfg.posenc_LapPE = CN()
    cfg.posenc_SignNet = CN()
    cfg.posenc_RWSE = CN()
    cfg.posenc_HKdiagSE = CN()
    cfg.posenc_ElstaticPE = CN()
    cfg.posenc_EquivStableLapPE = CN()
    cfg.posenc_GPSE = CN()
    cfg.posenc_GraphLog = CN()
    # NOTE: the following requires dataset.combine_output_pestat to be True
    cfg.posenc_CombinedPSE = CN()

    # Argument group for each Random Encoding class.
    cfg.randenc_FixedSE = CN()
    cfg.randenc_NormalSE = CN()
    cfg.randenc_UniformSE = CN()
    cfg.randenc_BernoulliSE = CN()
    cfg.randenc_UniformOSE = CN()
    cfg.randenc_BernoulliOSE = CN()

    # TODO: replace The aboves with the followings
    cfg.posenc_NormalRE = CN()
    cfg.posenc_NormalFixedRE = CN()
    cfg.posenc_UniformRE = CN()
    cfg.posenc_BernoulliRE = CN()
    cfg.posenc_UniformORE = CN()
    cfg.posenc_BernoulliORE = CN()

    # Argument group for each Graph Encoding class.
    cfg.graphenc_CycleGE = CN()
    cfg.graphenc_CycleGE.enable = False
    cfg.graphenc_RWGE = CN()
    cfg.graphenc_RWGE.enable = False

    # Common arguments to all PE types.
    for name in ['posenc_LapPE', 'posenc_SignNet', 'posenc_RWSE', 'posenc_GPSE',
                 'posenc_HKdiagSE', 'posenc_ElstaticPE', 'posenc_GraphLog',
                 'posenc_CombinedPSE']:
        pecfg = getattr(cfg, name)

        # Use extended positional encodings
        pecfg.enable = False

        # Neural-net model type within the PE encoder:
        # 'DeepSet', 'Transformer', 'Linear', 'none', ...
        pecfg.model = 'none'

        # Size of Positional Encoding embedding
        pecfg.dim_pe = 16

        # Number of layers in PE encoder model
        pecfg.layers = 3

        # Number of attention heads in PE encoder when model == 'Transformer'
        pecfg.n_heads = 4

        # Number of layers to apply in LapPE encoder post its pooling stage
        pecfg.post_layers = 0

        # Choice of normalization applied to raw PE stats: 'none', 'BatchNorm'
        pecfg.raw_norm_type = 'none'

        # In addition to appending PE to the node features, pass them also as
        # a separate variable in the PyG graph batch object.
        pecfg.pass_as_var = False

    # Config for EquivStable LapPE
    cfg.posenc_EquivStableLapPE.enable = False
    cfg.posenc_EquivStableLapPE.raw_norm_type = 'none'

    # Config for pretrained GNN P/SE encoder
    cfg.posenc_GPSE.model_dir = None
    cfg.posenc_GPSE.accelerator = "default"
    cfg.posenc_GPSE.rand_type = 'NormalSE'
    cfg.posenc_GPSE.use_repr = False  # use one layer before the output if True
    # What representation to use. 'one_layer_before' uses the representation of
    # the second to last layer in the poas_mp module as the repr. 'no_post_mp'
    # uses the input representation to the post_mp module as the repr, in other
    # words, no_post_mp skips the last pos_mp module.
    cfg.posenc_GPSE.repr_type = "one_layer_before"
    cfg.posenc_GPSE.virtual_node = False
    cfg.posenc_GPSE.input_dropout_be = 0.0
    cfg.posenc_GPSE.input_dropout_ae = 0.0
    cfg.posenc_GPSE.save = False
    cfg.posenc_GPSE.from_saved = False
    cfg.posenc_GPSE.tag = "1.0"

    # Loader for each graph
    cfg.posenc_GPSE.loader = CN()
    cfg.posenc_GPSE.loader.type = "full"
    cfg.posenc_GPSE.loader.num_neighbors = [30, 20, 10]
    cfg.posenc_GPSE.loader.fill_num_neighbors = 5
    cfg.posenc_GPSE.loader.batch_size = 1024

    # Multi MLP head hidden dimension. If None, set as the same as gnn.dim_inner
    cfg.gnn.multi_head_dim_inner = None
    cfg.posenc_GPSE.gnn_cfg = CN(cfg.gnn.copy())

    cfg.posenc_GraphLog.model_dir = "pretrained_models/graphlog.pth"

    # To be set at runtime based on the PSE to be combined
    cfg.posenc_CombinedPSE._raw_dim = None

    # Config for Laplacian Eigen-decomposition for PEs that use it.
    for name in ['posenc_LapPE', 'posenc_SignNet', 'posenc_EquivStableLapPE']:
        pecfg = getattr(cfg, name)
        pecfg.eigen = CN()

        # The normalization scheme for the graph Laplacian: 'none', 'sym', or 'rw'
        pecfg.eigen.laplacian_norm = 'sym'

        # The normalization scheme for the eigen vectors of the Laplacian
        pecfg.eigen.eigvec_norm = 'L2'

        # Maximum number of top smallest frequencies & eigenvectors to use
        pecfg.eigen.max_freqs = 10

        # Whether to stack eigenvalues as constant vectors with the
        # eigenvectors to form the final positional encoders
        pecfg.eigen.stack_eigval = False

        # Whether to skip eigenpairs correspond to the zero frequencies
        pecfg.eigen.skip_zero_freq = False

        # Whether to use the absolute value of the eigenvectors
        pecfg.eigen.eigvec_abs = False

    # Config for SignNet-specific options.
    cfg.posenc_SignNet.phi_out_dim = 4
    cfg.posenc_SignNet.phi_hidden_dim = 64

    for name in ['posenc_RWSE', 'posenc_HKdiagSE', 'posenc_ElstaticPE',
                 'graphenc_CycleGE', 'graphenc_RWGE']:
        pecfg = getattr(cfg, name)

        # Config for Kernel-based PE specific options.
        pecfg.kernel = CN()

        # List of times to compute the heat kernel for (the time is equivalent to
        # the variance of the kernel) / the number of steps for random walk kernel
        # Can be overridden by `posenc.kernel.times_func`
        pecfg.kernel.times = []

        # Python snippet to generate `posenc.kernel.times`, e.g. 'range(1, 17)'
        # If set, it will be executed via `eval()` and override posenc.kernel.times
        pecfg.kernel.times_func = ''

    # Override default, electrostatic kernel has fixed set of 7 measures.
    cfg.posenc_ElstaticPE.kernel.times_func = 'range(7)'

    cfg.randenc_FixedSE.enable = False
    cfg.randenc_FixedSE.dim_pe = 1

    cfg.randenc_NormalSE.enable = False
    cfg.randenc_NormalSE.dim_pe = 9

    cfg.randenc_UniformSE.enable = False
    cfg.randenc_UniformSE.dim_pe = 9

    cfg.randenc_UniformOSE.enable = False
    cfg.randenc_UniformOSE.dim_pe = 9

    cfg.randenc_BernoulliSE.enable = False
    cfg.randenc_BernoulliSE.threshold = 0.5
    cfg.randenc_BernoulliSE.dim_pe = 9

    cfg.randenc_BernoulliOSE.enable = False
    cfg.randenc_BernoulliOSE.dim_pe = 9

    for name in ["NormalRE", "NormalFixedRE", "UniformRE", "BernoulliRE", "UniformORE", "BernoulliORE"]:
        pecfg = getattr(cfg, f"posenc_{name}")
        pecfg.enable = False
        pecfg.dim_pe = 20
