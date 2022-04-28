class CFG:
    seed = 42
    model_name = 'dm_nfnet_f0'
    pretrained = True
    num_workers = 12
    img_size = 512
    num_classes = 100
    lr = 1e-4
    max_lr = 1e-3
    pct_start = 0.2
    div_factor = 1.0e+3
    final_div_factor = 1.0e+3
    num_epochs = 40
    batch_size = 36
    accum = 1
    precision = 16
    n_fold = 4
    fold_idx = 3
    test_time_augmentation = False
    loss = "cross_entropy"

    # arc_face_related
    arc_face_head = False
    arcface_m_x = 0.45
    arcface_m_y = 0.45
    arcface_s = 45
    arcface_m = 0.3
    neck_option = "option-N" #option-D, option-S
    @staticmethod
    def add_parser(parser):
        for k, v in CFG.__dict__.items():
            if isinstance(v, int) or isinstance(v,float) or isinstance(v, str) or isinstance(v, bool):
                if not isinstance(v, bool):
                    parser.add_argument("--"+k, default = v, type=type(v))
                else:
                    parser.add_argument("--no-"+k, action="store_false", dest=k) \
                        if v else parser.add_argument("--" + k, action="store_true", dest=k)
        return parser
