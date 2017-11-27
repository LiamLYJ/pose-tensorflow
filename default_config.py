from easydict import EasyDict as edict

cfg = edict()

cfg.stride = 8.0
cfg.weigh_part_predictions = False
cfg.weigh_negatives = False
cfg.fg_fraction = 0.25
cfg.weigh_only_present_joints = False
cfg.mean_pixel = [123.68, 116.779, 103.939]
cfg.shuffle = True
cfg.snapshot_prefix = "snapshot"
cfg.log_dir = "log"
cfg.global_scale = 1.0
cfg.location_refinement = False
cfg.locref_stdev = 7.2801
cfg.locref_loss_weight = 1.0
cfg.locref_huber_loss = True
cfg.optimizer = "sgd"
cfg.intermediate_supervision = False
cfg.intermediate_supervision_layer = 12
cfg.regularize = False
cfg.weight_decay = 0.0001
# cfg.mirror = False
cfg.mirror = True
cfg.crop = False
cfg.crop_pad = 0
cfg.scoremap_dir = "test"
cfg.dataset = ""
# cfg.dataset_type = "default"  # options: "default", "coco"
cfg.dataset_type = "coco"
cfg.use_gt_segm = False
cfg.batch_size = 1
cfg.video = False
cfg.video_batch = False
cfg.sparse_graph = []
cfg.pairwise_stats_collect = False
cfg.pairwise_stats_fn = "pairwise_stats.mat"
cfg.pairwise_predict = False
cfg.pairwise_huber_loss = True
cfg.pairwise_loss_weight = 1.0
cfg.tensorflow_pairwise_order = True
# 14 or 17 point 
cfg.default_coco = True
