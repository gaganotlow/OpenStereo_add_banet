# @Time    : 2023/8/26 13:02
# @Author  : zhangchenming

from .models.casnet.trainer import Trainer as CasStereoTrainer
from .models.cfnet.trainer import Trainer as CFNetTrainer
# from .models.aanet.trainer import Trainer as AANetTrainer
from .models.coex.trainer import Trainer as CoExTrainer
from .models.fadnet.trainer import Trainer as FADNetTrainer
from .models.gwcnet.trainer import Trainer as GwcNetTrainer
from .models.igev.trainer import Trainer as IGEVTrainer
from .models.msnet.trainer import Trainer as MSNetTrainer
from .models.psmnet.trainer import Trainer as PSMNetTrainer
from .models.sttr.trainer import Trainer as STTRTrainer
from .models.lightstereo.trainer import Trainer as LightStereoTrainer
from .models.banet2d.trainer import Trainer as BANet2DTrainer
from .models.stereobase.trainer import Trainer as StereoBaseGRUTrainer
# from .models.iinet.trainer import Trainer as IINetTrainer
from .models.monster.trainer import Trainer as MonsterTrainer
from .models.igevpp.trainer import Trainer as IGEVPPTrainer
from .models.igev_rt.trainer import Trainer as IGEVRTTrainer

try:
# FoundationStereo：环境与训练入口见仓库根目录 TRAIN.md；上游文档见 OpenStereo 官方仓库。
    from .models.foundationstereo.trainer import Trainer as FoundationStereoTrainer
except Exception:
    FoundationStereoTrainer = None

try:
    from timm.layers import DropPath
    from .models.fast_foundationstereo.trainer import Trainer as FastFoundationStereoTrainer
except Exception:
    FastFoundationStereoTrainer = None

# If you want to train/eval NMRF-Stereo, you need to build deformable attention and superpixel-guided disparity downsample operator: 'cd stereo/modeling/models/nmrf/ops && sh make.sh && cd ..'
# try:
#     from .models.nmrf.trainer import Trainer as NMRFTrainer
# except:
#     raise ValueError("If you want to train/eval NMRF-Stereo, you need to build deformable attention and superpixel-guided disparity downsample operator: 'cd stereo/modeling/models/nmrf/ops && sh make.sh && cd ..'")

__all__ = {
    'STTR': STTRTrainer,
    'PSMNet': PSMNetTrainer,
    'MSNet2D': MSNetTrainer,
    'MSNet3D': MSNetTrainer,
    'IGEV': IGEVTrainer,
    'GwcNet': GwcNetTrainer,
    'FADNet': FADNetTrainer,
    'CoExNet': CoExTrainer,
    # 'AANet': AANetTrainer,
    'CFNet': CFNetTrainer,
    'CasGwcNet': CasStereoTrainer,
    'CasPSMNet': CasStereoTrainer,
    'LightStereo': LightStereoTrainer,
    'BANet2D': BANet2DTrainer,
    'StereoBaseGRU': StereoBaseGRUTrainer,
    **({'FoundationStereo': FoundationStereoTrainer} if FoundationStereoTrainer is not None else {}),
    **({'FastFoundationStereo': FastFoundationStereoTrainer} if FastFoundationStereoTrainer is not None else {}),
    # 'IInet': IINetTrainer,
    # 'NMRF': NMRFTrainer
    "MonSter": MonsterTrainer,
    "IGEVPP": IGEVPPTrainer,
    "IGEVRT": IGEVRTTrainer
}


def build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer):
    trainer = __all__[cfgs.MODEL.NAME](args, cfgs, local_rank, global_rank, logger, tb_writer)
    return trainer
