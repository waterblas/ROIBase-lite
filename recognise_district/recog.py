import os
import logging
from . import detect
try:
    import pkg_resources
    get_module_res = lambda *res: pkg_resources.resource_stream(__name__,
                                                        os.path.join(*res))
except ImportError:
    get_module_res = lambda *res: open(os.path.normpath(os.path.join(
                       os.getcwd(), os.path.dirname(__file__), *res)), 'rb')


_region_data = '/data/region.pb2'
default_params = {'penalty_factor': 500,
                  'freq_threshold': 2,
                  'limit': 1,
                  'alpha': 1.4}


def init(log_path=None, debug=False, **kwargs):
    """initial setting
    Args:
        params:
        penalty_factor
            +: 提高长文稀疏地点的召回 -: 降低长文稀疏地点的召回
        freq_threshold
            +: 出现多个地点的新闻，召回下降，准确提升 -: 反之
        limit
            +: 开始判断阈值提高 -: 反之
        alpha
            模糊结果过滤，+: 要求频率高
    """
    # log setting
    logger = logging.getLogger(__name__)
    if log_path:
        file_handler = logging.FileHandler(log_path)
        logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level=level)

    default_params.update(kwargs)
    detectD = detect.DetectDistrict(default_params, _region_data)
    return detectD

