_LAZY_IMPORTS = {
    'DiffJPEG': ('basicsr.utils.diffjpeg', 'DiffJPEG'),
    'FileClient': ('basicsr.utils.file_client', 'FileClient'),
    'USMSharp': ('basicsr.utils.img_process_util', 'USMSharp'),
    'usm_sharp': ('basicsr.utils.img_process_util', 'usm_sharp'),
    'crop_border': ('basicsr.utils.img_util', 'crop_border'),
    'imfrombytes': ('basicsr.utils.img_util', 'imfrombytes'),
    'img2tensor': ('basicsr.utils.img_util', 'img2tensor'),
    'imwrite': ('basicsr.utils.img_util', 'imwrite'),
    'tensor2img': ('basicsr.utils.img_util', 'tensor2img'),
    'AvgTimer': ('basicsr.utils.logger', 'AvgTimer'),
    'MessageLogger': ('basicsr.utils.logger', 'MessageLogger'),
    'get_env_info': ('basicsr.utils.logger', 'get_env_info'),
    'get_root_logger': ('basicsr.utils.logger', 'get_root_logger'),
    'init_tb_logger': ('basicsr.utils.logger', 'init_tb_logger'),
    'init_wandb_logger': ('basicsr.utils.logger', 'init_wandb_logger'),
    'check_resume': ('basicsr.utils.misc', 'check_resume'),
    'get_time_str': ('basicsr.utils.misc', 'get_time_str'),
    'make_exp_dirs': ('basicsr.utils.misc', 'make_exp_dirs'),
    'mkdir_and_rename': ('basicsr.utils.misc', 'mkdir_and_rename'),
    'scandir': ('basicsr.utils.misc', 'scandir'),
    'set_random_seed': ('basicsr.utils.misc', 'set_random_seed'),
    'sizeof_fmt': ('basicsr.utils.misc', 'sizeof_fmt'),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'basicsr.utils' has no attribute {name!r}")

    from importlib import import_module

    module_name, attr_name = _LAZY_IMPORTS[name]
    attr = getattr(import_module(module_name), attr_name)
    globals()[name] = attr
    return attr
