"""
In this file we test the tensorflow specific logging methods.
"""
from time import sleep
import torch.nn as nn

from ml_logger import logger
from ml_logger_tests.test_ml_logger import setup, log_dir # noqa
from ml_logger_tests.conftest import LOCAL_TEST_DIR


class View(nn.Module):
    def __init__(self, *size):
        """
        reshape nn module.

        :param size: reshapes size of tensor to [batch, *size]
        """
        super().__init__()
        self.size = size

    def forward(self, x):
        return x.view(-1, *self.size)


demo_module = nn.Sequential(
    nn.Conv2d(20, 32, kernel_size=4, stride=2),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=4, stride=2),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=4, stride=2),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 32, kernel_size=4, stride=2),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    View(128),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)


def test_torch_save(setup):
    logger.remove("modules/test_torch_save.pkl")
    sleep(0.2)
    logger.torch_save(demo_module, "modules/test_torch_save.pkl")


def test_torch_load(setup):
    with logger.Sync():
        test_torch_save(setup)
    sleep(0.2)
    module = logger.torch_load("modules/test_torch_save.pkl",weights_only=False)
    print(module)


def test_torch_load_sync(setup):
    with logger.Sync():
        test_torch_save(setup)
        sleep(0.2)
        module = logger.torch_load("modules/test_torch_save.pkl",weights_only=False)
    print(module)


def test_module(setup):
    logger.save_module(demo_module, "modules/test_module.pkl")
    logger.remove("modules/test_module.pkl")


def xtest_modules(setup):
    logger.save_modules(mod_0=demo_module, mod_1=demo_module, path="modules/test_modules.pkl")


def test_load_module(setup):
    logger.save_module(demo_module, "modules/test_load_module.pkl")
    sleep(0.2)
    logger.load_module(demo_module, "modules/test_load_module.pkl")
    logger.remove("modules/test_load_module.pkl")


def test_jit_save(setup):
    logger.torch_jit_save(demo_module, "modules/test_module.pth")


def test_jit_load(setup):
    test_jit_save(setup)
    demo_module = logger.torch_jit_load("modules/test_module.pth")
    logger.remove("modules/test_module.pth")


if __name__ == "__main__":
    setup(LOCAL_TEST_DIR)
    test_module(setup)
    # test_modules(setup)
    test_load_module(setup)
