"""
# Tests for ml-logger.

## Testing with a server

To test with a live server, first run (in a separate console)
```
python -m ml_logger.server --logdir /tmp/ml-logger-debug
```
or do:
```bash
make start-test-server
```

Then run this test script with the option:
```bash
python -m pytest tests --capture=no --logdir http://0.0.0.0:8081
```
or do
```bash
make test-with-server
```
"""
from os.path import join as pathJoin
from time import sleep

import pytest

from ml_logger import logger
from ml_logger.helpers.color_helpers import percent
from ml_logger.ml_logger import Color, metrify


@pytest.fixture(scope='session')
def log_dir(request):
    return request.config.getoption('--logdir')


@pytest.fixture(scope="session")
def setup(log_dir):
    logger.configure('main_test_script', root=log_dir)
    print('cleaning this directory')
    logger.remove('')
    print(f"logging to {pathJoin(logger.root, logger.prefix)}")


@pytest.fixture(scope="session")
def setup_no_clean(log_dir):
    logger.configure('main_test_script', root=log_dir)

    print(f"logging to {pathJoin(logger.root, logger.prefix)}")


def test_glob(setup):
    kwargs = dict(query="*.pkl", wd="test-data")
    print(f'globbing {kwargs["query"]} under {kwargs["wd"]}')
    file_paths = logger.glob(**kwargs)
    print(f"globbed file paths: {[file_paths]}")


def test_log_data(setup):
    import numpy
    numpy.random.seed(0)
    d1 = numpy.random.randn(20, 10)
    logger.log_data(d1, 'test_file.pkl')
    sleep(0.1)
    d2 = numpy.random.randn(20, 10)
    logger.log_data(d2, 'test_file.pkl')
    sleep(0.1)

    data = logger.load_pkl('test_file.pkl')
    assert len(data) == 2, "data should contain two arrays"
    assert numpy.array_equal(data[0], d1), "first should be the same as d1"
    assert numpy.array_equal(data[1], d2), "first should be the same as d2"


def test_save_pkl_abs_path(setup):
    import numpy
    numpy.random.seed(0)

    d1 = numpy.random.randn(20, 10)
    logger.save_pkl(d1, "/tmp/ml-logger-test/test_file_1.pkl")
    sleep(0.1)

    data = logger.load_pkl("/tmp/ml-logger-test/test_file_1.pkl")
    assert len(data) == 1, "data should contain only one array because we overwrote it."
    assert numpy.array_equal(data[0], d1), "first should be the same as d2"


def test(setup):
    d = Color(3.1415926, 'red')
    s = "{:.1}".format(d)

    logger.log_params(G=dict(some_config="hey"))
    logger.log(step=0, some=Color(0.1, 'yellow'))
    logger.log(step=1, some=Color(0.28571, 'yellow', lambda v: "{:.5f}%".format(v * 100)))
    logger.log(step=2, some=Color(0.85, 'yellow', percent))
    logger.log({"some_var/smooth": 10}, some=Color(0.85, 'yellow', percent), step=3)
    logger.log(step=4, some=Color(10, 'yellow'))


def test_remove(setup):
    logger.log('this is a file', file="test.txt", flush=True)
    assert 'test.txt' in logger.glob("*")
    logger.remove('test.txt')


def test_move(setup):
    logger.log('this is a file', file="test.txt", flush=True)
    logger.move('test.txt', 'test_2.txt')
    assert 'test_2.txt' in logger.glob("*")
    logger.remove('test_2.txt')


def test_copy_file(setup):
    logger.log('this is a file', file="test.txt", flush=True)
    logger.duplicate('test.txt', 'test_2.txt')
    assert 'test.txt' in logger.glob("*")
    assert 'test_2.txt' in logger.glob("*")
    logger.remove('test.txt', 'test_2.txt')


def test_copy_directory(setup):
    logger.log('this is a file', file="test/test.txt", flush=True)
    logger.duplicate('test', 'test_2')
    assert 'test/test.txt' in logger.glob("**/*")
    assert 'test_2/test.txt' in logger.glob("**/*")
    logger.remove('test.txt', 'test_2.txt')


def test_read_params(setup):
    from ml_logger import logger

    config = {'key_1': 10, 'key_2': 20}

    logger.log_params(Config=config)
    config = logger.read_params('Config.key_1')
    assert config == 10
    assert logger.read_params('Config') == {'key_1': 10, 'key_2': 20}


def test_metrics_prefix(setup):
    from ml_logger import logger

    logger.remove("metrics.pkl")

    with logger.Prefix(metrics="evaluate/", sep=""):
        logger.log(loss=0.5, flush=True)

    assert logger.read_metrics("evaluate/loss")[0] == 0.5


def test_metrics_prefix_2(setup):
    from ml_logger import logger

    logger.remove("metrics.pkl")

    with logger.Prefix(metrics="evaluate"):
        logger.log(loss=1, flush=True)

    assert logger.read_metrics("evaluate/loss")[0] == 1.0


def test_store_metrics_prefix(setup):
    from ml_logger import logger

    logger.remove("metrics.pkl")

    for i in range(10):
        with logger.Prefix(metrics="test"):
            logger.store_metrics(value=1.0)
        with logger.Prefix(metrics="eval"):
            logger.store_metrics(value=3.0)

    logger.log_metrics_summary(key_values={'step': 10})

    assert logger.read_metrics("test/value/mean")[0] == 1.0
    assert logger.read_metrics("step")[0] == 10


def test_json(setup):
    a = dict(a=0)
    logger.save_json(dict(a=0), "data/d.json")
    b = logger.load_json("data/d.json")
    assert a == b, "a and b should be the same"


def test_json_abs(setup):
    a = dict(a=0)
    logger.save_json(dict(a=0), "/tmp/ml-logger-test/data/d.json")
    b = logger.load_json("/tmp/ml-logger-test/data/d.json")
    assert a == b, "a and b should be the same"


def test_jsonl(setup):
    logger.log_jsonl(dict(a=10), "data/d.jsonl")
    logger.log_jsonl(dict(a=12), "data/d.jsonl")

    result = logger.load_jsonl("data/d.jsonl")
    print(result)


def test_yaml(setup):
    a = dict(a=0)
    logger.save_yaml(a, "data/d.yaml")
    b = logger.load_yaml("data/d.yaml")
    assert a == b, "a and b should be identical"


def test_image(setup):
    import scipy.misc
    from ml_logger import logger

    logger.configure('ge_debug/debug/main_test_script', root="http://luma01.csail.mit.edu:4000")
    logger.job_started()

    print(logger)

    face_rgba = scipy.misc.face()

    logger.save_image(face_rgba / 255, 'rgb.png', normalize='grid')
    logger.save_image(face_rgba / 1000, f'rgba_face_{100}.png', normalize=False)
    path = logger.save_image(face_rgba / 1000, f'rgba_face_{100}_normalized.png', normalize="individual")
    print(f"files:/{logger.root}/{logger.prefix}/{path}")


def test_image_cmap(setup):
    import scipy.misc
    import numpy as np

    face_bw = scipy.misc.face().mean(axis=-1)
    # the image dtype can not be uint8 --- in that case no cmap is applied
    logger.save_image(face_bw, f"face_bw_cmap.png", cmap="cool", normalize=True)
    logger.save_image(face_bw.astype(np.uint8), f"face_bw_no_cmap.png")


def test_image_zero_norm(setup):
    import numpy as np

    image_zero = np.zeros((64, 64, 1))
    logger.save_image(image_zero, f"zero_{100}.png", normalize="grid")


def test_image_normalization(setup):
    import numpy as np

    image_zero = np.zeros((64, 64, 1))
    logger.save_image(image_zero, f"zero_{100}.png", normalize="grid")


def test_make_video(setup):
    import scipy.misc
    from time import sleep

    face = scipy.misc.face()
    h, w, _ = face.shape
    with logger.Sync():
        for i in range(10):
            logger.save_image(face[i:h - 100 + i, i:w - 100 + i], f"face_{i:04d}.png")

    file_list = logger.glob("face_*.png")
    # This one makes the video according to ascending order
    logger.make_video("face_*.png", key="videos/face_move_glob.mp4")
    logger.make_video("face_*.png", order='descending', key="videos/face_move_glob_descending.mp4")
    # This one makes the video according to the random order given by glob
    logger.make_video(file_list, key="videos/face_move_list.mp4")

    # this should not reaise an error
    path = logger.make_video("no_exist_*.png", key="videos/not_exist.mp4")
    assert path is None


def test_pyplot(setup):
    import scipy.misc
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    face = scipy.misc.face()
    logger.save_image(face, "face.png")

    fig = plt.figure(figsize=(4, 2))
    xs = np.linspace(0, 5, 1000)
    plt.plot(xs, np.cos(xs))
    logger.savefig("face_02.png", fig=fig)
    plt.close()

    fig = plt.figure(figsize=(4, 2))
    xs = np.linspace(0, 5, 1000)
    plt.plot(xs, np.cos(xs))
    logger.savefig('sine.pdf')


def test_video(setup):
    import numpy as np

    def im(x, y):
        canvas = np.ones((640, 480), dtype=np.float32) * 0.001
        for i in range(200):
            for j in range(200):
                if x - 5 < i < x + 5 and y - 5 < j < y + 5:
                    canvas[i, j] = 1
        return canvas

    frames = [im(100 + i, 80) for i in range(20)]

    logger.save_video(frames, "test_video.mp4")


def test_video_gif(setup):
    import numpy as np

    def im(x, y):
        canvas = np.zeros((200, 200))
        for i in range(200):
            for j in range(200):
                if x - 5 < i < x + 5 and y - 5 < j < y + 5:
                    canvas[i, j] = 1
        return canvas

    frames = [im(100 + i, 80) for i in range(20)]

    logger.save_video(frames, "test_video.gif")
    assert 'test_video.gif' in logger.glob('*.gif')
    logger.save_video(frames, "/tmp/ml-logger-test/videos/test_video.gif")
    assert 'ml-logger-test/videos/test_video.gif' in logger.glob('**/videos/*.gif', wd="/tmp")


def test_load_params(setup):
    pass


def test_diff(setup):
    logger.diff()


def test_git_rev(setup):
    print([logger.__head__])


def test_git_tags(setup):
    print([logger.__tags__])


def test_current_branch(setup):
    print([logger.__current_branch__])


def test_hostname(setup):
    assert len(logger.hostname) > 0, 'hostname should be non-trivial'
    print([logger.hostname])


def test_split(setup):
    assert logger.split() is None, 'The first tick should be None'
    assert type(logger.split()) is float, 'Then it should return a a float in the seconds.'


def test_start(setup):
    assert isinstance(logger.start(), float), "should be perf_counter (float)"
    assert isinstance(logger.split(), float), "should be time delta (float)"
    assert isinstance(logger.since(), float), "should be time delta (float)"


def test_ping(setup):
    print('test ping starts')
    signals = logger.ping('alive', 0.1)
    print(f"signals => {signals}")
    sleep(0.2)
    signals = logger.ping('alive', 0.2)
    print(f"signals => {signals}")

    logger.client.send_signal(logger.prefix, signal="stop")
    sleep(0.25)
    logger.client.send_signal(logger.prefix, signal="pause")
    sleep(0.15)

    for i in range(4):
        signals = logger.ping('other ping')
        print(f"signals => {signals}")
        sleep(0.4)

    logger.ping('completed')


def test_metrify():
    import numpy as np
    d = np.array(10)
    assert metrify(d) == 10
    d = np.array(10.0)
    assert metrify(d) == 10.0
    d = np.array([10.0, 2])
    assert metrify(d) == [10.0, 2]
    d = np.array([10.0, 2])
    assert metrify(d) == [10.0, 2]


def test_every():
    acc = sum([i for i in range(100) if logger.every(10)])
    assert acc == sum(list(range(100))[9::10])

    acc = sum([i for i in range(101) if logger.every(10, 'tail', start_on=1)])
    assert acc == sum(list(range(0, 101, 10)))

    i_sum, j_sum = 0, 0
    for i in range(100):
        for j in range(100):
            if logger.every(5, "j"):
                j_sum += j
            if logger.every(50, "i"):
                i_sum += i
    assert i_sum == 4950 * 2, "i should be summed twice each iteration"
    assert j_sum == sum(list(range(4, 100, 5))) * 100, "j should be the sum ⨉ 100"


def test_timing():
    with logger.time("upload files"):
        import time
        time.sleep(0.1)

    for i in range(100):
        with logger.time("upload files", interval=50):
            import time
            time.sleep(0.001)


def test_capture_error():
    with logger.capture_error():
        raise RuntimeError("this should not fail")

    logger.print("works!", color="green")


def test_job_status(setup):
    logger.job_started()
    logger.job_running()
    logger.job_completed()
    logger.job_errored()


def test_get_exps(setup_no_clean):
    with logger.Prefix('..'):
        all = logger.get_exps("**/parameters.pkl")
        assert 'job.status' in all.columns


def test_get_exps_without_pkl(setup_no_clean):
    all = logger.get_exps("**")
    assert 'job.status' in all.columns


def test_shell(setup):
    # only needed for local test
    if not logger.root.startswith("http"):
        import os
        os.makedirs(logger.root + "/" + logger.prefix, exist_ok=True)

    stdout, stderr, code = logger.shell("ls")
    assert code == 0, "return code should be 0"


if __name__ == "__main__":
    # setup(LOCAL_TEST_DIR)
    # test(None)
    # test_video(None)
    # test_video_gif(None)
    test_every()
