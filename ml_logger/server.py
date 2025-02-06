import os
from datetime import datetime
from io import BytesIO
from typing import Sequence, Union

import dill  # done: switch to dill instead
from ml_logger.helpers import load_from_file
from ml_logger.serdes import deserialize, serialize
from ml_logger.structs import (
    ALLOWED_TYPES,
    LogEntry,
    LogOptions,
    LoadEntry,
    RemoveEntry,
    PingData,
    GlobEntry,
    MoveEntry,
    CopyEntry,
    MakeVideoEntry,
    ArchiveEntry,
    ShellEntry,
)
from termcolor import cprint


class LoggingServer:
    silent = None

    def abs_path(self, key: str):
        if key.startswith("/"):
            return os.path.join(self.root, key[1:]).replace("/./", "/")
        return os.path.join(self.cwd, key).replace("/./", "/")

    def __init__(self, cwd, root="/", allow_shell=False, silent=False):
        self.cwd = os.path.abspath(cwd)
        self.root = os.path.abspath(root)
        os.makedirs(self.cwd, exist_ok=True)
        os.makedirs(self.root, exist_ok=True)

        self.silent = silent
        self.allow_shell = allow_shell

        if not silent:
            cprint(f"logging data to {root}", "green")

    configure = __init__

    def serve(self, host, port, workers):
        import sanic

        self.app = sanic.Sanic("ml_logger.server")
        self.app.add_route(self.log_handler, "/", methods=["POST"])
        self.app.add_route(self.read_handler, "/", methods=["GET"])
        self.app.add_route(self.stream_handler, "/stream", methods=["GET"])
        # done: need a file serving url
        self.app.static("/files", self.root, stream_large_files=True)
        self.app.add_route(self.ping_handler, "/ping", methods=["POST"])
        self.app.add_route(self.glob_handler, "/glob", methods=["POST"])
        self.app.add_route(self.remove_handler, "/", methods=["DELETE"])
        self.app.add_route(self.move_handler, "/move", methods=["POST"])
        self.app.add_route(self.copy_handler, "/copy", methods=["POST"])
        self.app.add_route(self.make_video_handler, "/make_video", methods=["POST"])
        self.app.add_route(self.make_archive_handler, "/make_archive", methods=["POST"])
        self.app.add_route(self.shell_handler, "/shell", methods=["POST"])

        self.app.run(host=host, port=port, workers=workers, debug=Params.debug)

    async def stream_handler(self, req):
        import sanic

        if not req.json:
            msg = f"request json is empty: {req.text}"
            print(msg)
            return sanic.response.text(msg)
        load_entry = LoadEntry(**req.json)
        print(f"streaming: {load_entry.key}")
        path = self.abs_path(load_entry.key)
        return await sanic.response.file_stream(path)

    async def ping_handler(self, req):
        import sanic

        if not req.json:
            msg = f"request json is empty: {req.text}"
            print(msg)
            return sanic.response.text(msg)
        ping_data = PingData(**req.json)
        print("received ping data: {} type: {}".format(ping_data.status, ping_data.exp_key))
        data = self.ping(ping_data.exp_key, ping_data.status, ping_data.burn)
        return sanic.response.text(data)

    def ping(self, exp_key, status, burn=True):
        status_path = os.path.join(exp_key, "__presence")
        self.log(status_path, dict(status=status, time=datetime.now()), dtype="yaml", options=LogOptions(overwrite=True, write_mode="key"))
        signal_path = os.path.join(exp_key, "__signal.pkl")
        res = self.load(signal_path, "pkl")
        if burn:
            self.remove(signal_path)
        return serialize(res)

    async def glob_handler(self, req):
        import sanic

        if not req.json:
            msg = f"request json is empty: {req.text}"
            print(msg)
            return sanic.response.text(msg)

        glob_entry = GlobEntry(**req.json)
        print(
            "globbing: work directory {} query: {} is_recursive: {} start: {}, stop: {}".format(
                glob_entry.wd, glob_entry.query, glob_entry.recursive, glob_entry.start, glob_entry.stop
            )
        )
        try:
            file_paths = self.glob(
                query=glob_entry.query,
                wd=glob_entry.wd,
                recursive=glob_entry.recursive,
                start=glob_entry.start,
                stop=glob_entry.stop,
            )
            return sanic.response.json(file_paths, status=200)
        # note: do we need this? sanic doesn't already do this?
        except FileNotFoundError as e:
            return sanic.response.text(e, status=404)

    async def read_handler(self, req):
        import sanic

        if not req.json:
            msg = f"request json is empty: {req.text}"
            print(msg)
            return sanic.response.text(msg)
        load_entry = LoadEntry(**req.json)
        print("loading: {} type: {}: start: {}, stop: {}".format(load_entry.key, load_entry.type, load_entry.start, load_entry.stop))
        res = self.load(load_entry.key, load_entry.type, load_entry.start, load_entry.stop)
        data = serialize(res)
        return sanic.response.text(data)

    async def remove_handler(self, req):
        import sanic

        if not req.json:
            msg = f"request json is empty: {req.text}"
            cprint(msg, "red")
            return sanic.response.text(msg)
        remove_entry = RemoveEntry(**req.json)
        print(f"removing: {remove_entry.key}")
        self.remove(remove_entry.key)
        return sanic.response.text("ok")

    async def move_handler(self, req):
        import sanic

        if not req.json:
            msg = f"request json is empty: {req.text}"
            cprint(msg, "red")
            return sanic.response.text(msg)
        move_entry = MoveEntry(**req.json)
        print(f"moving {move_entry.source} to {move_entry.to}")
        self.move(move_entry.source, move_entry.to)
        return sanic.response.text(move_entry.to)

    async def copy_handler(self, req):
        import sanic

        if not req.json:
            msg = f"request json is empty: {req.text}"
            cprint(msg, "red")
            return sanic.response.text(msg)
        copy_entry = CopyEntry(**req.json)
        print(f"duplicating {copy_entry.source} to {copy_entry.to}")
        self.duplicate(
            copy_entry.source, copy_entry.to, copy_entry.exists_ok, follow_symlink=copy_entry.follow_symlink, symlinks=copy_entry.symlinks
        )
        return sanic.response.text(copy_entry.to)

    async def log_handler(self, req):
        import sanic
        from collections import Sequence

        if req.files:
            (file,) = req.files["file"]
            print(f"uploading: {file.name} len: {len(file.body)}")
            self.log(file.name, file.body, "byte", LogOptions(overwrite=True))
            return sanic.response.json(dict(name=file.name, length=len(file.body), overwrite=True))
        elif not req.json:
            cprint(f"request json is empty: {req.text}", "red")
            return sanic.response.text("Request json is empty")
        log_entry = LogEntry(**req.json)
        print(f"writing: {log_entry.key} type: {log_entry.type} options: {log_entry.options}")
        data = deserialize(log_entry.data)
        if not log_entry:
            options = None
        elif log_entry.options is None:
            options = log_entry.options
        elif isinstance(log_entry.options, Sequence):
            options = LogOptions(*log_entry.options)
        elif isinstance(log_entry.options, dict):
            options = LogOptions(**log_entry.options)
        else:
            options = None
        self.log(log_entry.key, data, log_entry.type, options)
        return sanic.response.text("ok")

    async def make_video_handler(self, req):
        import sanic

        if not req.json:
            msg = f"request json is empty: {req.text}"
            cprint(msg, "red")
            return sanic.response.json({"message": msg})

        entry = MakeVideoEntry(**req.json)
        print(f"making video to {entry.key}, {entry}")
        result = self.make_video(files=entry.files, key=entry.key, wd=entry.wd, order=entry.order, options=entry.options)
        return sanic.response.json({"result": result})

    async def make_archive_handler(self, req):
        import sanic

        if not req.json:
            msg = f"request json is empty: {req.text}"
            cprint(msg, "red")
            return sanic.response.json({"message": msg})

        entry = ArchiveEntry(**req.json)
        print(f"making archive to {entry.root_dir}, {entry.base_name}")
        result = self.make_archive(
            base_name=entry.base_name, format=entry.format, root_dir=entry.root_dir, base_dir=entry.base_dir, options=entry.options
        )
        return sanic.response.json({"result": result})

    async def shell_handler(self, req):
        import sanic

        if not req.json:
            msg = f"request json is empty: {req.text}"
            cprint(msg, "red")
            return sanic.response.json({"message": msg})

        entry = ShellEntry(**req.json)
        print(f"executing shell command {entry}")
        result = self.shell(command=entry.command, wd=entry.wd, options=entry.options)
        return sanic.response.json({"result": result})

    def glob(self, query, wd, recursive: bool, start, stop):
        """
        Glob under the work directory. (so that the wd is not included in the file paths that are returned.)

        :param query:
        :param wd: Use double slash //home/directory to access absolute path of the
            server host environment. single leading slash accesses the data_dir.
        :param recursive:
        :param start:
        :param stop:
        :return:
        """

        from glob import iglob
        from itertools import islice

        from ml_logger.helpers.file_helpers import CwdContext

        try:
            with CwdContext(self.abs_path(wd or "")):
                return list(islice(iglob(query, recursive=recursive), start, stop))
        except PermissionError:
            print("PermissionError:", os.path.join(self.root, wd or ""))
            return None
        except FileNotFoundError:
            return None

    def load(self, key, dtype, start=None, stop=None):
        """
        when key starts with a single slash as in "/debug/some-run", the leading slash is removed
        and the remaining path is pathJoin'ed with the data_dir of the server.

        So if you want to access absolute path of the filesystem that the logging server is in,
        you should append two leadning slashes. This way, when the leanding slash is removed,
        the remaining path is still an absolute value and joining with the data_dir would post
        no effect.

        "//home/ubuntu/ins-runs/debug/some-other-run" would point to the system absolute path.

        Modes:
            "byte": returns the binary string.
            "text": returns the file's content as plain-text
            "pkl": reads the content of a pickle file
            "np": reads the content of numpy file.

        Note: We might want to do the hydration of the pickle and numpy files on the client-side.
        This way we only send the serilized data over-the-wire.

        :param key: a path string
        :param dtype: (str), one of "byte", "text", "pickle", "np"
        :param start: end index
        :param stop: start index
        :return: None, or a tuple of each one of the data chunks logged into the file.
        """
        abs_path = self.abs_path(key)
        if dtype == "byte":
            try:
                return list(load_from_file(abs_path))[start:stop]
            except FileNotFoundError:
                return None
        elif dtype == "text":
            try:
                with open(abs_path, "r") as f:
                    return f.read()
            except FileNotFoundError:
                return None
        elif dtype == "pkl":
            from .helpers import load_from_pickle

            try:
                return list(load_from_pickle(abs_path))[start:stop]
            except FileNotFoundError:
                return None
        elif dtype == "np":
            import numpy

            try:
                return numpy.load(abs_path)
            except FileNotFoundError:
                return None
        elif dtype == "json":
            import json

            try:
                with open(abs_path, "r") as f:
                    return json.load(f)
            except Exception:
                return None
        elif dtype == "h5":
            import h5py

            file_path, rest = abs_path.split(":")
            object_keys = rest.split(",")
            try:
                with h5py.File(file_path, "r") as f:
                    return tuple(f[key][:] for key in object_keys)
            except FileNotFoundError:
                return None

    def remove(self, key):
        """
        removes by key.

        :param key: the path from the logging directory.
        :return: None
        """
        abs_path = self.abs_path(key)
        try:
            os.remove(abs_path)
        except FileNotFoundError:
            return None
        except OSError:
            import shutil

            return shutil.rmtree(abs_path, ignore_errors=True)

    def move(self, source, to):
        """
        move directories or files

        :param source:
        :param to:
        :param dirs_exist_ok:
        :return:
        """
        import shutil

        assert isinstance(source, str), "src needs to be a string"

        abs_source = self.abs_path(source)
        abs_to = self.abs_path(to)
        if abs_to == abs_source:
            return None
        # try:
        #     # remove that tree first
        #     shutil.rmtree(abs_to)
        # except FileNotFoundError as e:
        #     pass
        return shutil.move(abs_source, abs_to)

    def duplicate(self, src, target, exists_ok=True, follow_symlink=True, symlinks=False):
        import shutil

        assert isinstance(src, str), "src needs to be a string"

        abs_target = self.abs_path(target)
        abs_src = self.abs_path(src)
        try:
            # copy directory
            return shutil.copytree(abs_src, abs_target, symlinks=symlinks, dirs_exist_ok=exists_ok)
        except NotADirectoryError:
            # copy files
            os.makedirs(os.path.dirname(abs_target), exist_ok=exists_ok)
            return shutil.copyfile(abs_src, abs_target, follow_symlinks=follow_symlink)

    def save_buffer(self, key, buff):
        assert isinstance(buff, BytesIO), "buff needs to be a BytesIO object."

        abs_path = self.abs_path(key)

        with open(abs_path, "wb") as t:
            while True:
                content = buff.read()
                if content == b"":
                    break
                t.write(content)
        return key

    def log(self, key, data, dtype, options: LogOptions = None):
        """
        handler function for writing data to the server. Can be called directly.

        :param key:
        :param data:
        :param dtype:
        :param options:
        :return:
        """
        # todo: overwrite mode is not tested and not in-use.
        write_mode = "w" if options and options.overwrite else "a"
        abs_path = self.abs_path(key)
        parent_dir = os.path.dirname(abs_path)
        # fixme: There is a race condition with multiple requests
        if dtype == "log":
            try:
                with open(abs_path, write_mode + "b") as f:
                    dill.dump(data, f)
            except FileNotFoundError:
                os.makedirs(parent_dir, exist_ok=True)
                with open(abs_path, write_mode + "b") as f:
                    dill.dump(data, f)
        if dtype == "byte":
            try:
                with open(abs_path, write_mode + "b") as f:
                    f.write(data)
            except FileNotFoundError:
                os.makedirs(parent_dir, exist_ok=True)
                with open(abs_path, write_mode + "b") as f:
                    f.write(data)
        elif dtype.startswith("text"):
            try:
                with open(abs_path, write_mode + "+", encoding="utf-8") as f:
                    f.write(data)
            except FileNotFoundError:
                os.makedirs(parent_dir, exist_ok=True)
                with open(abs_path, write_mode + "+", encoding="utf-8") as f:
                    f.write(data)
        elif dtype.startswith("yaml"):
            import ruamel.yaml

            if ruamel.yaml.version_info < (0, 15):
                yaml = ruamel.yaml
                StringIO = ruamel.yaml.StringIO
                load_fn = yaml.safe_load
            else:
                from ruamel.yaml import YAML, StringIO

                yaml = YAML()
                yaml.explict_start = True
                load_fn = yaml.load

            stream = StringIO()
            yaml.dump(data, stream)
            output = stream.getvalue()
            try:
                with open(abs_path, write_mode + "+") as f:
                    if options.write_mode == "key":
                        d = load_fn("\n".join(f))
                        if d is not None:
                            d.update(output)
                            output = d
                    f.write(output)
            except FileNotFoundError:
                os.makedirs(parent_dir, exist_ok=True)
                with open(abs_path, write_mode + "+") as f:
                    if options.write_mode == "key":
                        d = load_fn("\n".join(f))
                        if d is not None:
                            d.update(output)
                            output = d
                    f.write(output)
        elif dtype.startswith("image"):
            abs_path = self.abs_path(key)
            if "." not in key:
                abs_path = abs_path + ".png"
            from PIL import Image

            assert data.dtype in ALLOWED_TYPES, "image datatype must be one of {}".format(ALLOWED_TYPES)
            if len(data.shape) == 3 and data.shape[-1] == 1:
                data.resize(data.shape[:-1])
            im = Image.fromarray(data)
            try:
                im.save(abs_path)
            except FileNotFoundError:
                os.makedirs(parent_dir, exist_ok=True)
                im.save(abs_path)

    def make_video(self, files: Union[str, Sequence], key, wd, order: str, options=None):
        import imageio

        abs_key = self.abs_path(key)
        abs_wd = self.abs_path(wd or "")

        if isinstance(files, str):
            files = self.glob(files, wd=wd, recursive=True, start=None, stop=None)
            if order in ["descending", -1]:
                files = sorted(files)[::-1]
            elif order in ["ascending", 1]:
                files = sorted(files)

        if not files:
            return

        try:
            writer = imageio.get_writer(abs_key, **options)
        except FileNotFoundError:
            os.makedirs(os.path.dirname(abs_key), exist_ok=True)
            writer = imageio.get_writer(abs_key, **options)

        for fname in files:
            img = imageio.imread(os.path.join(abs_wd, fname))
            writer.append_data(img)

        writer.close()
        return key

    def make_archive(self, base_name, format, root_dir, base_dir, options=None):
        import shutil

        shutil.make_archive(
            base_name=self.abs_path(base_name), format=format, root_dir=self.abs_path(root_dir), base_dir=base_dir, **options
        )
        return base_name

    def shell(self, command, wd="", options=None):
        """
        Logger server root path is exposed through $ROOT environment variable.
        The current work direction is exposed through $PWD environment variable.

        :param command:
        :param wd:
        :param options:
        :return:
        """
        import subprocess

        if not self.allow_shell:
            return "", "shell commands are disabled. using --shell flag to turn it on.", 1

        abs_wd = self.abs_path(wd)
        out = subprocess.run(command, **options, shell=True, capture_output=True, cwd=abs_wd, env={"ROOT": self.root, "PWD": self.cwd})
        return out.stdout.decode("utf-8"), out.stderr.decode("utf-8"), out.returncode


if __name__ == "__main__":
    from params_proto import ParamsProto, Proto, Flag

    class Params(ParamsProto):
        logdir: str = Proto("/tmp/logging-server", help="The directory for saving the logs")
        port = Proto(8081, help="port for the logging server")
        host = Proto(
            "127.0.0.1",
            help="IP address for running the server. Default only allows localhost from "
            "making requests. If you want to allow all ip, set this to '0.0.0.0'.",
        )
        workers = Proto(1, help="Number of workers to run in parallel")
        debug = Flag(help="boolean flag for printing out debug traces")
        shell = Flag(help="boolean flag for allowing shell commands to be executed on the server.")

    import pkg_resources

    v = pkg_resources.get_distribution("ml_logger").version

    print(f"running ml_logger.server version {v}")
    server = LoggingServer(Params.logdir, root=Params.logdir, allow_shell=Params.shell)
    server.serve(host=Params.host, port=Params.port, workers=Params.workers)
