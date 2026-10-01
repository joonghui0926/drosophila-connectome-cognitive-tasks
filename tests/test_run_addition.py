import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_script():
    path = Path(__file__).parents[1] / "scripts" / "run_addition.py"
    spec = importlib.util.spec_from_file_location("run_addition", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(tmp_path, gpus):
    return SimpleNamespace(
        graph_dir=Path("graph"),
        dataset=Path("data.json"),
        config=Path("config.json"),
        output=tmp_path / "runs",
        gpus=gpus,
        seeds=[11, 22, 33, 44],
        max_epochs=100,
        min_epochs=20,
        patience=15,
        resume=False,
    )


class Process:
    def __init__(self):
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


def test_addition_launcher_uses_available_gpus_in_waves(tmp_path, monkeypatch):
    module = load_script()
    launches = []

    def launch(command, *, env, stdout, stderr):
        launches.append((command, env["CUDA_VISIBLE_DEVICES"]))
        return Process()

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    module.run(arguments(tmp_path, [2, 5, 7]))
    assert [item[1] for item in launches] == ["2", "5", "7", "2"]
    assert [item[0][-1] for item in launches] == ["11", "22", "33", "44"]


def test_addition_launcher_accepts_one_gpu(tmp_path, monkeypatch):
    module = load_script()
    launches = []

    def launch(command, *, env, stdout, stderr):
        launches.append(env["CUDA_VISIBLE_DEVICES"])
        return Process()

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    module.run(arguments(tmp_path, [0]))
    assert launches == ["0", "0", "0", "0"]


def test_addition_launcher_runs_only_requested_seeds(tmp_path, monkeypatch):
    module = load_script()
    args = arguments(tmp_path, [0])
    args.seeds = [11, 22]
    launches = []

    def launch(command, **kwargs):
        launches.append(command)
        return Process()

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    module.run(args)
    assert [command[-1] for command in launches] == ["11", "22"]


@pytest.mark.parametrize("seeds", [[], [11, 11], [-1]])
def test_addition_launcher_rejects_invalid_seeds(tmp_path, seeds):
    module = load_script()
    args = arguments(tmp_path, [0])
    args.seeds = seeds
    with pytest.raises(ValueError, match="--seeds"):
        module.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize("resume", [False, True])
def test_addition_launcher_checks_every_directory_before_launch(tmp_path, monkeypatch, resume):
    module = load_script()
    args = arguments(tmp_path, [0])
    args.resume = resume
    for seed in args.seeds:
        directory = args.output / f"seed-{seed}"
        directory.mkdir(parents=True)
        if resume and seed != args.seeds[-1]:
            (directory / "checkpoint-latest.pt").write_bytes(b"checkpoint")
    log = args.output / f"seed-{args.seeds[-1]}" / "train.log"
    log.write_text("existing log\n")

    def forbidden_launch(*args, **kwargs):
        pytest.fail("all directories must be checked before launching any fit")

    monkeypatch.setattr(module.subprocess, "Popen", forbidden_launch)
    with pytest.raises(FileNotFoundError if resume else FileExistsError):
        module.run(args)
    assert log.read_text() == "existing log\n"
    assert not (args.output / "seed-11" / "train.log").exists()


def test_addition_launcher_appends_log_on_resume(tmp_path, monkeypatch):
    module = load_script()
    args = arguments(tmp_path, [0])
    args.seeds = [11]
    args.resume = True
    directory = args.output / "seed-11"
    directory.mkdir(parents=True)
    (directory / "checkpoint-latest.pt").write_bytes(b"checkpoint")
    log = directory / "train.log"
    log.write_text("previous output\n")

    def launch(command, *, stdout, **kwargs):
        assert command[-1] == "--resume"
        stdout.write("resumed output\n")
        return Process()

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    module.run(args)
    assert log.read_text() == "previous output\nresumed output\n"


@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
def test_addition_launcher_cleans_up_after_partial_launch(tmp_path, monkeypatch, error):
    module = load_script()
    worker = Process()
    logs = []

    def launch(command, *, stdout, **kwargs):
        logs.append(stdout)
        if len(logs) == 2:
            raise error("launch interrupted")
        return worker

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    with pytest.raises(error):
        module.run(arguments(tmp_path, [0, 1]))
    assert worker.terminated
    assert not worker.killed
    assert all(log.closed for log in logs)


def test_addition_launcher_cleans_up_when_wait_is_interrupted(tmp_path, monkeypatch):
    module = load_script()
    logs = []

    class InterruptedProcess(Process):
        def wait(self, timeout=None):
            if not self.terminated:
                raise KeyboardInterrupt
            return super().wait(timeout)

    workers = [InterruptedProcess(), Process()]

    def launch(command, *, stdout, **kwargs):
        logs.append(stdout)
        return workers[len(logs) - 1]

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    with pytest.raises(KeyboardInterrupt):
        module.run(arguments(tmp_path, [0, 1]))
    assert all(worker.terminated for worker in workers)
    assert all(log.closed for log in logs)


def test_addition_launcher_kills_worker_after_cleanup_timeout(tmp_path, monkeypatch):
    module = load_script()
    logs = []

    class UnresponsiveProcess(Process):
        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            if self.returncode is None and timeout is not None:
                raise module.subprocess.TimeoutExpired("worker", timeout)
            return super().wait(timeout)

    worker = UnresponsiveProcess()

    def launch(command, *, stdout, **kwargs):
        logs.append(stdout)
        if len(logs) == 2:
            raise OSError("launch failed")
        return worker

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    with pytest.raises(OSError, match="launch failed"):
        module.run(arguments(tmp_path, [0, 1]))
    assert worker.terminated
    assert worker.killed
    assert all(log.closed for log in logs)
