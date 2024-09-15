import subprocess
import os
import threading
import tempfile
import time


class Shell:
    """Provides a shell callout with buffered stdout/stderr, error handling and timeout."""

    def __init__(self, cmd, prefix=None, timeout_secs=1 * 60 * 60, log_cmd=False, log_outerr=False):
        self._cmd = cmd
        self._prefix = prefix if prefix else "shell"
        self._timeout_secs = timeout_secs
        self._log_cmd = log_cmd
        self._log_outerr = log_outerr
        self._process = None
        self._outerr = ""
        self._duration = 0.0

    def run(self):
        # temp file for the stdout/stderr
        _out_file = tempfile.TemporaryFile(prefix=f"{self._prefix}-", suffix=".out")

        def target():
            self._process = subprocess.Popen(
                self._cmd,
                shell=True,
                stdout=_out_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp,
            )
            self._process.communicate()

        if self._log_cmd:
            print(self._cmd)

        ts_start = time.time()

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(self._timeout_secs)
        if thread.is_alive():
            # do our best to kill the whole process group
            try:
                kill_cmd = f"kill -TERM -{os.getpgid(self._process.pid)}"
                kp = subprocess.Popen(kill_cmd, shell=True)
                kp.communicate()
                self._process.terminate()
            except Exception:
                pass
            thread.join()
            # log the output
            _out_file.seek(0)
            stdout = _out_file.read().decode("utf-8").strip()
            if self._log_outerr and len(stdout) > 0:
                print(stdout)
            _out_file.close()
            raise RuntimeError(f"Command timed out after {self._timeout_secs} seconds: {self._cmd}")

        self._duration = time.time() - ts_start

        # log the output
        _out_file.seek(0)
        self._outerr = _out_file.read().decode("utf-8").strip()
        if self._log_outerr and len(self._outerr) > 0:
            print(self._outerr)
        _out_file.close()

        if self._process.returncode != 0:
            raise RuntimeError(
                "Command exited with non-zero exit code "
                f"({self._process.returncode}): {self._cmd}\n{self._outerr}"
            )

    def get_outerr(self):
        """Returns the combined stdout and stderr from the command."""
        return self._outerr

    def get_exit_code(self):
        """Returns the exit code from the process."""
        return self._process.returncode

    def get_duration(self):
        """Returns the timing of the command (seconds)"""
        return self._duration
