import datetime
import os
import subprocess
import re
import tempfile
from typing import Literal, Optional
from pydantic import BaseModel, Field, validator
from simple_slurm import Slurm
from utilix import logger

USER = os.environ.get("USER")
if USER is None:
    raise ValueError("USER environment variable is not set")

SCRATCH_DIR = os.environ.get("SCRATCH", ".")
# SCRATCH_DIR must have write permission
if not os.access(SCRATCH_DIR, os.W_OK):
    raise ValueError(
        f"SCRATCH_DIR {SCRATCH_DIR} does not have write permission. You may need to set SCRATCH_DIR manually in your .bashrc or .bash_profile."
    )

PARTITIONS = ["dali", "lgrandi", "xenon1t", "broadwl", "kicp", "caslake"]
TMPDIR = {
    "dali": os.path.expanduser(f"/dali/lgrandi/{USER}/tmp"),
    "lgrandi": os.path.join(SCRATCH_DIR, "tmp"),
    "xenon1t": os.path.join(SCRATCH_DIR, "tmp"),
    "broadwl": os.path.join(SCRATCH_DIR, "tmp"),
    "kicp": os.path.join(SCRATCH_DIR, "tmp"),
    "caslake": os.path.join(SCRATCH_DIR, "tmp"),
}
SINGULARITY_DIR = {
    "dali": "/dali/lgrandi/xenonnt/singularity-images",
    "lgrandi": "/project2/lgrandi/xenonnt/singularity-images",
    "xenon1t": "/project2/lgrandi/xenonnt/singularity-images",
    "broadwl": "/project2/lgrandi/xenonnt/singularity-images",
    "kicp": "/project2/lgrandi/xenonnt/singularity-images",
    "caslake": "/project2/lgrandi/xenonnt/singularity-images",
}
DEFAULT_BIND = [
    "/project2/lgrandi/xenonnt/dali:/dali",
    "/project2",
    "/project",
    "/scratch/midway2/%s" % (USER),
    "/scratch/midway3/%s" % (USER),
]
DALI_BIND = [
    "/dali/lgrandi",
    "/dali/lgrandi/xenonnt/xenon.config:/project2/lgrandi/xenonnt/xenon.config",
    "/dali/lgrandi/grid_proxy/xenon_service_proxy:/project2/lgrandi/grid_proxy/xenon_service_proxy",
]


def _make_executable(path: str) -> None:
    """
    Make a file executable by the user, group and others.

    Args:
        path (str): Path to the file to make executable.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist")
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2
    os.chmod(path, mode)


def _get_qos_list() -> list[str]:
    """
    Get the list of available qos.

    Returns:
        list[str]: The list of available qos.
    """
    cmd = "sacctmgr show qos format=name -p"
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=True)
        qos_list = result.stdout.strip().split("\n")
        qos_list = [qos[:-1] for qos in qos_list]
        return qos_list
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing sacctmgr: {e}")
        return []


class JobSubmission(BaseModel):
    jobstring: str = Field(..., description="The command to execute")
    log: str = Field("job.log", description="Where to store the log file of the job")
    partition: Literal["dali", "lgrandi", "xenon1t", "broadwl", "kicp", "caslake"] = Field(
        "xenon1t", description="Partition to submit the job to"
    )
    qos: str = Field("xenon1t", description="QOS to submit the job to")
    account: str = Field("pi-lgrandi", description="Account to submit the job to")
    jobname: str = Field("somejob", description="How to name this job")
    sbatch_file: Optional[str] = Field(None, description="Deprecated")
    dry_run: bool = Field(
        False, description="Only print how the job looks like, without submitting"
    )
    mem_per_cpu: int = Field(1000, description="MB requested for job")
    container: str = Field(
        "xenonnt-development.simg", description="Name of the container to activate"
    )
    bind: list[str] = Field(
        default_factory=lambda: DEFAULT_BIND,
        description="Paths to add to the container. Immutable when specifying dali as partition",
    )
    cpus_per_task: int = Field(1, description="CPUs requested for job")
    hours: Optional[float] = Field(None, description="Max hours of a job")
    node: Optional[str] = Field(None, description="Define a certain node to submit your job")
    exclude_nodes: Optional[str] = Field(
        None, description="Define a list of nodes which should be excluded from submission"
    )
    dependency: Optional[str] = Field(
        None, description="Provide list of job ids to wait for before running this job"
    )
    verbose: bool = Field(False, description="Print the sbatch command before submitting")

    @validator("bind", pre=True, each_item=True)
    def check_bind(cls, v) -> str:
        if not isinstance(v, str):
            raise ValueError("Each bind must be a string")

        if not os.path.exists(v):
            raise FileNotFoundError(f"Bind path {v} does not exist")

        return v

    @validator("partition", pre=True, always=True)
    def overwrite_for_dali(cls, v, values) -> str:
        # You can access other fields in the model using the "values" dict
        if v == "dali":
            bind = DALI_BIND
            values["bind"] = bind
            logger.warning(f"Binds are overwritten to {bind}")
            # If log path top level is not /dali
            abs_log_path = os.path.abspath(values["log"])
            if not abs_log_path.startswith("/dali"):
                log_filename = os.path.basename(abs_log_path)
                new_log_path = f"{TMPDIR['dali']}/{log_filename}"
                values["log"] = new_log_path
                print("Your log is relocated at: %s" % (new_log_path))
                logger.warning(f"Log path is overwritten to {new_log_path}")
        return v

    @validator("qos", pre=True, always=True)
    def check_qos(cls, v) -> str:
        qos_list = _get_qos_list()
        if v not in qos_list:
            logger.warning(
                f'QOS {v} is not in the list of available qos: {qos_list}, using "normal"'
            )
            return "normal"
        return v

    @validator("hours")
    def check_hours_value(cls, v) -> Optional[float]:
        if v is not None and (v <= 0 or v > 72):
            raise ValueError("Hours must be between 0 and 72")
        return v

    @validator("node", "exclude_nodes", "dependency")
    def check_node_format(cls, v) -> Optional[str]:
        if v is not None and not re.match(r"^[a-zA-Z0-9,\[\]-]+$", v):
            raise ValueError("Invalid format for node/exclude_nodes/dependency")
        return v

    @validator("container")
    def check_container_format(cls, v, values) -> str:
        if not v.endswith(".simg"):
            raise ValueError("Container must end with .simg")
        # Check if the container exists
        partition = values.get("partition")
        if not os.path.exists(os.path.join(SINGULARITY_DIR[partition], v)):
            raise FileNotFoundError(
                f"Singularity image {v} does not exist in {SINGULARITY_DIR[partition]}"
            )
        return v

    @validator("sbatch_file")
    def check_sbatch_file(cls, v) -> Optional[str]:
        if v is not None:
            logger.warning("sbatch_file is deprecated")
        return v

    def _create_singularity_jobstring(self) -> str:
        """
        Wrap the jobstring with the singularity command.

        Raises:
            FileNotFoundError: If the singularity image does not exist.

        Returns:
            str: The new jobstring with the singularity command.
        """
        if self.dry_run:
            file_discriptor = None
            exec_file = f"{TMPDIR[self.partition]}/tmp.sh"
        else:
            file_discriptor, exec_file = tempfile.mkstemp(suffix=".sh", dir=TMPDIR[self.partition])
            _make_executable(exec_file)
            os.write(file_discriptor, bytes("#!/bin/bash\n" + self.jobstring, "utf-8"))
        bind_string = " ".join([f"--bind {b}" for b in self.bind])
        image = os.path.join(SINGULARITY_DIR[self.partition], self.container)
        if not os.path.exists(image):
            raise FileNotFoundError(f"Singularity image {image} does not exist")
        # Warn user if CUTAX_LOCATION is unset due to INSTALL_CUTAX
        if os.environ.get("INSTALL_CUTAX") == "1":
            logger.warning(
                "INSTALL_CUTAX is set to 1, ignoring CUTAX_LOCATION and unsetting it for the job."
            )
        new_job_string = (
            f"unset X509_CERT_DIR\n"
            f'if [ "$INSTALL_CUTAX" == "1" ]; then unset CUTAX_LOCATION; fi\n'
            f"module load singularity\n"
            f"singularity exec {bind_string} {image} {exec_file}\n"
            f"exit_code=$?\n"
            f"rm {exec_file}\n"
            f"if [ $exit_code -ne 0 ]; then\n"
            f"    echo Python script failed with exit code $exit_code\n"
            f"    exit $exit_code\n"
            f"fi\n"
        )
        if file_discriptor is not None:
            os.close(file_discriptor)
        return new_job_string

    def submit(self):
        """
        Submit the job to the SLURM queue.
        """
        os.makedirs(TMPDIR[self.partition], exist_ok=True)
        # Initialize a dictionary with mandatory parameters
        slurm_params = {
            "job_name": self.jobname,
            "output": self.log,
            "qos": self.qos,
            "error": self.log,
            "account": self.account,
            "partition": self.partition,
            "mem_per_cpu": self.mem_per_cpu,
            "cpus_per_task": self.cpus_per_task,
        }

        # Conditionally add optional parameters if they are not None
        if self.hours is not None:
            slurm_params["time"] = datetime.timedelta(hours=self.hours)
        if self.node is not None:
            slurm_params["nodelist"] = self.node
        if self.exclude_nodes is not None:
            slurm_params["exclude"] = self.exclude_nodes
        if self.dependency is not None:
            slurm_params["dependency"] = {"afterok": self.dependency}
            slurm_params["kill_on_invalid"] = "yes"

        # Create the Slurm instance with the conditional arguments
        slurm = Slurm(**slurm_params)

        # Process the jobstring with the container if specified
        if self.container is not None:
            self.jobstring = self._create_singularity_jobstring()
        elif self.verbose:
            print(f"No container specified, running job as is")

        # Add the job command
        slurm.add_cmd(self.jobstring)

        print(f"Your log is located at: {self.log}")

        # Handle dry run scenario
        if self.verbose or self.dry_run:
            print(f"Generated slurm script:\n{slurm.script()}")

        if self.dry_run:
            return
        # Submit the job
        slurm.sbatch()


def submit_job(*args, **kwargs):
    """
    Adapter to old function name.
    You should use JobSubmission to get modern code editor support.

    Args:
        **kwargs: Keyword arguments to pass to JobSubmission
    """
    logger.warning(
        "Using legacy function name, please use JobSubmission to get modern code editor support"
    )
    job = JobSubmission(*args, **kwargs)
    job.submit()


def count_jobs(string="") -> int:
    """
    Count the number of jobs in the queue.

    Args:
        string (str, optional): String to search for in the job names. Defaults to "".

    Returns:
        int: Number of jobs in the queue.
    """
    output = subprocess.check_output(["squeue", "-u", str(USER)]).decode("utf-8")
    lines = output.split("\n")
    return len([job for job in lines if string in job])
