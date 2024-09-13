"""This module provides utilities to generate and submit jobs to the SLURM queue."""

import datetime
import os
import subprocess
import re
import tempfile
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, validator
from simple_slurm import Slurm  # type: ignore
from . import logger

USER: Optional[str] = os.environ.get("USER")
if USER is None:
    raise ValueError("USER environment variable is not set")

SCRATCH_DIR: str = os.environ.get("SCRATCH", ".")
# SCRATCH_DIR must have write permission
if not os.access(SCRATCH_DIR, os.W_OK):
    raise ValueError(
        f"SCRATCH_DIR {SCRATCH_DIR} does not have write permission. "
        "You may need to set SCRATCH_DIR manually in your .bashrc or .bash_profile."
    )

PARTITIONS: List[str] = [
    "dali",
    "lgrandi",
    "xenon1t",
    "broadwl",
    "kicp",
    "caslake",
    "build",
    "bigmem2",
    "gpu2",
]
TMPDIR: Dict[str, str] = {
    "dali": f"/dali/lgrandi/{USER}/tmp",
    "lgrandi": os.path.join(SCRATCH_DIR, "tmp"),
    "xenon1t": os.path.join(SCRATCH_DIR, "tmp"),
    "broadwl": os.path.join(SCRATCH_DIR, "tmp"),
    "kicp": os.path.join(SCRATCH_DIR, "tmp"),
    "caslake": os.path.join(SCRATCH_DIR, "tmp"),
    "build": os.path.join(SCRATCH_DIR, "tmp"),
    "bigmem2": os.path.join(SCRATCH_DIR, "tmp"),
    "gpu2": os.path.join(SCRATCH_DIR, "tmp"),
}

SINGULARITY_DIR: str = "lgrandi/xenonnt/singularity-images"

DEFAULT_BIND: List[str] = [
    "/project2/lgrandi/xenonnt/dali:/dali",
    "/project2",
    "/project",
    f"/scratch/midway2/{USER}",
    f"/scratch/midway3/{USER}",
]
DALI_BIND: List[str] = [
    "/dali/lgrandi",
    "/dali/lgrandi/xenonnt/xenon.config:/project2/lgrandi/xenonnt/xenon.config",
    "/dali/lgrandi/grid_proxy/xenon_service_proxy:/project2/lgrandi/grid_proxy/xenon_service_proxy",
]


class QOSNotFoundError(Exception):
    """Provided qos is not found in the qos list."""


class FormatError(Exception):
    """Format of file is not correct."""


def _make_executable(path: str) -> None:
    """Make a file executable by the user, group and others.

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


def _get_qos_list() -> List[str]:
    """Get the list of available qos.

    Returns:
        List[str]: The list of available qos.

    """
    cmd = "sacctmgr show qos format=name -p"
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=True)
        qos_list: List[str] = result.stdout.strip().split("\n")
        qos_list = [qos[:-1] for qos in qos_list]
        return qos_list
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing sacctmgr: {e}")
        return []


class JobSubmission(BaseModel):
    """Class to generate and submit a job to the SLURM queue."""

    jobstring: str = Field(..., description="The command to execute")
    bypass_validation: List[str] = Field(
        default_factory=list, description="List of parameters to bypass validation for"
    )
    exclude_lc_nodes: bool = Field(False, description="Exclude the loosely coupled nodes")
    log: str = Field("job.log", description="Where to store the log file of the job")
    bind: List[str] = Field(
        default_factory=lambda: DEFAULT_BIND,
        description="Paths to add to the container. Immutable when specifying dali as partition",
    )
    partition: Literal[
        "dali", "lgrandi", "xenon1t", "broadwl", "kicp", "caslake", "build", "bigmem2", "gpu2"
    ] = Field("xenon1t", description="Partition to submit the job to")
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
    cpus_per_task: int = Field(1, description="CPUs requested for job")
    hours: Optional[float] = Field(None, description="Max hours of a job")
    node: Optional[str] = Field(None, description="Define a certain node to submit your job")
    exclude_nodes: Optional[str] = Field(
        None,
        description="Define a list of nodes which should be excluded from submission",
    )
    dependency: Optional[str] = Field(
        None, description="Provide list of job ids to wait for before running this job"
    )
    verbose: bool = Field(False, description="Print the sbatch command before submitting")

    # Check if there is any positional argument which is not allowed
    def __new__(cls, *args, **kwargs):
        if args:
            raise ValueError(
                "Positional arguments are not allowed. "
                "Please use keyword arguments for all fields."
            )
        return super().__new__(cls)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def _skip_validation(cls, field: str, values: Dict[Any, Any]) -> bool:
        """Check if a field should be validated based on the bypass_validation list.

        Args:
            field (str): The name of the field to check.
            values (Dict[str, Any]): The values dictionary containing the bypass_validation list.

        Returns:
            bool: True if the field should be validated, False otherwise.

        """
        return field in values.get("bypass_validation", [])

    # validate the bypass_validation so that it can be reached in values
    @validator("bypass_validation", pre=True, each_item=True)
    def check_bypass_validation(cls, v: list) -> list:
        return v

    @validator("bind", pre=True)
    def check_bind(cls, v: list, values: Dict[Any, Any]) -> list:
        """Check if the bind path exists.

        Args:
            v (str): The bind path to check.

        Returns:
            str: The bind path if it exists.

        """
        if cls._skip_validation("bind", values):
            return v

        valid_bind = []
        invalid_bind = []
        for path in v:
            if ":" in path:
                actual_path = path.split(":")[0]
            else:
                actual_path = path
            if os.path.exists(actual_path):
                valid_bind.append(path)
            else:
                invalid_bind.append(path)
        if len(invalid_bind) > 0:
            logger.warning("Invalid bind paths: %s, skipped mounting", invalid_bind)
        return valid_bind

    @validator("partition", pre=True, always=True)
    def overwrite_for_dali(cls, v: str, values: Dict[Any, Any]) -> str:
        """Overwrite the partition to dali if the container is xenonnt-development.simg.

        Args:
            v (str): The partition to check.
            values (dict): The values of the model.

        Returns:
            str: The partition to use.

        """
        if cls._skip_validation("partition", values):
            return v
        if v == "dali":
            bind = DALI_BIND
            values["bind"] = bind
            logger.warning("Binds are overwritten to %s", bind)
            # If log path top level is not /dali
            abs_log_path = os.path.abspath(values["log"])
            if not abs_log_path.startswith("/dali"):
                log_filename = os.path.basename(abs_log_path)
                new_log_path = f"{TMPDIR['dali']}/{log_filename}"
                values["log"] = new_log_path
                print(f"Your log is relocated at: {new_log_path}")
                logger.warning("Log path is overwritten to %s", new_log_path)
        return v

    @validator("qos", pre=True, always=True)
    def check_qos(cls, v: str, values: Dict[Any, Any]) -> str:
        """Check if the qos is in the list of available qos.

        Args:
            v (str): The qos to check.

        Returns:
            str: The qos to use.

        """
        if cls._skip_validation("qos", values):
            return v
        qos_list = _get_qos_list()
        if v not in qos_list:
            # Raise an error if the qos is not in the list of available qos
            raise QOSNotFoundError(f"QOS {v} is not in the list of available qos: \n {qos_list}")
        return v

    @validator("hours")
    def check_hours_value(cls, v: Optional[float], values: Dict[Any, Any]) -> Optional[float]:
        """Check if the hours are between 0 and 72.

        Args:
            v (Optional[float]): The hours to check.

        Raises:
            ValueError: If the hours are not between 0 and 72.

        Returns:
            Optional[float]: The hours to use.

        """
        if cls._skip_validation("hours", values):
            return v
        if v is not None and (v <= 0 or v > 72):
            raise ValueError("Hours must be between 0 and 72")
        return v

    @validator("node", "exclude_nodes", "dependency")
    def check_node_format(
        cls, v: Optional[str], values: Dict[Any, Any], field: str
    ) -> Optional[str]:
        """Check if the node, exclude_nodes and dependency have the correct format.

        Args:
            v (Optional[str]): The node, exclude_nodes or dependency to check.

        Raises:
            ValueError: If the node, exclude_nodes or dependency do not have the correct format.

        Returns:
            Optional[str]: The node, exclude_nodes or dependency to use.

        """
        if cls._skip_validation(field, values):
            return v
        if v is not None and not re.match(r"^[a-zA-Z0-9,\[\]-]+$", v):
            raise ValueError("Invalid format for node/exclude_nodes/dependency")
        return v

    @validator("container")
    def check_container_format(cls, v: str, values: Dict[Any, Any]) -> str:
        """Check if the container ends with .simg and if it exists.

        Args:
            v (str): Full path of the container or the name of the container file.
            values (Dict[Any, Any]): The values of the model.

        Raises:
            ValueError: Container not ending with .simg
            FileNotFoundError: Container does not exist.

        Returns:
            str: The container to use.

        """
        if cls._skip_validation("container", values) or os.path.exists(v):
            return v
        if not v.endswith(".simg"):
            raise FormatError("Container must end with .simg")
        # Search for the container when the full path is not provided
        partition: str = values.get("partition", "xenon1t")
        if partition == "dali":
            root_dir = ["/dali"]
        else:
            root_dir = ["/project2", "/project"]
        for root in root_dir:
            image_path = os.path.join(root, SINGULARITY_DIR, v)
            print("searched in", image_path)
            if os.path.exists(image_path):
                return image_path
        raise FileNotFoundError(f"Container {v} does not exist")

    @validator("sbatch_file")
    def check_sbatch_file(cls, v: Optional[str], values: Dict[Any, Any]) -> Optional[str]:
        """Check if the sbatch_file is None.

        Args:
            v (Optional[str]): The sbatch_file to check.

        Returns:
            Optional[str]: The sbatch_file to use.

        """
        if cls._skip_validation("sbatch_file", values):
            return v

        if v is not None:
            logger.warning("sbatch_file is deprecated")
        return v

    def _create_singularity_jobstring(self) -> str:
        """Wrap the jobstring with the singularity command.

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
        bind_string = " ".join(
            [f"--bind {b}" for b in self.bind]  # pylint: disable=not-an-iterable
        )
        # Warn user if CUTAX_LOCATION is unset due to INSTALL_CUTAX
        if os.environ.get("INSTALL_CUTAX") == "1":
            logger.warning(
                "INSTALL_CUTAX is set to 1, ignoring CUTAX_LOCATION and unsetting it for the job."
            )
        new_job_string = (
            f"echo running on $SLURMD_NODENAME\n"
            f"unset X509_CERT_DIR\n"
            f'if [ "$INSTALL_CUTAX" == "1" ]; then unset CUTAX_LOCATION; fi\n'
            f"module load singularity\n"
            f"singularity exec {bind_string} {self.container} {exec_file}\n"
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

    def _get_lc_nodes(self) -> List[str]:
        """Get the list of 'lc' (loosely coupled) nodes in the specified partition.

        Returns:
            List[str]: The list of 'lc' node names.

        """
        cmd = f"nodestatus {self.partition}"
        try:
            output = subprocess.check_output(cmd, universal_newlines=True, shell=True)
            lines = output.split("\n")
            lc_nodes = []
            for line in lines:
                columns = line.split()
                if len(columns) >= 4 and "," in columns[3]:
                    features = columns[3].split(",")
                    if "lc" in features:
                        lc_nodes.append(columns[0])
            return lc_nodes
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing nodestatus: {e}")
            return []

    def submit(self) -> None:
        """Submit the job to the SLURM queue."""
        os.makedirs(TMPDIR[self.partition], exist_ok=True)
        # Initialize a dictionary with mandatory parameters
        slurm_params: Dict[str, Any] = {
            "job_name": self.jobname,
            "output": self.log,
            "qos": self.qos,
            "error": self.log,
            "account": self.account,
            "partition": self.partition,
            "mem_per_cpu": self.mem_per_cpu,
            "cpus_per_task": self.cpus_per_task,
        }

        # Exclude the loosely coupled nodes if required
        if self.exclude_lc_nodes:
            lc_nodes = self._get_lc_nodes()
            if lc_nodes:
                slurm_params["exclude"] = ",".join(lc_nodes)

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
        self.jobstring = self._create_singularity_jobstring()

        # Add the job command
        slurm.add_cmd(self.jobstring)

        # Handle dry run scenario
        if self.verbose or self.dry_run:
            print(f"Generated slurm script:\n{slurm.script()}")

        if self.dry_run:
            return
        # Submit the job

        try:
            job_id = slurm.sbatch(shell="/bin/bash")
            if job_id:
                print(f"Job submitted successfully. Job ID: {job_id}")
                print(f"Your log is located at: {self.log}")
            else:
                print("Job submission failed.")
        except Exception as e:
            print(f"An error occurred while submitting the job: {str(e)}")


def submit_job(
    jobstring: str,
    exclude_lc_nodes: bool = False,
    log: str = "job.log",
    partition: Literal[
        "dali", "lgrandi", "xenon1t", "broadwl", "kicp", "caslake", "build", "bigmem2", "gpu2"
    ] = "xenon1t",
    qos: str = "xenon1t",
    account: str = "pi-lgrandi",
    jobname: str = "somejob",
    sbatch_file: Optional[str] = None,
    dry_run: bool = False,
    mem_per_cpu: int = 1000,
    container: str = "xenonnt-development.simg",
    bind: Optional[List[str]] = None,
    cpus_per_task: int = 1,
    hours: Optional[float] = None,
    node: Optional[str] = None,
    exclude_nodes: Optional[str] = None,
    dependency: Optional[str] = None,
    verbose: bool = False,
    bypass_validation: Optional[List[str]] = [],
) -> None:
    """Submit a job to the SLURM queue.

    Args:
        jobstring (str): The command to execute.
        exclude_lc_nodes (bool): Exclude the loosely coupled nodes. Default is True.
        log (str): Where to store the log file of the job. Default is "job.log".
        partition (Literal["dali", "lgrandi", "xenon1t", "broadwl", "kicp", "caslake", "build", "bigmem2", "gpu2" (the only GPU node)]):  # noqa
            Partition to submit the job to. Default is "xenon1t".
        qos (str): QOS to submit the job to. Default is "xenon1t".
        account (str): Account to submit the job to. Default is "pi-lgrandi".
        jobname (str): How to name this job. Default is "somejob".
        sbatch_file (Optional[str]): Deprecated. Default is None.
        dry_run (bool): Only print how the job looks like, without submitting. Default is False.
        mem_per_cpu (int): MB requested for job. Default is 1000.
        container (str): Name of the container to activate. Default is "xenonnt-development.simg".
        bind (List[str]): Paths to add to the container. Default is None.
        cpus_per_task (int): CPUs requested for job. Default is 1.
        hours (Optional[float]): Max hours of a job. Default is None.
        node (Optional[str]): Define a certain node to submit your job. Default is None.
        exclude_nodes (Optional[str]):
            Define a list of nodes which should be excluded from submission. Default is None.
        dependency (Optional[str]):
            Provide list of job ids to wait for before running this job. Default is None.
        verbose (bool): Print the sbatch command before submitting. Default is False.
        bypass_validation (List[str]): List of parameters to bypass validation for.
            Default is None.

    """
    if bind is None:
        bind = DEFAULT_BIND
    job = JobSubmission(
        jobstring=jobstring,
        exclude_lc_nodes=exclude_lc_nodes,
        log=log,
        partition=partition,
        qos=qos,
        account=account,
        jobname=jobname,
        sbatch_file=sbatch_file,
        dry_run=dry_run,
        mem_per_cpu=mem_per_cpu,
        container=container,
        bind=bind,
        cpus_per_task=cpus_per_task,
        hours=hours,
        node=node,
        exclude_nodes=exclude_nodes,
        dependency=dependency,
        verbose=verbose,
        bypass_validation=bypass_validation,
    )
    job.submit()


def count_jobs(string: str = "") -> int:
    """Count the number of jobs in the queue.

    Args:
        string (str, optional): String to search for in the job names. Defaults to "".

    Returns:
        int: Number of jobs in the queue.

    """
    output: str = subprocess.check_output(["squeue", "-u", str(USER)]).decode("utf-8")
    lines = output.split("\n")
    return len([job for job in lines if string in job])


def used_nodes() -> list[str]:
    """Get the list of nodes that are currently being used.

    Returns:
        List[str]: List of nodes that are currently being used.

    """
    output = subprocess.check_output(["squeue", "-o", "%R", "-u", str(USER)]).decode("utf-8")
    return list(set(output.split("\n")[1:-1]))
