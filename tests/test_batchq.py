from pydantic import ValidationError
from utilix import batchq
from utilix.batchq import JobSubmission, QOSNotFoundError, FormatError, submit_job
import pytest
import os
from unittest.mock import patch
import inspect
import logging


# Get the SERVER type
def get_server_type():
    hostname = os.uname().nodename
    if "midway2" in hostname:
        return "Midway2"
    elif "midway3" in hostname:
        return "Midway3"
    elif "dali" in hostname:
        return "Dali"
    else:
        raise ValueError(
            f"Unknown server type for hostname {hostname}. Please use midway2, midway3, or dali."
        )


SERVER = get_server_type()


def get_partition_and_qos(server):
    if server == "Midway2":
        return "xenon1t", "xenon1t"
    elif server == "Midway3":
        return "lgrandi", "lgrandi"
    elif server == "Dali":
        return "dali", "dali"
    else:
        raise ValueError(f"Unknown server: {server}")


PARTITION, QOS = get_partition_and_qos(SERVER)


# Fixture to provide a sample valid JobSubmission instance
@pytest.fixture
def valid_job_submission() -> JobSubmission:
    return JobSubmission(
        jobstring="Hello World",
        partition=PARTITION,
        qos=QOS,
        hours=10,
        container="xenonnt-development.simg",
    )


def test_valid_jobstring(valid_job_submission: JobSubmission):
    """Test case to check if a valid jobstring is accepted."""
    assert valid_job_submission.jobstring == "Hello World"


def test_valid_container(valid_job_submission: JobSubmission):
    """Test case to check if a valid path for the container is found."""
    assert "xenonnt-development.simg" in valid_job_submission.container


def test_container_exists(valid_job_submission: JobSubmission, tmp_path: str):
    """
    Test case to check if the appropriate validation error is raised when the specified container does not exist.
    """
    invalid_container = "nonexistent-container.simg"
    with patch.object(batchq, "SINGULARITY_DIR", "/lgrandi/xenonnt/singularity-images"):
        with pytest.raises(FileNotFoundError) as exc_info:
            JobSubmission(
                **valid_job_submission.dict(exclude={"container"}),
                container=invalid_container,
            )
        assert f"Container {invalid_container} does not exist" in str(exc_info.value)


def test_invalid_container(valid_job_submission: JobSubmission):
    """Test case to check if the appropriate validation error is raised when an invalid value is provided for the container field."""
    job_submission_data = valid_job_submission.dict().copy()
    job_submission_data["container"] = "invalid.txt"
    with pytest.raises(FormatError) as exc_info:
        job_submission = JobSubmission(**job_submission_data)
    assert "Container must end with .simg" in str(exc_info.value)


def test_invalid_qos(valid_job_submission: JobSubmission):
    """Test case to check if the appropriate validation error is raised when an invalid value is provided for the qos field."""
    job_submission_data = valid_job_submission.dict().copy()
    job_submission_data["qos"] = "invalid_qos"
    with pytest.raises(QOSNotFoundError) as exc_info:
        JobSubmission(**job_submission_data)
    assert "QOS invalid_qos is not in the list of available qos" in str(exc_info.value)


def test_valid_qos(valid_job_submission: JobSubmission):
    """Test case to check if a valid qos is accepted."""
    assert valid_job_submission.qos == valid_job_submission.qos


def test_invalid_bind(valid_job_submission: JobSubmission, caplog):
    """Test case to check if the appropriate validation error is raised when an invalid value is provided for the bind field."""
    job_submission_data = valid_job_submission.dict().copy()
    invalid_bind = "/project999"
    job_submission_data["bind"].append(invalid_bind)
    with caplog.at_level(logging.WARNING):
        JobSubmission(**job_submission_data)

    assert "skipped mounting" in caplog.text
    assert invalid_bind in caplog.text


def test_invalid_hours(valid_job_submission: JobSubmission):
    """Test case to check if the appropriate validation error is raised when an invalid value is provided for the hours field."""
    job_submission_data = valid_job_submission.dict().copy()
    job_submission_data["hours"] = 1000
    with pytest.raises(ValidationError) as exc_info:
        JobSubmission(**job_submission_data)
    assert "Hours must be between 0 and 72" in str(exc_info.value)


def test_valid_hours(valid_job_submission: JobSubmission):
    """Test case to check if a valid hours value is accepted."""
    assert valid_job_submission.hours == valid_job_submission.hours


def test_bypass_validation_qos(valid_job_submission: JobSubmission):
    """
    Test case to check if the validation for the qos field is skipped when it is included in the bypass_validation list.
    """
    job_submission_data = valid_job_submission.dict().copy()
    job_submission_data["qos"] = "invalid_qos"
    job_submission_data["bypass_validation"] = ["qos"] + job_submission_data.get(
        "bypass_validation", []
    )
    job_submission = JobSubmission(**job_submission_data)
    assert job_submission.qos == "invalid_qos"


def test_bypass_validation_hours(valid_job_submission: JobSubmission):
    """
    Test case to check if the validation for the hours field is skipped when it is included in the bypass_validation list.
    """
    job_submission_data = valid_job_submission.dict().copy()
    job_submission_data["hours"] = 100
    job_submission_data["bypass_validation"] = ["hours"] + job_submission_data.get(
        "bypass_validation", []
    )
    job_submission = JobSubmission(**job_submission_data)
    assert job_submission.hours == 100


def test_bypass_validation_container(valid_job_submission: JobSubmission):
    """
    Test case to check if the validation for the container field is skipped when it is included in the bypass_validation list.
    """
    job_submission_data = valid_job_submission.dict().copy()
    job_submission_data["container"] = "invalid.ext"
    job_submission_data["bypass_validation"] = ["container"] + job_submission_data.get(
        "bypass_validation", []
    )
    job_submission = JobSubmission(**job_submission_data)
    assert job_submission.container == "invalid.ext"


def test_bypass_validation_multiple_fields(valid_job_submission: JobSubmission):
    """
    Test case to check if the validation for multiple fields is skipped when they are included in the bypass_validation list.
    """
    job_submission_data = valid_job_submission.dict().copy()
    job_submission_data["qos"] = "invalid_qos"
    job_submission_data["hours"] = 100
    job_submission_data["container"] = "invalid.ext"
    job_submission_data["bypass_validation"] = [
        "qos",
        "hours",
        "container",
    ] + job_submission_data.get("bypass_validation", [])
    job_submission = JobSubmission(**job_submission_data)
    assert job_submission.qos == "invalid_qos"
    assert job_submission.hours == 100
    assert job_submission.container == "invalid.ext"


# Check if all of the possible arguments are handled correctly
def test_submit_job_arguments():
    submit_job_params = inspect.signature(submit_job).parameters
    job_submission_fields = JobSubmission.__fields__

    missing_params = []
    for field_name in job_submission_fields:
        if field_name not in submit_job_params:
            missing_params.append(field_name)

    assert (
        len(missing_params) == 0
    ), f"Missing parameters in submit_job: {', '.join(missing_params)}"

    