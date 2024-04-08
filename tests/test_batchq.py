import os
from typing import Dict, Any
from copy import deepcopy
from pydantic import ValidationError
from utilix.batchq import JobSubmission, QOSNotFoundError, FormatError
import pytest
from unittest.mock import patch

import os
from typing import Dict, Any
from pydantic import ValidationError
from utilix.batchq import JobSubmission, QOSNotFoundError
import pytest
from unittest.mock import patch

# Fixture to provide a sample valid JobSubmission instance
@pytest.fixture
def valid_job_submission() -> JobSubmission:
    return JobSubmission(
        jobstring="Hello World",
        qos="xenon1t",
        hours=10,
        container="xenonnt-development.simg",
    )

def test_valid_jobstring(valid_job_submission: JobSubmission):
    """ Test case to check if a valid jobstring is accepted. """
    assert valid_job_submission.jobstring == "Hello World"

def test_invalid_qos():
    """ Test case to check if the appropriate validation error is raised when an invalid value is provided for the qos field. """
    with pytest.raises(QOSNotFoundError) as exc_info:
        JobSubmission(jobstring="Hello World", qos="invalid_qos", hours=10, container="xenonnt-development.simg")
    assert "QOS invalid_qos is not in the list of available qos" in str(exc_info.value)

def test_valid_qos(valid_job_submission: JobSubmission):
    """ Test case to check if a valid qos is accepted. """
    assert valid_job_submission.qos == "xenon1t"

def test_invalid_hours():
    """ Test case to check if the appropriate validation error is raised when an invalid value is provided for the hours field. """
    with pytest.raises(ValidationError) as exc_info:
        JobSubmission(jobstring="Hello World", qos="xenon1t", hours=100, container="xenonnt-development.simg")
    assert "Hours must be between 0 and 72" in str(exc_info.value)

def test_valid_hours(valid_job_submission: JobSubmission):
    """ Test case to check if a valid hours value is accepted. """
    assert valid_job_submission.hours == 10

def test_invalid_container():
    """ Test case to check if the appropriate validation error is raised when an invalid value is provided for the container field. """
    with pytest.raises(FormatError) as exc_info:
        JobSubmission(jobstring="Hello World", qos="xenon1t", hours=10, container="invalid.ext")
    assert "Container must end with .simg" in str(exc_info.value)

def test_valid_container(valid_job_submission: JobSubmission):
    """ Test case to check if a valid container value is accepted. """
    assert valid_job_submission.container == "xenonnt-development.simg"

def test_container_exists(valid_job_submission: JobSubmission, tmp_path: str):
    """ Test case to check if the appropriate validation error is raised when the specified container does not exist. """
    with patch.dict("utilix.batchq.SINGULARITY_DIR", {"xenon1t": str(tmp_path)}):
        with pytest.raises(FileNotFoundError) as exc_info:
            JobSubmission(**valid_job_submission.dict())
        assert "Singularity image xenonnt-development.simg does not exist" in str(exc_info.value)