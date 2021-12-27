import subprocess
import os
import tempfile
import shlex
from utilix import logger

sbatch_template = """#!/bin/bash

#SBATCH --job-name={jobname}
#SBATCH --output={log}
#SBATCH --error={log}
#SBATCH --account=pi-lgrandi
#SBATCH --qos={qos}
#SBATCH --partition={partition}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --cpus-per-task={cpus_per_task}
{hours}

{job}
"""

SINGULARITY_DIR = '/project2/lgrandi/xenonnt/singularity-images'
TMPDIR = os.path.join(os.environ.get('SCRATCH', '.'), 'tmp')


def make_executable(path):
    """Make the file at path executable, see """
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)


def singularity_wrap(jobstring, image, bind):
    """Wraps a jobscript into another executable file that can be passed to singularity exec"""
    _, exec_file = tempfile.mkstemp(suffix='.sh', dir=TMPDIR)
    make_executable(exec_file)

    bind_string = " ".join([f"--bind {b}" for b in bind])
    image = os.path.join(SINGULARITY_DIR, image)
    new_job_string = f"""cat > {exec_file} << EOF
#!/bin/bash
{jobstring}
EOF
singularity exec {bind_string} {image} {exec_file}
rm {exec_file}
"""
    return new_job_string


def submit_job(jobstring,
               log='job.log',
               partition='xenon1t',
               qos='xenon1t',
               account='pi-lgrandi',
               jobname='somejob',
               sbatch_file=None,
               dry_run=False,
               mem_per_cpu=1000,
               container='xenonnt-development.simg',
               bind=('/dali', '/project2', os.path.dirname(TMPDIR)),
               cpus_per_task=1,
               hours=None,
               **kwargs
               ):
    """
    Submit a job to the dali batch queue

    EXAMPLE
        from utilix import batchq
        import time

        job_log = 'job.log'
        batchq.submit_job('echo "say hi"', log=job_log)

        time.sleep(10) # Allow the job to run
        for line in open(job_log):
            print(line)

    :param jobstring: the command to execute
    :param log: where to store the log file of the job
    :param partition: partition to submit the job to
    :param qos: qos to submit the job to
    :param account: account to submit the job to
    :param jobname: how to name this job
    :param sbatch_file: where to write the job script to
    :param dry_run: only print how the job looks like
    :param mem_per_cpu: mb requested for job
    :param container: name of the container to activate
    :param bind: which paths to add to the container
    :param cpus_per_task: cpus requested for job
    :param hours: max hours of a job
    :param kwargs: are ignored
    :return: None
    """
    if 'delete_file' in kwargs:
        logger.warning('"delete_file" option for "submit_job" has been removed, ignoring for now')
    os.makedirs(TMPDIR, exist_ok=True)

    if container:
        # need to wrap job into another executable
        _, exec_file = tempfile.mkstemp(suffix='.sh')
        jobstring = singularity_wrap(jobstring, container, bind)
        jobstring = 'unset X509_CERT_DIR\n' + 'module load singularity\n' + jobstring

    if not hours is None:
        hours = '#SBATCH --time={:02d}:{:02d}:{:02d}'.format(int(hours), int(hours * 60 % 60), int(hours * 60 % 60 * 60 % 60))
    else:
        hours = ''
    sbatch_script = sbatch_template.format(jobname=jobname, log=log, qos=qos, partition=partition,
                                           account=account, job=jobstring, mem_per_cpu=mem_per_cpu,
                                           cpus_per_task=cpus_per_task, hours=hours)

    if dry_run:
        print("=== DRY RUN ===")
        print(sbatch_script)
        return

    if sbatch_file is None:
        remove_file = True
        _, sbatch_file = tempfile.mkstemp(suffix='.sbatch')
    else:
        remove_file = False

    with open(sbatch_file, 'w') as f:
        f.write(sbatch_script)

    command = "sbatch %s" % sbatch_file
    if not sbatch_file:
        print("Executing: %s" % command)
    subprocess.Popen(shlex.split(command)).communicate()

    if remove_file:
        os.remove(sbatch_file)


def count_jobs(string=''):
    username = os.environ.get("USER")
    output = subprocess.check_output(shlex.split("squeue -u %s" % username))
    lines = output.decode('utf-8').split('\n')
    return len([job for job in lines if string in job])

