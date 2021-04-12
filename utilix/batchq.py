import subprocess
import os
import tempfile
import shlex

sbatch_template = """#!/bin/bash

#SBATCH --job-name={jobname}
#SBATCH --output={log}
#SBATCH --error={log}
#SBATCH --account=pi-lgrandi
#SBATCH --qos={qos}
#SBATCH --partition={partition}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --cpus-per-task={cpus_per_task}

{job}
"""

SINGULARITY_DIR = '/project2/lgrandi/xenonnt/singularity-images'
TMPDIR = os.path.join(os.environ.get('SCRATCH', '.'), 'tmp')


def make_executable(path):
    """Make the file at path executable, see """
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)


def singularity_wrap(jobstring, image, bind=('/dali', '/project2', TMPDIR)):
    """Wraps a jobscript into another executable file that can be passed to singularity exec"""
    _, exec_file = tempfile.mkstemp(suffix='.sh', dir=TMPDIR)
    make_executable(exec_file)

    bind_string = " ".join([f"--bind {b}" for b in bind])
    image = os.path.join(SINGULARITY_DIR, image)
    jobstring = 'unset X509_CERT_DIR\n' + jobstring
    new_job_string = f"""cat > {exec_file} << EOF
#!/bin/bash
{jobstring}
EOF
singularity exec {bind_string} {image} {exec_file}
rm {exec_file}
"""
    return new_job_string


def submit_job(jobstring, log='job.log', partition='xenon1t', qos='xenon1t',
               account='pi-lgrandi', jobname='somejob',
               delete_file=True, dry_run=False, mem_per_cpu=1000,
               container='xenonnt-development.simg',
               cpus_per_task=1):

    os.makedirs(TMPDIR, exist_ok=True)

    if container:
        # need to wrap job into another executable
        _, exec_file = tempfile.mkstemp(suffix='.sh')
        jobstring = singularity_wrap(jobstring, container)
        jobstring = 'module load singularity\n' + jobstring

    sbatch_script = sbatch_template.format(jobname=jobname, log=log, qos=qos, partition=partition,
                                           account=account, job=jobstring, mem_per_cpu=mem_per_cpu,
                                           cpus_per_task=cpus_per_task)

    if dry_run:
        print("=== DRY RUN ===")
        print(sbatch_script)
        return

    _, file = tempfile.mkstemp(suffix='.sbatch')
    with open(file, 'w') as f:
        f.write(sbatch_script)

    command = "sbatch %s" % file
    if not delete_file:
        print("Executing: %s" % command)
    subprocess.Popen(shlex.split(command)).communicate()

    if delete_file:
        os.remove(file)


def count_jobs(string=''):
    username = os.environ.get("USER")
    output = subprocess.check_output(shlex.split("squeue -u %s" % username))
    lines = output.decode('utf-8').split('\n')
    return len([job for job in lines if string in job])

