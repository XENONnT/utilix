import subprocess
import os
import tempfile
import shlex
from utilix import logger
import getpass

sbatch_template = """#!/bin/bash

#SBATCH --job-name={jobname}
#SBATCH --output={log}
#SBATCH --error={log}
#SBATCH --account=pi-lgrandi
#SBATCH --qos={qos}
#SBATCH --partition={partition}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --cpus-per-task={cpus_per_task}
{node}
{exclude_nodes}
{hours}

{job}
"""

SINGULARITY_DIR = {
    'dali': '/dali/lgrandi/xenonnt/singularity-images',
    'lgrandi': '/project2/lgrandi/xenonnt/singularity-images',
    'xenon1t': '/project2/lgrandi/xenonnt/singularity-images',
    'broadwl': '/project2/lgrandi/xenonnt/singularity-images',
    'kicp': '/project2/lgrandi/xenonnt/singularity-images',
}

TMPDIR = {
    'dali': os.path.expanduser('/dali/lgrandi/%s/tmp'%(getpass.getuser())),
    'lgrandi': os.path.join(os.environ.get('SCRATCH', '.'), 'tmp'),
    'xenon1t': os.path.join(os.environ.get('SCRATCH', '.'), 'tmp'),
    'broadwl': os.path.join(os.environ.get('SCRATCH', '.'), 'tmp'),
    'kicp': os.path.join(os.environ.get('SCRATCH', '.'), 'tmp'),
}


def overwrite_dali_bind(bind, partition):
    """Check if we are binding non-dali storage when we are on dali compute node. If yes, then overwrite"""
    if partition == 'dali':
        bind = ['/dali', '/dali/lgrandi/xenonnt/xenon.config:/project2/lgrandi/xenonnt/xenon.config']
        print("You are using dali parition, and your bind has been fixed to %s"%(bind))
    return bind


def wrong_log_dir(path):
    """Check if the directory is NOT in dali"""
    abs_path = os.path.abspath(path)
    top_level = abs_path.split('/')[1] 
    wrong_log_dir = False   
    if top_level != 'dali':
        wrong_log_dir = True
    else:
        wrong_log_dir = False
    return wrong_log_dir


def overwrite_dali_job_log(path, partition):
    if partition == 'dali':
        if wrong_log_dir(path):
            logname = os.path.abspath(path).split('/')[-1]
            new_path = TMPDIR['dali']+'/'+logname
            print('Your log is relocated at: %s'%(new_path))
            return new_path
    else:
        print('Your log is located at: %s'%(os.path.abspath(path)))
        return path


def make_executable(path):
    """Make the file at path executable, see """
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)


def singularity_wrap(jobstring, image, bind, partition):
    """Wraps a jobscript into another executable file that can be passed to singularity exec"""
    file_descriptor, exec_file = tempfile.mkstemp(suffix='.sh', dir=TMPDIR[partition])
    make_executable(exec_file)
    os.write(file_descriptor, bytes('#!/bin/bash\n' + jobstring, 'utf-8'))
    bind_string = " ".join([f"--bind {b}" for b in bind])
    image = os.path.join(SINGULARITY_DIR[partition], image)
    new_job_string = f"""singularity exec {bind_string} {image} {exec_file}
rm {exec_file}
"""
    os.close(file_descriptor)
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
               bind=['/project2/lgrandi/xenonnt/dali:/dali', '/project2', '/project', '/scratch/midway2/%s'%(getpass.getuser()), '/scratch/midway3/%s'%(getpass.getuser())],
               cpus_per_task=1,
               hours=None,
               node=None,
               exclude_nodes=None,
               **kwargs
               ):
    """
    Submit a job to the dali/midway batch queue

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
    :param bind: which paths to add to the container. This is immutable when you specified dali as partition
    :param cpus_per_task: cpus requested for job
    :param hours: max hours of a job
    :param node: define a certain node to submit your job should be submitted to
    :param exclude_nodes: define a list of nodes which should be excluded from submission
    :param kwargs: are ignored
    :return: None
    """
    if 'delete_file' in kwargs:
        logger.warning('"delete_file" option for "submit_job" has been removed, ignoring for now')
    os.makedirs(TMPDIR[partition], exist_ok=True)
    # overwrite bind to make sure dali is isolated
    bind = overwrite_dali_bind(bind, partition)

    # overwrite log directory if it is not on dali and you are running on dali.
    log = overwrite_dali_job_log(log, partition)

    # temporary dirty fix. will remove these 3 from xenon1t soon.
    if partition == 'xenon1t' and exclude_nodes is None:
        exclude_nodes = 'dali028,dali029,dali030'

    if container:
        # need to wrap job into another executable
        jobstring = singularity_wrap(jobstring, container, bind, partition)
        jobstring = 'unset X509_CERT_DIR CUTAX_LOCATION\n' + 'module load singularity\n' + jobstring

    if not hours is None:
        hours = '#SBATCH --time={:02d}:{:02d}:{:02d}'.format(int(hours), int(hours * 60 % 60), int(hours * 60 % 60 * 60 % 60))
    else:
        hours = ''

    if not node is None:
        if not isinstance(node, str):
            raise ValueError(f'node should be str but given {type(node)}')
        node = '#SBATCH --nodelist={node}'.format(node=node)
    else:
        node = ''

    if not exclude_nodes is None:
        if not isinstance(exclude_nodes, str):
            raise ValueError(f'exclude_nodes should be str but given {type(exclude_nodes)}')
            # string like 'myCluster01,myCluster02,myCluster03' or 'myCluster[01-09]'
        exclude_nodes = '#SBATCH --exclude={exclude_nodes}'.format(exclude_nodes=exclude_nodes)
    else:
        exclude_nodes = ''

    sbatch_script = sbatch_template.format(jobname=jobname, log=log, qos=qos, partition=partition,
                                           account=account, job=jobstring, mem_per_cpu=mem_per_cpu,
                                           cpus_per_task=cpus_per_task, hours=hours, node=node,
                                           exclude_nodes=exclude_nodes)

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

