import os
from . import logger
from .shell import Shell


def _validate_x509_proxy(min_valid_hours=20):
    """Ensure $X509_USER_PROXY exists and has enough time left.

    This is necessary only if you are going to use Rucio.

    """
    x509_user_proxy = os.getenv("X509_USER_PROXY")
    if not x509_user_proxy:
        raise RuntimeError("Please provide a valid X509_USER_PROXY environment variable.")

    shell = Shell(f"grid-proxy-info -timeleft -file {x509_user_proxy}", "outsource")
    shell.run()
    outerr = eval(shell.get_outerr())
    if outerr < 0:
        raise RuntimeError("Failed to get timeleft of X509_USER_PROXY.")
    valid_hours = outerr / 3600
    logger.info(f"X509_USER_PROXY is valid for {valid_hours:.2f} hours.")
    if valid_hours < min_valid_hours:
        raise RuntimeError(
            f"User proxy is only valid for {valid_hours:.2f} hours. "
            f"Minimum required is {min_valid_hours} hours."
        )
