import os
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
    valid_hours = int(shell.get_outerr()) / 60 / 60
    if valid_hours < min_valid_hours:
        raise RuntimeError(
            f"User proxy is only valid for {valid_hours} hours. "
            f"Minimum required is {min_valid_hours} hours."
        )
