from utilix import uconfig
from utilix.globus import GlobusTransfer


def main():
    globus_transfer = GlobusTransfer()
    globus_transfer.transfer_client.operation_stat(
        uconfig.get("straxen", "globus_source"), "/shared"
    )


if __name__ == "__main__":
    main()
