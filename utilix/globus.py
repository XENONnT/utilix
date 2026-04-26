import os
import json

import globus_sdk
from globus_sdk.scopes import TransferScopes

from utilix import uconfig

TOKEN_PATH = os.path.join(os.environ["HOME"], ".globus_tokens.json")


class GlobusTransfer:
    def __init__(self):
        self.client_id = uconfig.get("straxen", "globus_client_id")
        self.source = uconfig.get("straxen", "globus_source")
        self.destination = uconfig.get("straxen", "globus_destination")
        self.source_path = uconfig.get("straxen", "globus_source_path")

        self._data_access_scope = (
            f"{TransferScopes.all}"
            f"[*https://auth.globus.org/scopes/{self.source}/data_access"
            f" *https://auth.globus.org/scopes/{self.destination}/data_access]"
        )
        self._auth_client = globus_sdk.NativeAppAuthClient(self.client_id)
        self.transfer_client = self._get_transfer_client()

    def _save_tokens(self, tokens):
        with open(TOKEN_PATH, "w") as f:
            json.dump(tokens, f)

    def _make_transfer_client(self, t, on_refresh):
        return globus_sdk.TransferClient(
            authorizer=globus_sdk.RefreshTokenAuthorizer(
                t["refresh_token"],
                self._auth_client,
                access_token=t["access_token"],
                expires_at=t["expires_at_seconds"],
                on_refresh=on_refresh,
            )
        )

    def _get_transfer_client(self):
        if os.path.exists(TOKEN_PATH):
            with open(TOKEN_PATH) as f:
                saved = json.load(f)
            t = saved.get("transfer.api.globus.org", {})
            if t.get("refresh_token"):
                return self._make_transfer_client(
                    t, on_refresh=lambda tr: self._save_tokens({**saved, **tr.by_resource_server})
                )

        self._auth_client.oauth2_start_flow(
            requested_scopes=self._data_access_scope, refresh_tokens=True
        )
        print(
            "First-time login required. "
            "Open the URL below in a browser, "
            "log in with your Globus account, "
            "and paste the authorization code back here:\n"
        )
        print(self._auth_client.oauth2_get_authorize_url())
        code = input("\nAuthorization code: ").strip()
        tokens = self._auth_client.oauth2_exchange_code_for_tokens(code)
        self._save_tokens(tokens.by_resource_server)
        t = tokens.by_resource_server["transfer.api.globus.org"]
        return self._make_transfer_client(
            t, on_refresh=lambda tr: self._save_tokens(tr.by_resource_server)
        )

    def build_transfer_data(self, src_path, dst_path):
        transfer_data = globus_sdk.TransferData(
            source_endpoint=self.source,
            destination_endpoint=self.destination,
        )
        transfer_data.add_item(src_path, dst_path)
        return transfer_data
