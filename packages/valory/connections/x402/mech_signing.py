# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""Hashing primitives shared with the mech facilitator and the marketplace.

Three byte-exact mirrors live here:

* ``canonical_request_data``: the bytes the facilitator commits into the
  marketplace ``requestData`` for an HTTP call. Same key order, separators
  and escaping as ``mech_facilitator/canonical.py`` in x402-poc.
* ``derive_request_id``: ``MechMarketplace.getRequestId``. A copy of the
  mech_interact skill's implementation (a connection cannot import a
  skill), pinned by the same golden vector in the tests.
* ``compute_safe_message_hash``: the Safe ``CompatibilityFallbackHandler``
  wrap the marketplace applies before ``isValidSignature`` on a Safe
  requester (Safe >= 1.3.0).

If any of the three drifts, the facilitator rejects the request_id (400)
or the marketplace recovers the wrong signer at settlement, so change
them only together with their counterparts.
"""

import json
from typing import Dict

from eth_abi import encode as abi_encode
from eth_utils import keccak

_MARKETPLACE_NAME = "MechMarketplace"
_MARKETPLACE_VERSION = "1.1.0"
_DOMAIN_TYPEHASH = keccak(
    text=(
        "EIP712Domain(string name,string version,uint256 chainId,"
        "address verifyingContract)"
    )
)
_SAFE_DOMAIN_TYPEHASH = keccak(
    text="EIP712Domain(uint256 chainId,address verifyingContract)"
)
_SAFE_MESSAGE_TYPEHASH = keccak(text="SafeMessage(bytes message)")


def canonical_request_data(
    *,
    method: str,
    path: str,
    query: Dict[str, str],
    body_b64: str,
    expires_at: int,
) -> bytes:
    """Return the deterministic UTF-8 encoding the facilitator hashes.

    :param method: HTTP method; upper-cased before encoding.
    :param path: upstream path, e.g. ``/v1beta/models/x:generateContent``.
    :param query: query parameters; encoded with sorted keys.
    :param body_b64: base64 of the raw request body ("" when empty).
    :param expires_at: unix seconds (UTC) after which the request is refused.
    :return: the canonical bytes.
    """
    payload = {
        "method": method.upper(),
        "path": path,
        "query": dict(sorted(query.items())),
        "body_b64": body_b64,
        "expires_at": int(expires_at),
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def compute_domain_separator(chain_id: int, marketplace_address: str) -> bytes:
    """Reproduce ``MechMarketplace._computeDomainSeparator``.

    The contract hashes the version through ``abi.encode`` rather than
    as raw bytes (``MechMarketplace.sol:160``), so the Python side
    encodes the version string the same way.

    :param chain_id: the settlement chain id.
    :param marketplace_address: the marketplace contract address.
    :return: the 32-byte domain separator.
    """
    name_hash = keccak(text=_MARKETPLACE_NAME)
    version_hash = keccak(abi_encode(["string"], [_MARKETPLACE_VERSION]))
    return keccak(
        abi_encode(
            ["bytes32", "bytes32", "bytes32", "uint256", "address"],
            [_DOMAIN_TYPEHASH, name_hash, version_hash, chain_id, marketplace_address],
        )
    )


def derive_request_id(  # pylint: disable=too-many-arguments
    *,
    marketplace_address: str,
    mech_address: str,
    requester: str,
    data: bytes,
    delivery_rate: int,
    payment_type: bytes,
    nonce: int,
    chain_id: int,
) -> bytes:
    """Local mirror of ``MechMarketplace.getRequestId``.

    :param marketplace_address: ``address(this)`` on the settlement chain.
    :param mech_address: the mech contract the request targets.
    :param requester: the Safe that pays the delivery rate.
    :param data: the bytes committed on-chain as ``requestData``.
    :param delivery_rate: per-request charge in token base units.
    :param payment_type: 32-byte ``paymentType`` constant of the mech.
    :param nonce: the requester's marketplace nonce for this request.
    :param chain_id: the settlement chain id.
    :return: the 32-byte request_id the contract computes at settlement.
    """
    if len(payment_type) != 32:
        raise ValueError("payment_type must be 32 bytes")
    domain_separator = compute_domain_separator(chain_id, marketplace_address)
    inner_hash = keccak(
        abi_encode(
            [
                "address",
                "address",
                "address",
                "bytes32",
                "uint256",
                "bytes32",
                "uint256",
            ],
            [
                marketplace_address,
                mech_address,
                requester,
                keccak(data),
                delivery_rate,
                payment_type,
                nonce,
            ],
        )
    )
    return keccak(b"\x19\x01" + domain_separator + inner_hash)


def compute_safe_message_hash(
    request_id_bytes: bytes, safe_address: str, chain_id: int
) -> bytes:
    """Wrap ``request_id`` in the Safe ``SafeMessage`` EIP-712 digest.

    This is what ``Safe.isValidSignature(request_id, sig)`` validates the
    owner signature against, so the owner must sign this digest, not the
    raw request_id.

    :param request_id_bytes: the 32-byte request_id.
    :param safe_address: the requester Safe.
    :param chain_id: the settlement chain id.
    :return: the 32-byte digest to sign.
    """
    if len(request_id_bytes) != 32:
        raise ValueError("request_id_bytes must be 32 bytes")
    domain_separator = keccak(
        abi_encode(
            ["bytes32", "uint256", "address"],
            [_SAFE_DOMAIN_TYPEHASH, chain_id, safe_address],
        )
    )
    message = abi_encode(["bytes32"], [request_id_bytes])
    struct_hash = keccak(
        abi_encode(["bytes32", "bytes32"], [_SAFE_MESSAGE_TYPEHASH, keccak(message)])
    )
    return keccak(b"\x19\x01" + domain_separator + struct_hash)
