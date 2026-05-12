# Adapted from https://github.com/coinbase/x402/tree/main/python/x402/src/x402

import concurrent.futures
import logging
from typing import Dict, List, Optional, Tuple, Union

from eth_account import Account
from httpx import AsyncClient, Request, Response, Timeout

from packages.valory.connections.x402.clients.base import (
    MissingRequestConfigError,
    PaymentError,
    PaymentRejectedAfterRetryError,
    PaymentSelectorCallable,
    x402Client,
)
from packages.valory.connections.x402.types import x402PaymentRequiredResponse


_logger = logging.getLogger(__name__)


DEFAULT_X402_HTTPX_TIMEOUT: Tuple[float, float] = (10.0, 60.0)

HttpxTimeout = Union[float, Tuple[float, float], Timeout]


def _coerce_httpx_timeout(value: HttpxTimeout) -> Timeout:
    """Convert (connect, read) tuples to an httpx Timeout instance."""
    if isinstance(value, Timeout):
        return value
    if isinstance(value, tuple):
        connect, read = value
        return Timeout(connect=connect, read=read, write=read, pool=connect)
    return Timeout(value)


class HttpxHooks:
    def __init__(
        self,
        client: x402Client,
        retry_timeout: HttpxTimeout = DEFAULT_X402_HTTPX_TIMEOUT,
    ):
        self.client = client
        self._is_retry = False
        self._retry_timeout = _coerce_httpx_timeout(retry_timeout)

    async def on_request(self, request: Request):
        """Handle request before it is sent."""
        pass

    async def on_response(self, response: Response) -> Response:
        """Handle response after it is received."""

        # If this is not a 402, just return the response
        if response.status_code != 402:
            return response

        # If this is a retry response, just return it
        if self._is_retry:
            return response

        self._is_retry = True
        try:
            if not response.request:
                raise MissingRequestConfigError("Missing request configuration")

            # Read the response content before parsing
            await response.aread()

            data = response.json()

            payment_response = x402PaymentRequiredResponse(**data)

            selected_requirements = self.client.select_payment_requirements(
                payment_response.accepts
            )

            payment_header = self.client.create_payment_header(
                selected_requirements, payment_response.x402_version
            )

            request = response.request
            request.headers["X-Payment"] = payment_header
            request.headers["Access-Control-Expose-Headers"] = "X-Payment-Response"

            async with AsyncClient(timeout=self._retry_timeout) as client:
                retry_response = await client.send(request)

            if retry_response.status_code == 402:
                _logger.warning(
                    "x402 retry returned 402 after payment header was attached; "
                    "upstream still rejects the request."
                )
                _logger.debug(
                    "x402 retry body (truncated): %r", retry_response.content[:500]
                )
                raise PaymentRejectedAfterRetryError(
                    status_code=retry_response.status_code,
                    body=retry_response.content,
                )

            response.status_code = retry_response.status_code
            response.headers = retry_response.headers
            response._content = retry_response._content
            return response

        except (PaymentError, concurrent.futures.CancelledError):
            raise
        except Exception as e:
            raise PaymentError(f"Failed to handle payment: {str(e)}") from e
        finally:
            self._is_retry = False


def x402_payment_hooks(
    account: Account,
    max_value: Optional[int] = None,
    payment_requirements_selector: Optional[PaymentSelectorCallable] = None,
    retry_timeout: HttpxTimeout = DEFAULT_X402_HTTPX_TIMEOUT,
) -> Dict[str, List]:
    """Create httpx event hooks dictionary for handling 402 Payment Required responses.

    Args:
        account: eth_account.Account instance for signing payments
        max_value: Optional maximum allowed payment amount in base units
        payment_requirements_selector: Optional custom selector for payment requirements.
            Should be a callable that takes (accepts, network_filter, scheme_filter, max_value)
            and returns a PaymentRequirements object.
        retry_timeout: Timeout applied to the fresh AsyncClient used for the
            post-payment retry. Accepts a float, a (connect, read) tuple, or
            an httpx.Timeout instance.

    Returns:
        Dictionary of event hooks that can be directly assigned to client.event_hooks
    """
    # Create x402Client
    client = x402Client(
        account,
        max_value=max_value,
        payment_requirements_selector=payment_requirements_selector,
    )

    # Create hooks
    hooks = HttpxHooks(client, retry_timeout=retry_timeout)

    # Return event hooks dictionary
    return {
        "request": [hooks.on_request],
        "response": [hooks.on_response],
    }


class x402HttpxClient(AsyncClient):
    """AsyncClient with built-in x402 payment handling."""

    def __init__(
        self,
        account: Account,
        max_value: Optional[int] = None,
        payment_requirements_selector: Optional[PaymentSelectorCallable] = None,
        retry_timeout: HttpxTimeout = DEFAULT_X402_HTTPX_TIMEOUT,
        **kwargs,
    ):
        """Initialize an AsyncClient with x402 payment handling.

        Args:
            account: eth_account.Account instance for signing payments
            max_value: Optional maximum allowed payment amount in base units
            payment_requirements_selector: Optional custom selector for payment requirements.
                Should be a callable that takes (accepts, network_filter, scheme_filter, max_value)
                and returns a PaymentRequirements object.
            retry_timeout: Timeout applied to the post-payment retry's
                AsyncClient. Defaults to ``DEFAULT_X402_HTTPX_TIMEOUT``.
            **kwargs: Additional arguments to pass to AsyncClient
        """
        super().__init__(**kwargs)
        self.event_hooks = x402_payment_hooks(
            account,
            max_value,
            payment_requirements_selector,
            retry_timeout=retry_timeout,
        )
