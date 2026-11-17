# Helix REST API Reference (v2)

## Authentication
All requests to the Helix API (`https://api.helix.dev/v2`) must include an `Authorization: Bearer <token>` header and an `X-Helix-Tenant` header specifying the workspace slug. API tokens expire after 90 days unless rotated via `POST /v2/auth/rotate`.

## Rate Limits
- Standard tier: 120 requests per minute per tenant.
- Enterprise tier: 1,200 requests per minute per tenant.
When rate-limited, the API returns HTTP `429 Too Many Requests` with a `Retry-After` header indicating seconds to wait.

## Pagination
List endpoints (such as `GET /v2/datasets` and `GET /v2/jobs`) use cursor-based pagination. Pass `?limit=50` (default 25, maximum 100) and `?cursor=<opaque_token>`. The response includes `next_cursor` which is `null` on the final page.

## Webhooks and Retries
Helix signs all webhook payloads with HMAC-SHA256 in the `X-Helix-Signature` header. If your endpoint fails to return a `2xx` status within 5 seconds, Helix retries delivery up to 5 times using exponential backoff: 10s, 30s, 2m, 10m, and 30m. After 5 failed attempts, the webhook is marked `suspended`.
