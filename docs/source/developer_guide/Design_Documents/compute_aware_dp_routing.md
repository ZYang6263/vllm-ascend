# Prefill-Aware DP Request Routing

Status: Experimental

Target baseline: vLLM `v0.27.1`

## Summary

This feature improves DP entry routing for DP+EP models in mixed
Prefill/Decode serving. Request count alone cannot distinguish a DP engine
holding several long prompts from one holding the same number of short
requests. Since EP collectives synchronize the DP engines at each model step,
that imbalance can make lighter engines wait for a Prefill-heavy engine.

The initial implementation deliberately models only Prompt tokens. The API
server records a request's full Prompt when routing it, and the selected Engine
reports the request's remaining Prefill work after every completed chunk. The
Coordinator and its protocol remain unchanged. The feature changes where a new
request enters the DP group; it does not migrate requests or alter continuous
batching inside an engine.

The feature is disabled by default and supports a shadow mode that records the
new decision while preserving vLLM's original destination.

## Scope

The initial implementation supports:

- online vLLM V1 serving with `data_parallel_size > 1`;
- internal or hybrid DP load balancing;
- PD-mixed text generation;
- DP+EP MoE deployments on Ascend.

External DP load balancing, application-level offline DP, P/D-disaggregated
producer or consumer nodes, pooling, and multimodal requests do not use the
new policy.

## Data path

The Coordinator and its three-field snapshots remain unchanged. Prefill
progress returns on the existing EngineCore output socket:

```text
DP Engines -------------------------------+
   | [waiting, running, kv_usage]          |
   v                                      | per-request absolute
upstream DP Coordinator                   | remaining Prefill tokens
   | unchanged snapshot                   |
   v                                      v
API Server / DPLBAsyncMPClient
   | snapshot + locally maintained Prefill debt
   v
selected DP Engine
```

For each supported request, the API server obtains the Prompt length before
routing. After choosing an engine it immediately records:

```text
local_prefill_debt[engine] += prompt_tokens
```

After a scheduled chunk finishes, the Engine reports:

```text
confirmed_tokens = num_computed_tokens - num_in_flight_tokens
remaining_prefill = prompt_tokens - min(prompt_tokens, confirmed_tokens)
```

Subtracting in-flight tokens is important for asynchronous scheduling: work
that has only been submitted must not be treated as completed. Reports are
absolute rather than deltas, so duplicate reports are idempotent. A report may
increase the debt after preemption when a request has to recompute Prompt
tokens. Debt is removed on completion or abort and rolled back if sending the
request fails.

vLLM's `EngineCoreOutputs` currently has no plugin extension field. This
prototype encodes progress as a reserved `EngineCoreOutput` whose payload uses
only existing serializable fields. `DPLBAsyncMPClient` consumes and removes
that control output before normal request-output processing. No message is
sent through the Coordinator.

## Routing score

The policy preserves vLLM's request-count floor and KV-pressure safeguard:

```text
base_score_i = max(
    client_count * local_inflight_request_count_i,
    waiting_i + running_i,
)

base_score_i +=
    waiting_i * 6 * max(0, kv_cache_usage_i - 0.5)
```

It adds only the locally observed Prefill debt:

```text
prefill_score_i =
    client_count
    * local_prefill_debt_i
    / max_num_batched_tokens

score_i = base_score_i + prefill_score_i
```

Dividing by `max_num_batched_tokens` converts Prompt debt into approximate
full Prefill scheduler steps without introducing a configurable weight. The
`client_count` multiplier follows vLLM's existing approximation for local
in-flight request counts when several API servers share the engines.

The incoming request's Prompt length is identical for every candidate and
does not change that request's ordering. Recording it immediately against the
selected engine changes the next routing decision.

Explicit DP-rank and late-interaction routing retain upstream precedence.
Malformed snapshots and unsupported requests use the upstream router.

## Configuration

No environment variable is added. The only options are:

```json
{
  "scheduler_config": {
    "compute_aware_routing_config": {
      "enabled": true,
      "shadow_mode": true
    }
  }
}
```

- `enabled`: enable API-local Prefill accounting and scoring.
- `shadow_mode`: calculate the Prefill-aware choice but route with the
  upstream policy. It defaults to `true` for safe evaluation.

## Interaction with scheduling controls

`prefill_schedule_interval` is not changed automatically. Routing is a spatial
control that distributes incoming Prefill work. The interval is a temporal
control that restricts which synchronized steps may admit Prefill work and can
reduce aggregate throughput when capacity is left unused.

Balance scheduling may remain enabled as a capacity safety net. It prevents
all DP ranks from admitting additional waiting requests when one rank reaches
its running capacity; Prefill-aware routing aims to avoid creating that
imbalance in the first place.

## Risks and limitations

### Multiple API servers do not share Prefill debt

Each API server knows only the requests it routed. Multiplying local debt by
`client_count` assumes traffic is distributed similarly across frontends. If
several API servers simultaneously see the same DP as least loaded, they can
all route long requests to it. This is the main accuracy limitation and is why
active mode should initially target a single API-server deployment.

### Progress is delayed until a chunk completes

The API initially charges the full Prompt and cannot account for prefix-cache
hits until the first Engine report. A running chunk and asynchronously queued
chunks remain charged until their outputs are processed. This is deliberately
conservative, but very large chunks can still make the load view lag.

### Prefix-cache hits are unknown before routing

The Prompt is charged at full length even if the selected engine can reuse
most of it from its local prefix cache. With asymmetric per-DP caches, the
policy can choose a lower-debt engine with a worse cache hit and increase
actual Prefill work. Prefix affinity is outside the initial design.

### Token count is only a proxy for compute time

Prefill cost depends on sequence-length distribution, attention complexity,
batch shape, model configuration, expert imbalance, and NPU utilization. Two
engines with the same total Prompt tokens need not take the same time. The
score also combines request-count units with normalized Prefill-step units
without a calibrated conversion factor.

### Engine and API must run matching plugin code

The prototype reuses the existing `EngineCoreOutput` schema but assigns a
reserved request ID and payload key. An API process without the consumer patch
would treat this as a normal output for an unknown request. Rolling upgrades
must therefore not mix Engine and API processes with and without this feature.
A typed upstream extension field is the preferred long-term protocol.

### Per-chunk reporting has overhead

Every change in remaining Prefill work adds a small control output. It shares
the normal output socket and does not enter user output processing, but workloads
with many small Prefill chunks increase Engine-to-API serialization and IPC
traffic.

### Local state depends on output lifecycle events

Missing progress, finish, abort, or send-failure cleanup can leave stale debt
and bias later routing. The first generated token is retained as a compatibility
fallback that sets remaining Prefill debt to zero. Elastic EP resize removes
state for engines that disappear, but active requests on a failed engine still
rely on normal vLLM failure handling.

## Compatibility and upstream plan

When disabled, the patch calls the original `DPLBAsyncMPClient` methods and
does not install the Scheduler reporting hook. When enabled, the Coordinator
payload remains `[waiting, running, kv_usage]`; however, Engine and API plugin
versions must match because of the reserved direct progress output described
above.

The routing lifecycle is hardware-independent and should ultimately use an
upstream `DPRoutingPolicy` interface. If accurate multi-API-server state is
required, a later upstream design can add typed global Prefill statistics or
Coordinator-side admission reservations. The initial vLLM-Ascend patch avoids
copying the Coordinator event loop while that interface is unavailable.

## Test plan

Unit coverage includes:

- equal request counts with unequal local Prompt debt;
- `client_count` scaling;
- preservation of request-count and KV safeguards;
- burst spreading before a Coordinator refresh;
- partial Chunked-Prefill progress and completion cleanup;
- asynchronous in-flight token exclusion and preemption debt restoration;
- internal progress-output removal before normal API output processing;
- shadow disagreement accounting;
- malformed snapshot and disabled-path fallback.

NPU integration and performance validation should cover DP=2/4, prefix
caching, chunked Prefill, async scheduling, elastic EP, balance scheduling
on/off, and `prefill_schedule_interval` values 1, 2, and 4. The primary
measurements are DP step-time skew, EP wait, TTFT, TPOT, throughput, and API
Server routing overhead.
