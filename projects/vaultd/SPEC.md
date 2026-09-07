---
title: vaultd — Hosted Second Brain Platform
tags: [rust, project, spec, llm, obsidian, telegram, postgres]
---

# vaultd — Full Specification

A hosted platform where each vault is a GitHub repository. Captures arrive from a phone,
run through configurable LLM stages, and land in a review queue. You promote what is worth
keeping; promoted notes are committed to the repository and enter the graph.

Version: 0.3 draft · Rust 2024 · supersedes the v0.1 single-user daemon spec

---

## 1. Purpose and scope

### Problem

Two problems, and the second is the one most tools miss.

**Capture friction.** On a phone, opening Obsidian, picking a folder, writing frontmatter
is enough friction that thoughts do not get recorded. Voice is the lowest-friction input
available and the one Obsidian handles worst.

**Review friction.** Anything that writes to a knowledge base without review turns it into
a landfill. LLM output is fast enough to bury you. The review step is not overhead — it is
the product.

### Goal

Open the app, hold the button, say the thing. It is transcribed, structured, and comes
straight back in the conversation as a note card: title, tags, where it would attach in your
graph, and promote / edit / discard on the card itself. Handle it now, or leave it for the
review queue. What you promote is committed to your repository and becomes part of a graph
you can watch grow.

Then flip the same box to **Ask** and it answers from what you already wrote — grounded in
your notes, citing them by path, and saying "nothing in your vault covers that" rather than
inventing. Capture writes. Ask only reads.

Capture and review are the same surface. The conversation is the fast path; the review queue
is the backlog of what you deferred; Ask is what makes the accumulated pile worth having.

### Shape

- **Storage is the user's.** One vault = one GitHub repository. Plain Markdown, Obsidian-
  compatible, readable without us, with a static export committed alongside.
- **The brain is ours.** A hosted multi-tenant app provides review, graph, graph diff,
  model configuration, and capture channels — none of which a static site can do.
- **Leaving costs nothing.** Disconnect and the repository, every note, and the export stay
  exactly as they are.

### In scope

- GitHub App auth; multi-vault per account; vault = repository
- **Built-in capture**: an in-app conversation taking text, voice and images, replying with the
  staged note itself — no integration required, works the moment a vault exists
- **Ask**: grounded read-only question answering over the vault, with mandatory citations,
  retrieval performed by us rather than by the model
- Optional capture integrations that feed the same queue: Telegram, email-in, API token
- Per-vault model configuration across three stages, cloud or self-hosted
- Per-vault "house style" appended to the structuring prompt
- Review queue with diff, graph impact, edit, promote, discard
- Graph index with time-windowed diff, orphan and wanted-note detection
- Static export committed on promote
- Bidirectional awareness: notes edited in Obsidian and pushed are reindexed

### Non-goals

Each removed item is a class of bug and a class of attack.

- **No agent loop.** The model never calls tools and never chooses actions. In Capture it
  transforms text and returns one document matching one schema; in Ask it receives notes we
  already selected and returns prose with citations. Retrieval is a database query we run,
  not a tool the model invokes. Everything stays one call, one turn, no loop.
- **No writes outside the inbox path on capture.** Promotion is the only thing that places
  a file in a curated folder, and a human triggers it.
- **No collaborative editing.** One owner per vault in v1. Read-only sharing later.
- **No vector search until it is earned.** Ask retrieves with Postgres full-text search plus
  one hop of graph expansion (§15). A vault is the user's own vocabulary, which is exactly
  the case where lexical search is strong. `pgvector` is one extension away when measured
  recall says so — see §15.7 for the threshold. Shipping embeddings first would be paying
  for infrastructure before knowing whether it helps.
- **Ask never writes.** No "update that note for me", no agentic edits. The only path into
  the repository is promotion of a staged note, triggered by a human. An answer can be saved
  as a note, and that runs through the identical staging and review path.
- **We are not an Obsidian replacement.** Editing long-form prose happens in Obsidian. We
  own capture, review, the graph, and now recall.

---

## 2. What changed from v0.1

v0.1 specified a single-user daemon on the user's own machine writing straight to a repo.
The product is now a hosted platform. The changes are not cosmetic.

| Area | v0.1 | v0.3 |
|---|---|---|
| Deployment | One binary, one user, their machine | Multi-tenant service, API + workers |
| Auth | Fine-grained PAT in the OS keychain | GitHub App, installation tokens, browser session |
| Store | sqlite, one file | Postgres with row-level security, Redis for queue and cache |
| Vaults | Exactly one | Many per account; vault = repository |
| Secrets | OS keychain | Envelope encryption, per-vault data keys |
| Write model | Capture commits straight to inbox | Capture **stages**; promote commits |
| Graph | Not modeled | First-class index, diff, orphans, wanted notes |
| Publishing | Quartz on the user's Pages | Hosted app + committed static export |
| Local models | Trivial — same machine | **Requires a connector agent.** See §14.3 |
| Capture surface | Telegram, required | **Built-in in-app capture, default.** Telegram optional |
| Recall | None — the vault was write-only | **Ask**: grounded Q&A over the vault, citations required |
| Frontend | None | TypeScript SPA against a REST + SSE API |

What survives unchanged: the channel adapter design, the pipeline stages, the schema-locked
structuring call, path sanitization, and the `draft` interlock. Those were right.

---

## 3. What we take from OpenClaw

[OpenClaw](https://github.com/openclaw/openclaw) is a Node/TypeScript personal assistant
gateway with adapters for WhatsApp, Telegram, Slack, Discord, Signal and iMessage. It solved
the messaging-adapter problem in public and its Telegram channel has absorbed years of Bot
API pain. Telegram is one optional channel for us rather than the product, but its adapter is still
the one with the most edge cases, so the shape we borrow is worth borrowing. We build a
capture pipe, not an assistant, so we take the adapter design and leave the agent runtime.

| OpenClaw concept | What we take | What we drop |
|---|---|---|
| Channel adapter with four duties: auth, inbound parsing, access control, outbound formatting | Exactly this, as a Rust trait | Runtime plugin loading |
| Session keys encoding trust boundary | Same scheme, scoped per vault: `vault:<uuid>:telegram:dm:<user>` | Per-session sandbox containers |
| `dmPolicy`: `pairing` / `allowlist` / `open` / `disabled`, codes expiring after 1 hour | All four, same default (`pairing`), same TTL | — |
| Polling default, webhook with a secret for production | Webhook only — a hosted service has a public URL and polling per tenant does not scale | Long polling |
| Editable status drafts instead of streaming answer text | Same. One message, edited through the stages | Token streaming |
| 4000-character outbound chunk limit | Same, with a UTF-16-correct chunker | — |
| 100 MB media cap, group `requireMention`, forum topics as `:topic:<id>` | All of it | Group history windows |
| "Trusted gateway, untrusted execution, deterministic policy" | The posture. We have no tool execution at all, so the hard half is free | Docker sandboxing |

The borrowed idea that matters most is the dullest: **the adapter normalizes, and nothing
downstream knows what Telegram is.** Every platform quirk lives in one crate.

---

## 4. System overview

```mermaid
flowchart TD
    WEB[TypeScript SPA - built-in capture] -->|REST + SSE| API
    OBJ[(object store)] <-.presigned PUT.- WEB
    TG[Telegram - optional] -->|webhook| API
    MAIL[email-in - optional] --> API
    GH[GitHub webhooks] -->|push events| API

    API[vaultd-api  axum] --> PG[(Postgres  RLS)]
    API --> RD[(Redis  streams, cache, rate limit)]
    RD --> W[vaultd-worker]

    W --> PIPE[capture pipeline]
    PIPE --> S1[materialize: fetch + hash to object store]
    S1 --> S2[extract: transcribe / vision]
    S2 --> S3[structure: schema-locked LLM call]
    S3 --> STAGE[(staged note in Postgres)]

    STAGE -.review.-> WEB
    WEB -->|promote| API
    API --> COMMIT[vaultd-github: git data API]
    COMMIT --> REPO[(user repository)]
    COMMIT --> GRAPH[vaultd-graph: index + edges]
    COMMIT --> EXPORT[vaultd-export: static site]
    EXPORT --> REPO

    GH --> REINDEX[reindex changed paths] --> GRAPH
    CONN[connector agent on user hardware] -.reverse tunnel.-> W
```

Three processes: `vaultd-api` (HTTP, sessions, webhooks, SSE), `vaultd-worker` (queue
consumers), and the TypeScript SPA. Postgres is the source of truth for platform state;
the user's repository is the source of truth for notes.

The SPA is both the capture surface and the review surface. Integrations are additional
inlets to the same queue — remove every one of them and the product still works.

---

## 5. Tenancy and data model

### 5.1 Isolation

Every tenant-owned row carries `account_id`, and Postgres row-level security enforces it.
Application code cannot forget a `WHERE` clause, because the database will not return the
rows.

```sql
ALTER TABLE vaults ENABLE ROW LEVEL SECURITY;
CREATE POLICY vault_tenant ON vaults
  USING (account_id = current_setting('app.account_id')::uuid);
```

Every request opens a transaction and sets the GUC before any query:

```rust
sqlx::query!("SELECT set_config('app.account_id', $1, true)", account_id.to_string())
    .execute(&mut *tx).await?;
```

`set_config(..., true)` is transaction-local, so a pooled connection cannot leak the
setting into the next request. The worker sets it per job from the job's vault.

The API's database role must **not** be `BYPASSRLS` and must not own the tables. Migrations
run as a separate owner role. This is the difference between RLS being a real boundary and
RLS being decoration.

### 5.2 Schema

```sql
-- identity ------------------------------------------------------------
CREATE TABLE accounts (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  github_user_id bigint UNIQUE NOT NULL,
  login          text NOT NULL,
  created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE installations (
  installation_id bigint PRIMARY KEY,          -- from GitHub
  account_id      uuid NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
  suspended       boolean NOT NULL DEFAULT false,
  updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE sessions (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  account_id uuid NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
  expires_at timestamptz NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- vaults --------------------------------------------------------------
CREATE TABLE vaults (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  account_id      uuid NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
  installation_id bigint NOT NULL REFERENCES installations(installation_id),
  repo_owner      text NOT NULL,
  repo_name       text NOT NULL,
  branch          text NOT NULL DEFAULT 'main',
  slug            text NOT NULL,               -- hosting path segment
  inbox_dir       text NOT NULL DEFAULT '05 - Inbox',
  attachments_dir text NOT NULL DEFAULT '05 - Inbox/attachments',
  timezone        text NOT NULL DEFAULT 'UTC',
  house_style     text,                        -- appended to structuring prompt
  auto_commit_inbox boolean NOT NULL DEFAULT false,
  export_enabled  boolean NOT NULL DEFAULT true,
  head_sha        text,                        -- last indexed commit
  created_at      timestamptz NOT NULL DEFAULT now(),
  UNIQUE (repo_owner, repo_name),
  UNIQUE (account_id, slug)
);

-- secrets -------------------------------------------------------------
CREATE TABLE vault_keys (                      -- one wrapped DEK per vault
  vault_id     uuid PRIMARY KEY REFERENCES vaults(id) ON DELETE CASCADE,
  wrapped_dek  bytea NOT NULL,
  key_version  int  NOT NULL,
  rotated_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE secrets (
  vault_id   uuid NOT NULL REFERENCES vaults(id) ON DELETE CASCADE,
  purpose    text NOT NULL,                    -- anthropic_api_key, telegram_bot_token, …
  nonce      bytea NOT NULL,
  ciphertext bytea NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (vault_id, purpose)
);

-- model configuration -------------------------------------------------
CREATE TABLE model_slots (
  vault_id  uuid NOT NULL REFERENCES vaults(id) ON DELETE CASCADE,
  stage     text NOT NULL,                     -- transcribe | vision | structure | answer
  provider  text NOT NULL,                     -- anthropic | openai | deepgram | connector
  model     text NOT NULL,
  base_url  text,                              -- connector / self-hosted endpoints only
  PRIMARY KEY (vault_id, stage)
);

-- channels ------------------------------------------------------------
CREATE TABLE channels (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  vault_id   uuid NOT NULL REFERENCES vaults(id) ON DELETE CASCADE,
  kind       text NOT NULL,                    -- web | telegram | email | api
  dm_policy  text NOT NULL DEFAULT 'pairing',  -- ignored for kind='web'
  webhook_path text UNIQUE,                    -- random, secret; null for web
  enabled    boolean NOT NULL DEFAULT true,
  UNIQUE (vault_id, kind) DEFERRABLE           -- one web channel per vault
);
-- Every vault gets kind='web' at creation. It has no credentials, no webhook and
-- no policy: its sender is the authenticated session owner.

CREATE TABLE chat_messages (                   -- the built-in capture conversation
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  vault_id   uuid NOT NULL REFERENCES vaults(id) ON DELETE CASCADE,
  role       text NOT NULL,                    -- user | note | answer
  mode       text NOT NULL DEFAULT 'capture',  -- capture | ask
  capture_id uuid REFERENCES captures(id) ON DELETE SET NULL,
  note_id    uuid REFERENCES notes(id) ON DELETE SET NULL,
  body       text,
  answer     jsonb,                            -- { answer, citations, sources[] }
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX chat_recent ON chat_messages(vault_id, created_at DESC);

CREATE TABLE channel_senders (                 -- allowlist
  channel_id uuid NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
  sender_id  text NOT NULL,
  added_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (channel_id, sender_id)
);

CREATE TABLE pairings (
  code       text PRIMARY KEY,
  channel_id uuid NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
  sender_id  text NOT NULL,
  expires_at timestamptz NOT NULL,
  used_at    timestamptz
);

-- capture -------------------------------------------------------------
CREATE TABLE captures (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  vault_id       uuid NOT NULL REFERENCES vaults(id) ON DELETE CASCADE,
  channel_id     uuid REFERENCES channels(id) ON DELETE SET NULL,
  channel_msg_id text NOT NULL,
  session_key    text NOT NULL,
  envelope       jsonb NOT NULL,
  stage          text NOT NULL DEFAULT 'queued',
  state          text NOT NULL DEFAULT 'pending',   -- pending|running|staged|dead
  attempts       int  NOT NULL DEFAULT 0,
  next_attempt_at timestamptz NOT NULL DEFAULT now(),
  status_msg_id  text,
  last_error     text,
  created_at     timestamptz NOT NULL DEFAULT now(),
  UNIQUE (channel_id, channel_msg_id)
);

CREATE TABLE capture_artifacts (
  capture_id uuid NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
  stage      text NOT NULL,
  payload    jsonb NOT NULL,
  PRIMARY KEY (capture_id, stage)
);

CREATE TABLE blobs (
  hash       text PRIMARY KEY,                 -- blake3
  vault_id   uuid NOT NULL REFERENCES vaults(id) ON DELETE CASCADE,
  mime       text NOT NULL,
  size_bytes bigint NOT NULL,
  object_key text NOT NULL,                    -- S3/R2 key
  created_at timestamptz NOT NULL DEFAULT now()
);

-- notes ---------------------------------------------------------------
CREATE TABLE notes (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  vault_id    uuid NOT NULL REFERENCES vaults(id) ON DELETE CASCADE,
  capture_id  uuid REFERENCES captures(id) ON DELETE SET NULL,
  status      text NOT NULL,                   -- staged | promoted | discarded
  path        text,                            -- null until promoted
  title       text NOT NULL,
  slug        text NOT NULL,
  tags        text[] NOT NULL DEFAULT '{}',
  body_md     text NOT NULL,
  frontmatter jsonb NOT NULL,
  commit_sha  text,
  created_at  timestamptz NOT NULL DEFAULT now(),
  promoted_at timestamptz
);
CREATE INDEX notes_review ON notes(vault_id, status, created_at DESC);

-- graph ---------------------------------------------------------------
CREATE TABLE graph_nodes (
  vault_id      uuid NOT NULL REFERENCES vaults(id) ON DELETE CASCADE,
  path          text NOT NULL,
  title         text NOT NULL,
  aliases       text[] NOT NULL DEFAULT '{}',
  tags          text[] NOT NULL DEFAULT '{}',
  first_seen_at timestamptz NOT NULL DEFAULT now(),
  last_seen_at  timestamptz NOT NULL DEFAULT now(),
  x             real,                          -- persisted layout, see 13.4
  y             real,
  PRIMARY KEY (vault_id, path)
);

CREATE TABLE graph_edges (
  vault_id      uuid NOT NULL REFERENCES vaults(id) ON DELETE CASCADE,
  src_path      text NOT NULL,
  dst_title     text NOT NULL,                 -- as written in the [[link]]
  dst_path      text,                          -- null when unresolved
  first_seen_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (vault_id, src_path, dst_title)
);
CREATE INDEX edges_unresolved ON graph_edges(vault_id) WHERE dst_path IS NULL;

-- accounting ----------------------------------------------------------
CREATE TABLE usage (
  vault_id uuid NOT NULL REFERENCES vaults(id) ON DELETE CASCADE,
  at       timestamptz NOT NULL DEFAULT now(),
  stage    text NOT NULL,
  provider text NOT NULL,
  model    text NOT NULL,
  units    double precision NOT NULL,
  unit     text NOT NULL,                      -- tokens | audio_seconds | images
  usd      numeric(10,5)
);
CREATE INDEX usage_window ON usage(vault_id, at DESC);
```

### 5.3 Why staged notes live in Postgres, not the repository

A capture is written to `notes` with `status = 'staged'` and **nothing is committed**.
Promotion is the first git write.

This is the opposite of v0.1 and it is deliberate: the repository stays curated, `/undo`
becomes "click discard" instead of a revert commit, and a run of bad captures leaves no
trace in history.

The cost is real: a staged note not yet reviewed exists only in our database. Mitigations —

- Postgres is backed up with point-in-time recovery; staged notes are covered.
- Raw capture blobs go to object storage immediately, so the source survives independently.
- `vaults.auto_commit_inbox` (default off) restores v0.1 behavior for users who want the
  note on their phone's Obsidian before they review it.

State the trade-off in the UI. Do not let a user discover it after losing something.

---

## 6. Repository layout

vaultd is its own Cargo workspace. The parent `learn-hft` manifest already excludes it, so
nested crates resolve correctly:

```toml
# learn-hft/Cargo.toml
[workspace]
members = ["projects/*"]
exclude = ["projects/vaultd"]     # vaultd is a workspace of its own
```

```
projects/vaultd/
├── Cargo.toml                    # [workspace] members = ["crates/*"]
├── SPEC.md
├── docker-compose.yml            # postgres, redis, minio for local dev
├── migrations/                   # sqlx migrations, numbered
├── crates/
│   ├── vaultd-core/              # domain types, traits, errors. No I/O.
│   ├── vaultd-db/                # sqlx pool, RLS helpers, repositories
│   ├── vaultd-crypto/            # envelope encryption, key rotation
│   ├── vaultd-github/            # App auth, git data API, repo sync, webhooks
│   ├── vaultd-channels/          # Channel trait + telegram + http ingest
│   ├── vaultd-models/            # Transcriber/Vision/Structurer + providers
│   ├── vaultd-pipeline/          # capture stages, staging
│   ├── vaultd-graph/             # index, diff, orphans, wanted notes, layout
│   ├── vaultd-export/            # static site generator
│   ├── vaultd-api/               # axum: REST, SSE, webhooks, sessions
│   └── vaultd-worker/            # Redis stream consumers
└── web/                          # TypeScript SPA (Vite + React)
    ├── src/routes/{changes,graph,models,vault}.tsx
    ├── src/lib/api.ts
    └── src/graph/                # canvas renderer
```

`vaultd-core` has no I/O and no database types. Everything else depends on it; it depends
on nothing. That constraint is what keeps the pipeline testable without a container.

### Dependencies

Verified against crates.io on 2026-09-07. Pin exactly; re-verify before starting.

```toml
tokio        = { version = "1.53", features = ["rt-multi-thread","macros","signal","fs","time","sync"] }
axum         = { version = "0.8",  features = ["macros"] }
tower-http   = { version = "0.6",  features = ["trace","cors","limit"] }
sqlx         = { version = "0.9",  features = ["runtime-tokio","tls-rustls","postgres","uuid","chrono","json","macros"] }
redis        = { version = "0.32", features = ["tokio-comp","connection-manager","streams"] }
teloxide     = { version = "0.17", features = ["macros","webhooks-axum"] }
octocrab     = "0.54"
reqwest      = { version = "0.13", features = ["json","multipart","stream","rustls-tls"], default-features = false }
jsonwebtoken = "9"                      # GitHub App JWT (RS256)
chacha20poly1305 = "0.10"               # XChaCha20-Poly1305 for secrets at rest
aws-sdk-s3   = "1"                      # object store (R2-compatible)
serde        = { version = "1", features = ["derive"] }
serde_json   = "1"
figment      = { version = "0.10", features = ["toml","env"] }
jiff         = { version = "0.2", features = ["serde"] }
blake3       = "1.8"
governor     = "0.10"
backon       = "1.6"
async-trait  = "0.1"
thiserror    = "2.0"
anyhow       = "1"
tracing      = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter","json"] }
metrics      = "0.24"
metrics-exporter-prometheus = "0.18"
pulldown-cmark = "0.13"
unicode-segmentation = "1"
hmac = "0.12"
sha2 = "0.10"
subtle = "2"                            # constant-time comparison
url = "2"
ipnet = "2"                             # SSRF guard: private range checks
hickory-resolver = "0.25"               # resolve-then-pin

# dev
wiremock = "0.6"
proptest = "1.11"
```

Notes on choices:

- **`sqlx` over `diesel`** — compile-time-checked queries against a real schema, native
  async, and `sqlx migrate` in the same toolchain. RLS needs raw `set_config` calls, which
  sqlx makes trivial.
- **Redis Streams, not a list.** Consumer groups give acknowledgement, pending-entry
  inspection, and `XAUTOCLAIM` for jobs orphaned by a crashed worker. A `LPUSH`/`BRPOP`
  queue silently loses the job a worker was holding when it died.
- **`chacha20poly1305` over AES-GCM** — no AES-NI dependency for consistent performance on
  small instances, and XChaCha20's 192-bit nonce means random nonces without a counter.
- **No `serde_yaml`** (deprecated). Frontmatter is emitted by hand for deterministic key
  order and parsed with a maintained fork when reading back.

---

## 7. Core domain types

```rust
/// Trust boundary within a vault. Rendered as:
///   vault:<uuid>:telegram:dm:123456789
///   vault:<uuid>:telegram:group:-1001234567890:topic:42
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SessionKey(String);

/// Normalized inbound message. Nothing downstream knows what Telegram is.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Envelope {
    pub vault_id: VaultId,
    pub channel: ChannelId,
    pub channel_msg_id: String,
    pub session: SessionKey,
    pub sender: SenderRef,
    pub received_at: Timestamp,
    pub text: Option<String>,
    pub attachments: Vec<Attachment>,
    pub command: Option<Command>,
    pub raw: serde_json::Value,     // original update, kept for replay
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attachment {
    pub kind: AttachmentKind,       // Voice{secs} | Image{w,h} | Document | Video
    pub mime: String,
    pub size_bytes: u64,
    pub remote_id: String,          // telegram file_id — resolve lazily, they expire
    pub filename: Option<String>,
}

/// Output contract of the structuring stage. The model returns exactly this.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DraftNote {
    pub title: String,              // <= 80 chars, no newlines
    pub slug: String,               // ^[a-z0-9]+(-[a-z0-9]+)*$, <= 60 chars
    pub tags: Vec<String>,          // 1..=6, lowercase kebab
    pub summary: Option<String>,
    pub body_md: String,            // no frontmatter, no H1
    pub links: Vec<String>,         // proposed targets, validated against the index
}

/// What the review UI shows and what promotion consumes.
pub struct StagedNote {
    pub id: NoteId,
    pub draft: DraftNote,
    pub resolved_links: Vec<ResolvedLink>,
    pub dropped_links: Vec<String>,
    pub impact: GraphImpact,        // nodes_added, edges_added, would_be_orphan
    pub provenance: Provenance,     // models, durations, capture kind, cost
}
```

`Envelope::raw` is not decoration. When the adapter mis-parses something, replaying the
original update JSON is the only way to reproduce it.

---

## 8. Authentication

### 8.1 GitHub App, not an OAuth App

A GitHub App is the correct primitive here and it is what the setup UI shows:

- **Per-repository permissions.** The user selects which repositories vaultd can touch.
- **Repository creation** via `administration: write` on the account — a fine-grained PAT
  scoped to one repository cannot create that repository, which is the whole first step.
- **Installation tokens expire in one hour** and are minted on demand. We store only the
  installation id; there is no long-lived repository credential in our database at all.
- **One revoke button** at `github.com/settings/installations`, outside our control.

Permissions requested:

| Permission | Level | Why |
|---|---|---|
| Contents | read & write | Read notes, commit promotions and exports |
| Metadata | read | Mandatory |
| Administration | write | Create a new vault repository |
| Pages | write | Optional, only if the user publishes the export themselves |

### 8.2 Token flow

```
1. App JWT      RS256, signed with the app private key, 9-minute expiry (max 10)
2. POST /app/installations/{id}/access_tokens  ->  installation token, 1 h
3. Cache in Redis under inst:{id} with TTL = expiry - 5 min
4. Use as a Bearer token for that installation's repositories
```

The app private key is the platform's single most sensitive secret. It lives in the KMS or
the deploy secret store, never in Postgres, never on disk in the image.

### 8.3 Browser sessions

Login is the GitHub App's user-authorization flow (`/login/oauth/authorize` against the
App, not a separate OAuth App). We exchange the code for a user token once, read the user's
id and login, then discard the user token — it is never stored. A `sessions` row and an
opaque cookie carry identity from there.

Cookie: `HttpOnly`, `Secure`, `SameSite=Lax`, 30-day sliding expiry, cleared server-side on
sign-out. State parameter is a random value in a short-lived signed cookie, verified on
callback, to close CSRF on the authorization step.

For headless installs (`vaultctl` on a server), the device flow applies. Same result.

---

## 9. Secret custody

### 9.1 Envelope encryption

```
KMS master key  (never leaves the KMS)
   └── wraps: per-vault data key (DEK), stored in vault_keys.wrapped_dek
          └── encrypts: secrets.ciphertext  (XChaCha20-Poly1305)
```

- The DEK is unwrapped on use and held in memory for the duration of a request or job.
- AAD binds ciphertext to its context: `vault_id || purpose || key_version`. A ciphertext
  moved to a different vault or purpose fails to decrypt rather than decrypting into the
  wrong place.
- Nonces are random 24 bytes (XChaCha20's nonce space makes collision negligible without
  a counter).
- `key_version` supports rotation: rewrap DEKs under a new master, re-encrypt secrets
  lazily on next write, sweep the remainder in a background job.

Secrets are typed by `purpose` (`anthropic_api_key`, `telegram_bot_token`, …), never a
free-text bag. A `Secret<String>` newtype whose `Debug` prints `[redacted]` means
`#[derive(Debug)]` on a config struct cannot leak a key into a log line.

### 9.2 What we cannot honestly promise

Marketing copy in this space likes the phrase "zero knowledge". It does not apply here, and
the spec should say so plainly.

The capture pipeline runs while the user is asleep. To transcribe a voice note that arrives
at 3 a.m., the worker must be able to decrypt that vault's API key without a user session.
Therefore the platform can decrypt user keys. Full stop.

What we can honestly offer:

- Keys encrypted at rest with per-vault isolation, so a leaked database row is not a leaked
  key without the KMS.
- **Bring your own key** — the user's own provider account, so spend and revocation are
  theirs and a compromise is bounded by their provider's controls.
- **Connector mode (§14.3)** — model calls never leave the user's hardware, so there is no
  provider key on our side at all. This is the real answer for users who need it.

Say this in the product, not only in the spec.

---

## 10. Channels

### 10.1 The trait

Four duties, straight from OpenClaw: authenticate, parse inbound, enforce access, format
outbound.

```rust
#[async_trait]
pub trait Channel: Send + Sync + 'static {
    fn kind(&self) -> ChannelKind;

    /// Parse a platform payload into normalized envelopes. Pure where possible —
    /// no network, so it is trivially testable against fixture JSON.
    fn parse(&self, raw: serde_json::Value) -> Result<Vec<Envelope>, ChannelError>;

    /// Resolve a platform file reference into bytes. Called lazily, only after
    /// policy has approved the message.
    async fn fetch_attachment(&self, a: &Attachment) -> Result<Bytes, ChannelError>;

    async fn send(&self, to: &SessionKey, reply: Reply) -> Result<String, ChannelError>;
    async fn edit(&self, to: &SessionKey, msg_id: &str, reply: Reply) -> Result<(), ChannelError>;
    async fn probe(&self) -> Result<ChannelHealth, ChannelError>;
}

pub enum Reply {
    Text { markdown: String },
    Status { markdown: String },     // the editable draft
    Document { name: String, bytes: Bytes, caption: Option<String> },
}
```

Splitting `parse` (sync, pure) from `fetch_attachment` (async, network) is what makes the
adapter testable and what stops a stranger's 100 MB video from costing us bandwidth before
the policy gate runs.

The built-in channel implements the same trait. It is not a special case bolted beside the
adapters — it is the adapter with the least work to do, which is why it can be the default.

### 10.2 Built-in capture — the default channel

Every vault has a `web` channel from the moment it is created. No token, no bot, no webhook,
no pairing. Its sender is the authenticated session owner, so access control is the session
cookie and nothing else.

**Surface.** A conversation in the app: a thread of what you sent and what came back, and a
composer with a text field, a record button and an attach button. Add the app to a phone's
home screen and it is a capture app — `MediaRecorder` for voice, `<input capture>` for the
camera. No app store, no review, ships when we ship.

**Input path.**

| Input | How it arrives |
|---|---|
| Text | `POST /api/vaults/:v/capture` with a JSON body |
| Voice | `MediaRecorder` → `audio/webm;codecs=opus` blob → presigned PUT → capture references the blob key |
| Image | File or camera → presigned PUT (client-side downscale above 2048px) |

Large media never passes through the API process. The client asks for a presigned URL, PUTs
straight to object storage, and posts only the resulting key. This keeps a 20 MB voice note
off the request path that also serves the review queue.

**Reply path.** The conversation is not fire-and-forget. Each capture's row updates live over
SSE through the pipeline stages, and the final message is the staged note itself — title,
tags, resolved links, graph impact — with **Promote**, **Edit** and **Discard** on the card.

That collapses capture and review into one surface. Say the thing, see what it became, keep
it or throw it away, without ever leaving the conversation. The Changes tab is then exactly
what its name says: the backlog of captures you chose not to handle immediately.

**Not an assistant.** The reply is the note, not an answer. The model still gets one
schema-locked call with no tools (§14.2). A vault Q&A assistant is a plausible next product
and it is deliberately not this one — see open question 7.

**Offline.** Captures made without a connection queue in IndexedDB and flush on reconnect.
Phone capture happens in exactly the places signal is bad; a capture surface that loses
input when offline is worse than no capture surface, because you believe it worked.

### 10.3 Telegram — an optional integration

Telegram is one of several integrations a user may add. None are required and setup can skip
all of them. The others:

- **Email-in** — a per-vault address (`<slug>@in.vaultd.app`). Forward anything. Attachments
  become captures. Sender must be a verified address on the account.
- **API token** — `POST /api/vaults/:v/capture` with a bearer token, for iOS Shortcuts,
  scripts, watch complications, whatever the user builds.

All of them normalize to `Envelope` and land in the same queue, the same review, the same
graph. An integration adds a way in, never a second pipeline.

What follows is Telegram-specific. Everything in §10.3 through §10.5 and §10.8 applies to
that integration only.

**Webhook, not polling.** v0.1 defaulted to long polling. A hosted platform cannot: polling
means one outbound connection per tenant bot, forever.

- One bot per vault. Its token is a vault secret.
- `setWebhook` to `https://api.vaultd.app/hooks/tg/{webhook_path}` with a `secret_token`.
- `webhook_path` is a 32-byte random value stored in `channels`; the `secret_token` is a
  separate value. An attacker needs both.
- Verify `X-Telegram-Bot-Api-Secret-Token` in constant time (`subtle::ConstantTimeEq`).
  Reject with `401` and an empty body — never say which half was wrong.
- The handler does nothing but validate, dedupe, insert a `captures` row, and `XADD` to
  Redis. Telegram retries aggressively on slow responses; target under 50 ms.

### 10.4 Inbound normalization (Telegram)

1. **Chat kind → session scope.** Private → `dm`, group/supergroup → `group`. Supergroup
   ids are negative and prefixed `-100`; carry them as strings, never truncate to `i32`.
2. **Forum topics.** `message.thread_id` on a forum chat appends `:topic:<id>`. Different
   topics are different sessions and may route to different inbox subfolders.
3. **Media selection.** Photos arrive as an array of sizes; take the largest under the cap.
   Voice stays OGG/Opus — do not transcode, providers accept it.
4. **Caption is text.** A photo with a caption is one envelope with both.
5. **Media groups.** Album photos arrive as separate updates sharing `media_group_id`.
   Buffer 1.5 s in Redis keyed by that id, then emit one envelope. Without this, a
   three-photo whiteboard becomes three notes.
6. **Size gate before download.** Reject over the cap at parse time.

### 10.5 Outbound formatting (Telegram)

**HTML parse mode, not MarkdownV2.** MarkdownV2 requires escaping eighteen characters,
including inside link URLs, and one miss returns an opaque 400. HTML needs three (`&`, `<`,
`>`) and covers everything we emit. Render through `pulldown-cmark` into Telegram's subset:
`<b> <i> <u> <s> <code> <pre> <a> <blockquote> <tg-spoiler>`. Unsupported constructs
degrade visibly — headings become bold lines, tables become code blocks.

**Chunking.** Telegram's 4096 limit counts **UTF-16 code units**. In Rust, `s.len()` is
bytes and `s.chars().count()` is scalars; neither is correct, and an emoji outside the BMP
is one `char` but two UTF-16 units.

```rust
fn utf16_len(s: &str) -> usize { s.encode_utf16().count() }
```

Chunker contract, in priority order:

1. Never exceed 4000 UTF-16 units per chunk.
2. Split at blank line → newline → sentence → word → grapheme cluster. Never mid-grapheme.
3. Never split inside a fenced code block; close and reopen the fence with its language tag.
4. Never split inside an HTML tag or entity.
5. Reassembly with injected fences removed equals the input exactly.

Property-test this (§21). Failures are silent — Telegram 400s and the user sees nothing.

**Rate limits.** ~30 messages/second global, 20/minute per group, and edits throttle harder
— treat as one per second per chat. `governor` with a per-chat keyed bucket in Redis so the
limit holds across worker replicas. Coalesce status edits: replace the pending value rather
than queueing both.

### 10.6 The reply, on every channel

One message, edited in place, exactly as OpenClaw does it:

```
📥  queued
🎧  transcribing (0:12 voice note)…
🧠  structuring…
🔗  resolving links against 213 notes…
📄  staged for review — vaultd.app/…/changes
```

Five messages would be lock-screen noise. One edited message is a progress bar. The final
state links into the review queue rather than claiming the note was filed.

The built-in channel renders the identical sequence inline over SSE, then replaces it with
the note card. Same stage events, same copy, one implementation — `Reply::Status` for a
channel that can edit, an SSE frame for one that can re-render. A channel that could do
neither would fall back to a single terminal message, and none currently do.

### 10.7 Access control (integrations only)

Applies to integrations that accept messages from arbitrary senders. The built-in channel
has no policy: its sender is an authenticated session, already scoped by RLS to one account.

```rust
pub enum DmPolicy { Pairing, Allowlist, Open, Disabled }
```

Default `Pairing`. `Open` requires an explicit wildcard — you cannot reach an open bot by
forgetting a setting.

Pairing: unknown sender gets a 6-character code with a **1 hour TTL**; their message is
discarded, never queued, never downloaded, never sent to a model. The owner approves in the
web app or from their own paired session. Codes are single-use, compared in constant time,
and pairing attempts are limited to 3 per sender per hour.

### 10.8 Known Telegram traps

- **Privacy mode is on by default.** In a group the bot sees only commands and replies to
  itself until it is disabled in BotFather or the bot is an admin. `probe()` must detect
  and surface this — it is the most common "why is nothing happening".
- **`getFile` caps downloads at 20 MB** even though uploads allow more. Reject above that
  with an accurate message.
- **Download URLs expire** in about an hour; resolve and fetch in the same job.
- **403 means the user blocked the bot.** Mark the session dormant, stop retrying.

---

## 11. Capture pipeline

Four stages, each persisting its output before the next begins, ending in a **staged** note
rather than a commit.

```rust
pub async fn run(ctx: &Ctx, cap: &mut Capture) -> Result<NoteId> {
    let mat  = resume_or(cap, Stage::Materialized, || materialize(ctx, cap)).await?;
    let ext  = resume_or(cap, Stage::Extracted,    || extract(ctx, &mat)).await?;
    let dft  = resume_or(cap, Stage::Structured,   || structure(ctx, &ext)).await?;
    stage(ctx, cap, dft).await                       // writes notes(status='staged')
}
```

**Stage 1 — materialize.** Download each attachment via `Channel::fetch_attachment`, hash
with blake3, put to object storage at `vaults/{vault_id}/blobs/{hash}`, insert into `blobs`.
An existing hash skips the download. Re-check size (declared sizes lie) and sniff the real
MIME from magic bytes rather than trusting the platform label.

**Stage 2 — extract.** Audio → `Transcriber`, biased with the previous note's title from
this session (measurably better on proper nouns: "Xilinx", "seqlock", "tick-to-trade").
Images → `VisionExtractor` with a fixed prompt: *transcribe all visible text verbatim, then
describe diagrams structurally; do not interpret or summarize.* Interpretation here destroys
information the structuring stage needs. Empty result after extraction fails `Permanent`
with "nothing readable in that capture".

**Stage 3 — structure.** The only stage where a model decides anything. Input is the raw
capture, the vault index, and the vault's house style. Output is a `DraftNote`.

**Stage 4 — stage.** Validate slug and links, compute graph impact, insert `notes`, publish
an SSE event so an open browser updates without a refresh, and edit the Telegram status
draft to its final state.

### Retry and failure

Exponential backoff with jitter via `backon`: 2 s, 8 s, 30 s, 2 m, 10 m, then dead-letter.

```rust
pub enum Severity {
    Transient,   // 5xx, timeout, 429  -> retry with backoff
    Permanent,   // 400, schema violation, oversized -> dead-letter, tell the user why
    Fatal,       // revoked installation, suspended account -> stop the channel, alert
}
```

Retrying a `Permanent` error burns money for a guaranteed failure — classify honestly. On
429 respect `Retry-After` over the backoff curve. Dead-lettered captures keep their
artifacts; replay resumes from the last successful stage.

---

## 12. Review and promote

This is the product loop. Nothing else in the system matters if this is not good.

### 12.1 States

```
staged ──promote──> promoted        (git commit, graph updated, export rebuilt)
   │
   ├──edit──> staged                (title, tags, body; no commit)
   └──discard──> discarded          (soft delete, never touches git)
```

### 12.2 What review shows

For each staged note:

- **Diff** — the unified diff of the file that would be created, frontmatter included.
  Generated from the rendered note, not hand-assembled, so what you approve is byte-for-byte
  what gets committed.
- **Graph impact** — nodes added, edges added, which existing notes it attaches to, and
  which invented links were dropped.
- **Orphan warning** — if the note would resolve zero links, say so before it is committed,
  not after. An orphan is not necessarily wrong; it usually means a hub note is missing.
- **Provenance** — which models ran, how long, what it cost.

### 12.3 Promotion

```rust
pub struct PromoteRequest {
    pub note_id: NoteId,
    pub target_dir: String,     // chosen from the vault's directory list — never free text
    pub title: Option<String>,  // user override
    pub tags: Option<Vec<String>>,
    pub keep_draft_flag: bool,  // default false on promote
}
```

Server-side sequence, all inside one transaction plus one commit:

1. Re-validate `target_dir` against the vault's actual directory listing. A directory that
   does not exist is a `400`, not a `mkdir`.
2. Sanitize the slug again (§12.4). Build `path = target_dir/YYYY-MM-DD-slug.md`.
3. Render the note. `draft` is set by the renderer, not the model.
4. Commit note plus any attachments atomically via the git data API (§16.2).
5. Update `notes` (status, path, commit_sha, promoted_at), upsert `graph_nodes`, insert
   `graph_edges`, re-resolve edges that were previously unresolved and now match this title.
6. Enqueue an export rebuild.

Step 5's re-resolution matters: a note written last month that linked `[[Colocation]]` when
no such note existed becomes connected the moment you promote a note by that title. The
graph heals backwards.

### 12.4 Path construction — the rule that contains prompt injection

```rust
fn sanitize_slug(raw: &str) -> Result<String, ValidationError> {
    let s: String = raw.trim().to_lowercase().chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    let s = s.split('-').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-");
    let s: String = s.chars().take(60).collect();
    if s.is_empty() { return Err(ValidationError::EmptySlug); }
    Ok(s)
}
```

ASCII only, dash separated, length capped. It cannot produce `.`, `/`, or `..`. The
directory comes from a validated list, never from the model, and never from free-form user
text. Assert the resolved path starts with an allowed directory before writing — belt and
braces, costs nothing.

**No LLM-produced string is ever passed to a path join.** This single rule is what makes
prompt injection boring: the worst outcome is a weird staged note that a human reads and
discards.

### 12.5 Frontmatter

Emitted by hand for deterministic key order.

```yaml
---
title: FPGA timing closure notes
tags: [fpga, timing, hardware]
draft: false
source: telegram
capture_kind: voice
captured: 2026-09-07T09:12:44+07:00
promoted: 2026-09-07T10:41:02+07:00
duration_s: 12
transcribe_model: connector:whisper-large-v3
structure_model: anthropic:claude-opus-5
vaultd_note_id: 0192f3a1-...
---
```

Provenance is not bureaucracy. When a note reads wrong six months later, the first question
is whether the transcript was bad or the structuring was bad, and without
`transcribe_model` and `duration_s` there is no way to tell.

---

## 13. Graph

### 13.1 Index

Built from two sources and kept identical by both paths:

- **Promotion** — the note we just wrote, parsed from the rendered Markdown.
- **Repo sync** — a GitHub `push` webhook, for notes edited in Obsidian and pushed from a
  laptop. Compare `head_sha` to the new head, fetch only changed paths, reparse those.

Both call the same `index_note(vault_id, path, markdown)`. A second code path that parses
Markdown slightly differently is how graphs drift out of sync with their repositories.

Parsing extracts: frontmatter `title`, `aliases`, `tags`, `draft`; body `[[wikilinks]]`
(including `[[Target|display]]` and `[[Target#heading]]` forms); and inline `#tags`.

### 13.2 Link resolution

A link resolves if its target matches a node's `title` or one of its `aliases`,
case-insensitively, after trimming. Unresolved links are kept, not discarded — they are the
most useful signal the graph produces:

```sql
-- "wanted notes": what the vault keeps trying to link to but does not have
SELECT dst_title, count(*) AS wanted_by
FROM graph_edges
WHERE vault_id = $1 AND dst_path IS NULL
GROUP BY dst_title ORDER BY wanted_by DESC LIMIT 20;
```

Three notes linking `[[Colocation]]` when no such note exists is the vault telling you what
to write next.

### 13.3 Diff and orphans

Time-windowed diff needs no snapshots — `first_seen_at` on nodes and edges makes it a
`WHERE` clause:

```sql
SELECT path, title, first_seen_at FROM graph_nodes
WHERE vault_id = $1 AND first_seen_at > now() - $2::interval;
```

Orphans are nodes with no resolved edge in either direction:

```sql
SELECT n.path, n.title FROM graph_nodes n
WHERE n.vault_id = $1
  AND NOT EXISTS (SELECT 1 FROM graph_edges e
                  WHERE e.vault_id = n.vault_id
                    AND (e.src_path = n.path OR e.dst_path = n.path)
                    AND e.dst_path IS NOT NULL);
```

Orphan and wanted-note detection are what make the graph a management tool rather than
decoration. A knowledge base grows in clusters; seeing which cluster this week attached to
tells you what you are actually thinking about, and seeing what attached to nothing tells
you where the structure is missing.

### 13.4 Layout stability

Positions are persisted on `graph_nodes` (`x`, `y`). This is not an optimization.

Users memorize the shape of their own graph. A force-directed layout recomputed from
scratch on every load produces a different picture each time and destroys that recognition.
Instead:

- Full layout runs server-side in `vaultd-graph`, in a background job, seeded from existing
  positions.
- New nodes are placed by one short incremental relaxation pinned against their neighbours;
  existing nodes barely move.
- Full recompute only on explicit request or when node count changes by more than 20%.

The frontend renders precomputed coordinates to canvas. It runs no simulation, which also
means a 2000-node vault opens instantly on a phone.

---

## 14. Models

### 14.1 Traits

```rust
#[async_trait]
pub trait Transcriber: Send + Sync {
    fn name(&self) -> &str;                      // "anthropic:…", "connector:…"
    async fn transcribe(&self, audio: &Blob, hint: Option<&str>)
        -> Result<(Transcript, Usage), ModelError>;
}

#[async_trait]
pub trait VisionExtractor: Send + Sync {
    async fn extract(&self, image: &Blob, prompt: &str) -> Result<(String, Usage), ModelError>;
}

#[async_trait]
pub trait Structurer: Send + Sync {
    async fn structure(&self, raw: &RawCapture, ix: &VaultIndex, house_style: Option<&str>)
        -> Result<(DraftNote, Usage), ModelError>;
}

#[async_trait]
pub trait Answerer: Send + Sync {
    /// Notes are already retrieved and ordered. This call selects nothing.
    async fn answer(&self, q: &Question, notes: &[RetrievedNote], history: &[Turn],
                    house_style: Option<&str>)
        -> Result<(Answer, Usage), ModelError>;
}
```

Four stages, four independently configurable slots. Answering is the one that runs many times
per captured note rather than once, so it is the slot where a smaller model usually wins —
`claude-sonnet-5` by default, against `claude-opus-5` for structuring.

Selected per vault from `model_slots`. Every implementation must honour a per-call timeout,
return `Usage` for cost accounting, classify errors into `Severity`, and **never retry
internally** — retry belongs to the queue, which persists attempt counts across restarts.

### 14.2 Structuring is schema-locked

Use forced tool use, not "please return JSON". With Anthropic: an `emit_note` tool whose
`input_schema` is the `DraftNote` schema, with `tool_choice: {"type":"tool","name":"emit_note"}`.
The model cannot reply with anything else, which removes the entire class of "it wrapped the
JSON in a code fence" failures.

Prompt skeleton:

```
You convert raw captured thought into one note.

Rules:
- Preserve the author's meaning and voice. Add no facts, opinions, or conclusions.
- Transcripts are speech: remove fillers and false starts, keep the content.
- Title: a specific noun phrase, <= 80 chars. Not a summary sentence.
- Tags: 1-6, lowercase kebab, drawn from the existing tags below when one fits.
- Links: only titles from the existing-notes list below. Never invent one.
- Body: Markdown, no H1, no frontmatter. Headings and lists only if the source has structure.
- If the capture is fragmentary, keep it fragmentary. Do not pad it into an essay.

Existing tags: {tags}
Existing notes: {titles}

{house_style}
```

`house_style` is the vault's own text, appended last so it wins. It is the difference
between a generic note and one that sounds like the person who dictated it — the closest
thing to "your own model" that does not involve fine-tuning:

> Keep my voice. I write in short declaratives and I do not hedge. Prefer the vault's
> vocabulary: say "tick-to-trade", not "end to end latency". When I speculate, mark it under
> `## Open`, never as a conclusion.

Index budget: cap the injected index at 8k tokens; beyond that, pre-filter by keyword
overlap with the capture text.

### 14.3 Self-hosted models — the connector

**The problem the pivot created.** v0.1 ran on the user's machine, so
`http://127.0.0.1:11434/v1` worked. Our workers cannot reach the user's laptop. A hosted
platform with a user-supplied `base_url` has two bad options and one good one:

| Option | Verdict |
|---|---|
| Fetch the user's URL from our workers | **Server-side request forgery.** A `base_url` of `http://169.254.169.254/…` reads cloud instance metadata; `http://10.0.0.5:6379` reaches our own Redis. Unacceptable as drawn. |
| Require a public HTTPS endpoint | Works, but the user must expose their inference server to the internet. Most will not, and the ones who do will do it badly. |
| **Connector agent** | Correct. Ship a small Rust binary the user runs next to their models. |

**Connector design.** The agent dials out to `wss://api.vaultd.app/connector` with a
per-vault token, holds the connection, and receives model-call requests over it. Nothing
inbound, no port forwarding, no public endpoint. It speaks the OpenAI-compatible protocol to
whatever is on `127.0.0.1` and streams the response back.

- Registered as `provider = 'connector'`, `base_url` interpreted **by the agent**, never by us.
- Per-vault token, revocable, rotates on rotation of the vault DEK.
- Offline connector → the slot fails `Transient`; the capture retries with backoff and the
  status draft says "waiting for your connector".
- Optional per-slot cloud fallback, off by default. Silently shipping audio to a cloud
  provider because a local one was down is exactly the betrayal this feature exists to
  avoid.

**If a public `base_url` is allowed at all**, it must pass a hard guard: HTTPS only,
resolve-then-pin (resolve the hostname, reject any address in a private, loopback,
link-local, or CGNAT range, then connect to the pinned IP with SNI preserved), no
redirects, 10-second connect timeout, response size cap. `ipnet` plus `hickory-resolver`.
Re-resolve on every call — DNS rebinding defeats a check done only at configuration time.

---

## 15. Ask — grounded recall

A vault that only accumulates is a landfill with good typography. Ask is what makes the pile
worth having: the same box, flipped to read mode, answering from what you already wrote.

The design constraint that makes this safe: **retrieval is a query we run, not a tool the
model calls.** We select the notes, then make exactly one call. No loop, no tools, no writes.

### 15.1 Mode routing

Two modes on one composer, explicit and sticky per vault: **Capture** (default) and **Ask**.

No automatic routing. The two failure directions are not symmetric — a capture misrouted to
Ask is a thought silently lost, which is the worst thing this product can do. An ask
misrouted to Capture merely produces a junk draft you discard. Given that asymmetry, guessing
is not worth it.

What we do instead: when Capture input parses as a short interrogative, the resulting note
card carries a secondary **Answer this instead** action. The thought is already saved; the
affordance teaches the mode without ever risking the input.

### 15.2 Retrieval

Postgres full-text search, then one hop of graph expansion.

```sql
CREATE TABLE note_content (
  vault_id   uuid NOT NULL REFERENCES vaults(id) ON DELETE CASCADE,
  path       text NOT NULL,
  title      text NOT NULL,
  body       text NOT NULL,
  status     text NOT NULL,                 -- staged | promoted
  updated_at timestamptz NOT NULL DEFAULT now(),
  tsv tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
    setweight(to_tsvector('english', coalesce(body,  '')), 'B')
  ) STORED,
  PRIMARY KEY (vault_id, path)
);
CREATE INDEX note_fts ON note_content USING GIN (tsv);
```

Maintained by the same `index_note()` that feeds the graph (§13.1 — one parse, one write
path, no drift).

```
1. lexical   websearch_to_tsquery over note_content, ts_rank_cd, top 8
2. expand    1-hop graph neighbours of the top 3 hits, up to 4 more notes
3. scope     promoted only by default; include_drafts is an explicit per-answer toggle
4. budget    <= 12 notes, <= 8k tokens; long notes truncated to headings + first 1200 chars
5. order     rank first, recency as tiebreak
```

**Step 2 is why the graph is not decoration.** The answer to "what did I decide about
colocation pricing" may live in a note that never says "colocation" but is linked from one
that does. Lexical search alone misses it. A vector store alone misses it too — semantic
similarity is not the same relation as *the author deliberately linked these*. Expansion
along edges the user drew by hand is retrieval no general-purpose index can reproduce.

Default scope is promoted notes: the vault is what you decided to keep, and drafts are noise
you have not yet judged. Including drafts is one toggle, and an answer that used them says so.

### 15.3 The answer call

One call, schema-locked the same way structuring is (§14.2): an `emit_answer` tool with
forced `tool_choice`.

```rust
pub struct Answer {
    pub answer: String,            // prose, inline [1] [2] markers
    pub citations: Vec<Citation>,
    pub sufficient: bool,          // false => the vault does not cover this
}

pub struct Citation {
    pub note: usize,               // index into the retrieved set
    pub quote: String,             // verbatim span copied from that note
}

/// What the client receives, after the server locates each quote.
pub struct ResolvedCitation {
    pub path: String,
    pub quote: String,
    pub span: Option<(usize, usize)>,   // char offsets, None when not located
}
```

Asking for a **verbatim quote** rather than a note index is what makes highlighting possible,
and it costs nothing extra: a model that must copy the sentence it is relying on is a model
that cannot cite a note it did not actually use.

System prompt:

```
Answer only from the numbered notes below. They are the user's own notes.

- Cite every claim with the note index it came from: [1], [2].
- For each citation, copy the exact sentence you relied on, verbatim, from that note.
  Do not paraphrase it, do not merge two sentences, do not trim mid-word.
- If the notes do not answer the question, set sufficient = false and say what is missing.
  Never answer from general knowledge. Never guess.
- If the notes disagree with each other, say so and cite both.
- Match the user's vocabulary as it appears in their notes.
- Distinguish what they decided from what they were still considering.

{house_style}

Notes:
[1] 20 - Deep Dives/32 - Order Book Engine.md
    …
```

Conversation memory: the last 3 turns are concatenated into the retrieval query and the last
6 into the prompt, capped at 2k tokens. No separate query-rewrite call in v1 — measure
whether follow-ups actually miss before paying for one.

### 15.4 Locating quotes — highlight or nothing

A citation is only useful if tapping it lands you on the sentence. That means turning the
model's quote into character offsets in the note, server-side, before the answer ships.

```
1. exact          find the quote verbatim in the note body
2. normalized     collapse whitespace, normalize quotes/dashes, retry
3. anchored       match first 6 and last 6 words, take the span between them
4. give up        span = None
```

Step 4 is the important one. **A citation whose quote cannot be located renders as a link to
the note with no highlight** — never a highlight placed by guessing. Highlighting the wrong
sentence is worse than highlighting none: it silently attributes a claim to text that does
not support it, and the user has no way to tell.

Log every miss with the quote and the note. A rising miss rate means the model is
paraphrasing instead of copying, which is a prompt regression worth catching early.

Offsets are computed against the **rendered Markdown source**, and the client maps them onto
the DOM after render. Store them on the answer row so reopening a week-old answer still
highlights, and recompute on reindex if the note changed — a note edited after the answer was
given shows the citation with a "this note changed since" marker rather than a stale span.

### 15.5 Related notes

The note view opened from a citation also lists related notes, ranked and **labelled by
reason**, never blended into an unexplained "related" score:

| Rank | Relation | Shown as |
|---|---|---|
| 1 | This note links to it | `→ linked from this note` |
| 2 | It links to this note | `← links here` |
| 3 | Shares a tag | `# shares tag #latency` |

All three come from links and tags the user wrote by hand. Nothing is inferred, and the
interface says so, because a "related" list that quietly mixes authored links with model
guesses teaches people to distrust both.

### 15.6 Citations and refusal

Server-side validation before the answer is shown:

- Every `note` index must exist in the retrieved set. Out-of-range means a fabricated
  citation; drop the answer and retry once, then surface the failure.
- Every `quote` must be locatable by §15.4 step 1, 2 or 3. A quote that is not present in the
  note at all — not merely unlocatable, but absent — is a fabrication: drop the whole answer
  and retry. This is the cheapest hallucination check available and it runs on every answer.
- `citations` empty or `sufficient == false` renders as a refusal: *"Nothing in your vault
  covers that."* — plus the wanted-note affordance (§13.2), because a question your vault
  cannot answer is precisely a note worth writing.
- The UI shows source paths as links, and an answer drawn only from unpromoted drafts carries
  a visible caveat. An answer that rests on one unreviewed capture is weaker than one drawn
  from four settled notes, and the interface should say which it is.

**Refusal is the product.** A second brain that confabulates is worse than no second brain,
because you will believe it — it is speaking in your own vocabulary, citing your own filing
system. Every incentive in this feature points at making refusal loud and cheap.

### 15.7 When to add vectors

Not yet, and the threshold is measurable rather than aesthetic.

Log every Ask with its retrieved set and whether the user acted on the answer. Add `pgvector`
when either holds over a rolling 200 questions:

- refusal rate above 20% on questions whose answer demonstrably exists in the vault, or
- the user's follow-up rephrases the same question more than 15% of the time.

Both mean lexical recall is failing. Until then, embeddings buy an embedding call per note,
a reindex path, a similarity threshold to tune, and no measured gain. Notes are the user's
own vocabulary, which is the case lexical search handles best.

When it lands: hybrid, not replacement. Reciprocal-rank fusion over FTS and vector results,
graph expansion unchanged on top.

### 15.8 Saving an answer

An answer card offers **Save as note**. It runs the ordinary staging path — same validation,
same review, same promote — with `capture_kind: answer` and provenance recording the question
and the source paths.

This matters structurally: there is exactly one way for text to enter the repository, and a
human triggers it. Ask gains no write path of its own.

### 15.9 Cost and abuse

Capture is naturally bounded — people record a handful of thoughts a day. Chat is not.

- Per-vault daily Ask cap and a monthly spend cap shared with the capture pipeline, hard-stop
  on breach with a clear message rather than a silent degrade.
- Rate limit per session: 20 questions/minute, token bucket in Redis.
- `usage` rows carry `stage = 'answer'` so the cost view separates recall from capture. They
  have very different shapes and averaging them hides both.

### 15.10 Security

Retrieved note content is **not fully trusted input**, even though it is the user's own vault.
A note can be the transcript of a photographed whiteboard someone else wrote, or a forwarded
email. Assume text inside notes may try to steer the answer.

Containment is structural rather than prompt-based:

- The answer call has no tools, no network, and no write path. The worst outcome of a
  successful injection is a wrong answer, and the citations point at the note that caused it.
- Note content is rendered as text. No `dangerouslySetInnerHTML`, no raw HTML in the Markdown
  renderer, no auto-linking of URLs found in note bodies into fetchable requests.
- Retrieval runs under RLS like every other query; the tenant-isolation test (§23) covers Ask.
- Questions and answers are logged at `debug` only, never at `info`. A question is often more
  revealing than the note it retrieves.

## 16. Repository integration

### 16.1 Vault discovery and adoption

After installation, list the installation's repositories. Classify each:

- **Vault** — contains `.vaultd/vault.json`. Offer to open.
- **Adoptable** — contains Markdown but no vault config. Offer to adopt: write the config,
  index existing notes, do not move or rewrite anything.
- **Empty** — offer to scaffold.

`.vaultd/vault.json` is the marker and holds only non-secret settings: schema version,
inbox directory, timezone, export toggle. It is committed, human-readable, and diffable.
Secrets never appear in it.

### 16.2 Writing

GitHub git data API, no local clone, atomic across note plus attachments:

```
1. GET   /repos/{o}/{r}/git/ref/heads/{branch}    -> base_sha
2. POST  /repos/{o}/{r}/git/blobs                 -> one per file (base64)
3. POST  /repos/{o}/{r}/git/trees                 -> base_tree = base_sha
4. POST  /repos/{o}/{r}/git/commits               -> parent = base_sha
5. PATCH /repos/{o}/{r}/git/refs/heads/{branch}   -> force = false
```

Step 5 returns **422** if the ref moved — the user pushed from a laptop mid-job. Refetch
from step 1 and retry, up to 3 times, then fail the promotion and tell them to retry.
**Never `force = true`.** The vault is also edited by a human on other devices; a forced
update destroys their work.

Commit message:

```
promote: FPGA timing closure notes

Captured 2026-09-07 via telegram (voice, 0:12)
vaultd-note: 0192f3a1-...
```

Batch promotions approved within 60 seconds into one commit.

### 16.3 Reading back

`push` webhook → verify `X-Hub-Signature-256` (HMAC-SHA256 over the raw body, constant-time
compare, **before** parsing JSON) → enqueue reindex. Compare `head_sha` to the new head via
the compare API, fetch only changed paths, reparse, update nodes and edges, delete nodes for
removed files.

Ignore pushes whose head commit we authored — check the commit's committer against our App
identity — or the platform reindexes its own writes on a loop.

---

## 17. Static export

`vaultd-export` renders promoted notes (`draft: true` excluded) into `site/`: static HTML,
one page per note, a search index, and `graph.json`. Committed on promote, in the same
commit as the note.

It is deliberately much less than the app: no graph diff, no model configuration, no review
queue, no auth. Those need a server holding credentials. What the export guarantees is that
the repository is useful without us — clone it and open `site/index.html`.

Quartz remains a supported alternative for users who already run it: set
`export.generator = "quartz"` and we commit a Quartz-compatible config plus a workflow
instead of our own HTML. Ours is the default because it has no Node build step and no
Actions minutes.

---

## 18. HTTP API

REST plus SSE. Everything is scoped by vault and everything runs under RLS.

```
POST   /auth/github/start                     -> redirect to GitHub
GET    /auth/github/callback                  -> session cookie
POST   /auth/signout

GET    /api/vaults                            -> list, with pending counts
POST   /api/vaults                            -> create repo + scaffold
GET    /api/vaults/:v
PATCH  /api/vaults/:v                         -> house_style, timezone, dirs, export
DELETE /api/vaults/:v                         -> disconnect (never deletes the repo)

POST   /api/vaults/:v/ask                     -> question; streams the answer over SSE
GET    /api/vaults/:v/ask/:a                  -> answer, citations, retrieved set
POST   /api/vaults/:v/ask/:a/save             -> stage the answer as a note

POST   /api/vaults/:v/capture                 -> text, or refs to uploaded blobs
POST   /api/vaults/:v/capture/upload          -> presigned PUT for audio or image
GET    /api/vaults/:v/chat?before=<cursor>    -> built-in capture conversation

GET    /api/vaults/:v/changes?status=staged
GET    /api/vaults/:v/changes/:n              -> draft, diff, impact, provenance
PATCH  /api/vaults/:v/changes/:n              -> title, tags, body
POST   /api/vaults/:v/changes/:n/promote
POST   /api/vaults/:v/changes/:n/discard

GET    /api/vaults/:v/graph?since=7d          -> nodes, edges, positions, orphans, wanted
POST   /api/vaults/:v/graph/relayout

GET    /api/vaults/:v/models
PUT    /api/vaults/:v/models                  -> slots; secrets by value, never returned
POST   /api/vaults/:v/models/:stage/test      -> probe, returns latency only

GET    /api/vaults/:v/channels
POST   /api/vaults/:v/channels                -> create bot binding, set webhook
POST   /api/vaults/:v/channels/:c/pair/:code

GET    /api/vaults/:v/usage?since=30d
GET    /api/vaults/:v/events                  -> SSE: capture.staged, promote.done, sync.done

GET    /api/vaults/:v/integrations
POST   /api/vaults/:v/integrations            -> telegram | email | api token

POST   /hooks/tg/:path                        -> Telegram, secret_token verified
POST   /hooks/email/:path                     -> inbound mail relay, signature verified
POST   /hooks/github                          -> HMAC verified
WS     /connector                             -> connector agent
```

Secrets are write-only through the API. `GET /models` returns provider, model, and a
`configured: true` flag — never a key, not even masked, because a mask still confirms
length and prefix.

SSE is what makes the built-in capture work at all: the conversation subscribes once and the
pipeline's stage events drive the reply in place, ending with the note card. The same stream
updates the review queue, so a capture from Telegram appears in an open browser without a
refresh.

`POST /capture` returns as soon as the row is durable — it does not wait for the pipeline.
The client renders an optimistic message and lets SSE fill it in. A capture that took 4
seconds to transcribe must never hold an HTTP request open for 4 seconds.

---

## 19. Frontend

**Mobile first, and not as a slogan.** The first release is a PWA on a phone; iOS follows.
Desktop is the layout that adapts upward, not the one that gets designed and then squeezed.
Every screen below was drawn at 392&#8202;px and only then widened.

TypeScript, Vite, React. Bottom tab bar — **Chat · Review · Graph · More** — with pushed
views (note reader, draft detail) that keep the tab bar and put back at top-left.

### 19.1 Screens

- **Chat** — the front door. One composer, two modes: **Note** stages a draft, **Ask** answers
  from the vault. The mode is explicit, sticky, and labelled with what it does, because one
  writes and one does not. Capture uses `MediaRecorder` (`audio/webm;codecs=opus`, or
  `audio/mp4` on Safari — feature detect, never sniff the user agent) and
  `<input type="file" accept="image/*" capture>`. Uploads go direct to object storage by
  presigned PUT. Replies stream over SSE and resolve into a card carrying **Keep**, **Read**
  and **Delete**.
- **Review** — the backlog. Rows with a state pill; tap for a draft detail showing a readable
  **preview**, not a diff, plus a folder picker and Keep / Edit / Delete.
- **Note reader** — pushed from a citation (§15.4). Highlighted spans, a jump bar stepping
  between them, and related notes below.
- **Graph** — canvas at desktop width; at phone width a portrait layout with fewer, larger
  labels. Server supplies coordinates (§13.4), so the client draws and hit-tests but runs no
  simulation — a 2000-node vault opens instantly on a phone.
- **More** — AI slots, vault, integrations, account.

### 19.2 Language

The interface is for someone capturing a thought on a train, not reading a spec.

| Internal | On screen |
|---|---|
| Promote | **Keep** |
| Staged | **Waiting** |
| Discard | **Delete** |
| Unified diff | **Preview** |
| Orphan node | **Not linked** |
| `+1 node, +3 edges` | **Connects to 3 notes** |
| Repository, commit SHA, frontmatter | behind **⚙ Technical details** |

Progressive disclosure, not dumbing down: every technical fact stays one tap away, and the
words that reach the surface are the ones a person would use out loud. "Promote" is a fine
word in this document and a bad button.

### 19.3 Touch and PWA

- 44&#8202;px minimum targets. Primary action bottom-right within thumb reach; destructive
  actions kept away from it.
- `env(safe-area-inset-bottom)` on every bottom bar.
- Installable manifest, `display: standalone`, maskable icon. Service worker caches the shell
  and holds the offline capture queue (§10.2) in IndexedDB.
- Optimistic sends: a capture appears in the thread immediately and reconciles on the SSE
  event. A failed send stays in the thread with a retry and never vanishes silently.
- iOS specifics to verify before the iOS build: `MediaRecorder` codec, PWA push permission
  (16.4+ and installed only), and the fact that iOS evicts service-worker storage after
  roughly seven weeks unused — the offline queue must tolerate being cleared.

## 20. Queue and concurrency

**Redis Streams**, one stream per job type, consumer groups per worker pool.

```
XADD  jobs:capture  * capture_id <uuid> vault_id <uuid>
XREADGROUP GROUP workers <consumer> COUNT 8 BLOCK 5000 STREAMS jobs:capture >
XACK  jobs:capture workers <id>
XAUTOCLAIM jobs:capture workers <id> 60000 0 COUNT 32   -- reclaim after a worker dies
```

Streams over a list because a crashed worker holding a `BRPOP`ped job loses it silently;
`XAUTOCLAIM` recovers it after an idle timeout.

**Per-session ordering.** Captures within one session run strictly in order; across sessions
they run concurrently. Reason: follow-ups. "Meeting notes about the FPGA build" then, ten
seconds later, "actually tag that one as hardware-procurement" only makes sense if the first
finished. Implemented with a Redis lock keyed on the session, held for the job's duration
with a watchdog-extended TTL; a worker that cannot take the lock re-queues with a short delay.

**Global fairness.** A per-vault concurrency cap (default 2) prevents one heavy user from
consuming the pool. A per-account monthly capture quota and a per-vault spend cap with a
hard stop protect against a runaway retry loop billing someone's card.

---

## 21. Security

### Threat model

| Threat | Mitigation |
|---|---|
| Cross-tenant data access | Postgres RLS on every tenant table, GUC set per transaction, API role without `BYPASSRLS`. Integration test asserts a query for vault B under account A returns zero rows. |
| Database compromise | Secrets are ciphertext; the DEK is wrapped by a KMS key that is not in the database. |
| App private key compromise | Full compromise of every installation. Key lives in KMS only, rotated on any suspicion, never in an image or repo. |
| Prompt injection | No tools, no agent loop. We compute paths; the model contributes a validated slug. Links filtered against the index. `draft` set by the renderer. Worst case is a weird staged note a human discards. |
| SSRF via self-hosted model URL | Connector agent by default (§14.3). Any public URL passes HTTPS-only, resolve-then-pin, private-range denial, no redirects, re-resolved every call. |
| Webhook forgery | Telegram `secret_token` and GitHub HMAC-SHA256, both constant-time, verified before parsing the body. |
| Malicious repository content | Note bodies are untrusted. The export escapes on render; the app never uses `dangerouslySetInnerHTML` on note content; Markdown rendering disables raw HTML. |
| Injection via retrieved notes | A note may be a transcribed whiteboard or a forwarded email. The answer call has no tools, no network and no write path, so the worst outcome is a wrong answer whose citation names the note that caused it (§15.10). |
| Chat cost exhaustion | Ask is unbounded where capture is not. Per-vault daily cap, per-session rate limit, hard stop on the shared monthly spend cap. |
| Unauthorized capture | Pairing default; strangers' messages discarded before download or model call. |
| Cost exhaustion | Per-vault spend cap with hard stop, per-account quota, `Permanent` errors never retried. |
| Session theft | HttpOnly + Secure + SameSite=Lax cookies, server-side revocation, 30-day sliding expiry. |
| Abuse of presigned uploads | URLs are single-use, expire in 5 minutes, pin content-length and content-type, and are issued only for a key under `vaults/{vault_id}/`. The capture referencing a key is rejected unless that key was issued to this vault. |
| Malicious upload content | MIME sniffed from magic bytes server-side, never trusted from the client. Images are re-encoded before vision; the original is stored but never served inline. |

### Data handling

- Capture text and transcripts are logged only at `debug`, which is off in production.
  Spans carry ids, never content.
- Blobs are deleted on vault disconnect and after a configurable retention window
  (default 90 days) once their capture is promoted or discarded.
- Deleting an account cascades to vaults, secrets, captures, blobs, and graph. It does not
  touch the user's repositories.

---

## 22. Errors

```rust
#[derive(Debug, thiserror::Error)]
pub enum VaultdError {
    #[error("channel: {0}")]  Channel(#[from] ChannelError),
    #[error("model: {0}")]    Model(#[from] ModelError),
    #[error("github: {0}")]   Github(#[from] GithubError),
    #[error("validation: {0}")] Validation(#[from] ValidationError),
    #[error("crypto: {0}")]   Crypto(#[from] CryptoError),
    #[error("store: {0}")]    Store(#[from] StoreError),
}

impl VaultdError {
    pub fn severity(&self) -> Severity;
    pub fn status(&self) -> StatusCode;
    /// One line, no jargon, no internals. Shown in the UI and the status draft.
    pub fn user_message(&self) -> String;
}
```

`thiserror` in libraries, `anyhow` only in binaries. Every user-visible error carries a
`user_message()` — "that voice note is 43 MB, the limit is 20 MB", not a debug-formatted
`reqwest::Error`.

---

## 23. Testing

**Unit**

- `sanitize_slug` — proptest that no input escapes the allowed directory and output always
  matches the slug regex. Seed with `../`, null bytes, RTL overrides, 10k chars, pure emoji.
- Chunker — proptest that every chunk is within the limit in UTF-16 units, concatenation
  round-trips, fences stay balanced, no grapheme splits. Seed with emoji, CJK, code blocks.
- Frontmatter — golden files, asserting `draft` is present and renderer-controlled.
- Crypto — encrypt/decrypt round trip; decryption fails when AAD's vault_id is swapped.
- Link resolution — aliases, case, `[[A|B]]`, `[[A#h]]`, and backward healing on promote.

**Integration**

- **Tenant isolation.** Seed two accounts, query every endpoint as A for B's vault, assert
  404 or empty. This test is the reason RLS is trustworthy; run it in CI on every commit.
- Built-in channel: capture → SSE stream → note card, asserting the stage sequence and that
  `POST /capture` returns before the pipeline finishes.
- Presigned upload: a key issued for vault A cannot be referenced by a capture in vault B.
- Ask: fabricated citation index is rejected; empty citations render as refusal; retrieval
  respects the promoted-only default; graph expansion pulls a note that lexical search alone
  misses; RLS prevents retrieval crossing vaults.
- Answer grounding: a fixture vault with a known answer, plus a control question the vault
  cannot answer, asserting refusal rather than a plausible fabrication. This is a regression
  test on trust, and it belongs in CI.
- Fixture corpus of real Telegram updates: voice, photo with caption, album, forum topic,
  command, oversized document, blocked user. `parse` is pure, so these are fast.
- Fake model implementations → the whole pipeline is deterministic and offline.
- `wiremock` for GitHub, including a 422 on the ref update to exercise conflict retry.
- Crash-resume: run to `Extracted`, kill the worker, restart, assert the transcription
  provider is not called again.
- SSRF guard: table-driven over `169.254.169.254`, `10.x`, `127.0.0.1`, `[::1]`,
  `100.64.x`, a hostname resolving to a private address, and a redirect to one.

**Manual, once per release**

- Real bot, real voice note, promote, confirm the commit and the graph edge.
- Connector agent against a local Ollama.

---

## 24. Deployment

Three deployables from one workspace: `vaultd-api`, `vaultd-worker`, and the SPA's static
build. Postgres with PITR, Redis with AOF, an S3-compatible object store, a KMS.

```dockerfile
FROM rust:1-alpine AS build
RUN apk add --no-cache musl-dev
WORKDIR /src
COPY . .
RUN cargo build --release -p vaultd-api -p vaultd-worker

FROM alpine:3
RUN apk add --no-cache ca-certificates
COPY --from=build /src/target/release/vaultd-api /usr/local/bin/
COPY --from=build /src/target/release/vaultd-worker /usr/local/bin/
```

Migrations run as a one-shot job with the owner role before rollout; the API and worker
roles have no DDL rights and no `BYPASSRLS`. Graceful shutdown on SIGTERM: stop reading the
stream, finish or checkpoint in-flight jobs, `XACK`, exit — bounded at 30 seconds.

Local development: `docker-compose up` for Postgres, Redis, and MinIO; `cargo run -p
vaultd-api`; `npm run dev` in `web/`. Telegram webhooks via a tunnel.

---

## 25. Milestones

| # | Deliverable | Proves |
|---|---|---|
| M0 | GitHub App install, list repos, create a vault, scaffold, one commit | Auth, tenancy, git write path |
| M1 | Built-in text capture → staged note → SSE reply. No models: title from first line | Channel trait, queue, staging, RLS under a worker, live reply |
| M2 | Note card in the conversation: diff, promote, discard. Commit on promote | The product loop |
| M3 | Voice and image: recorder, presigned upload, transcribe + vision + structure. Secrets encrypted. House style | Pipeline, crypto, per-vault config |
| M4 | Graph: index, edges, resolution, diff window, orphans, wanted notes | The reason this beats a folder |
| M5 | Repo sync webhook, static export, multi-vault switching | Bidirectional truth |
| M6 | Telegram integration; email-in and API token behind the same trait | That the channel abstraction was right |
| M7 | Connector agent for self-hosted models | The privacy promise, honestly kept |
| M8 | **Ask**: FTS + graph expansion, grounded answer, citations, refusal | The vault becomes worth having, not just worth filling |
| M9 | Save-as-note from an answer; wanted-note prompts from refusals | Recall feeding capture |
| M10 | Quotas, spend caps, usage view, metrics, PWA offline queue | Ready for people who are not you |
| M11 | iOS app — share sheet, background upload, lock-screen capture | The things a PWA cannot do on iOS |

M0 through M3 is the whole thesis: say something, see what it became, decide. Everything
after is amplification. If M3 does not feel good on a phone, no amount of M4 rescues it.

Telegram moving from M1 to M6 is the point of the previous revision. Building the integration
first made the integration the product; building the built-in surface first forces the channel
trait to be honest, because its first two implementations are as different as a web session
and a bot webhook.

**Ask sits at M8 on purpose, and it is worth defending.** It is the most demo-able feature
here and the most tempting to build early. It is also worthless before M4: retrieval needs
the graph for expansion, and answers need a vault with enough settled notes to be worth
asking. Built at M2 it would answer from four captures and look like a toy. Built at M8 it
answers from a real vault, which is the only condition under which anyone finds out whether
it is good.

---

## 26. Open questions

1. **Staged notes are only in Postgres.** `auto_commit_inbox` is the escape hatch, but the
   default is a real durability trade. Is a nightly "unreviewed captures" commit a better
   default than off?
2. **One bot per vault** means one BotFather registration per vault — friction at vault two.
   A single platform bot with vault routing by chat is friendlier but concentrates blast
   radius and makes the bot's identity ours rather than theirs.
3. **Adoption of a large existing vault.** Indexing 2000 notes on first connect is a long
   job with a bad first-run experience. Progressive indexing with a partial graph, or block
   until done?
4. **Conflict when a note is edited in Obsidian while staged edits exist.** Currently last
   writer wins on the file and we reindex. Probably fine; needs a real case to decide.
5. **Sharing.** Read-only vault sharing is the obvious next feature and the schema supports
   it (`vault_members`), but it changes RLS from account-scoped to membership-scoped. Design
   it before it is needed, do not build it yet.
6. **Pricing shape.** Bring-your-own-key means our cost is storage and compute, not tokens.
   That argues for a flat per-vault price rather than usage billing.
7. ~~Does the conversation ever answer?~~ **Resolved: yes.** §15 specifies grounded read-only
   Q&A with mandatory citations and loud refusal. The open part is narrower now: does an
   explicit mode toggle hold up in daily use, or do people forget which mode they are in and
   lose a thought to Ask? Instrument the mode switch and watch for questions typed in Capture
   mode — that ratio decides whether the toggle needs a smarter default.
9. **Do refusals convert?** A refusal names a note the vault is missing, which should feed
   straight into the wanted-note list (§13.2). If people ignore that prompt, the loop between
   recall and capture is not closing and the feature is half of what it looks like.
8. **Voice codec on Safari.** `MediaRecorder` gives Opus in WebM on Chrome and AAC in MP4 on
   Safari. Providers accept both, but transcription quality and per-second billing differ.
   Worth measuring before assuming it does not matter.

---

## 27. Sources

- [OpenClaw repository](https://github.com/openclaw/openclaw)
- [OpenClaw Telegram channel documentation](https://github.com/openclaw/openclaw/blob/main/docs/channels/telegram.md)
- [OpenClaw Architecture, Explained (ppaolo)](https://ppaolo.substack.com/p/openclaw-system-architecture-overview)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [GitHub Apps authentication](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app)
- [GitHub Git Database API](https://docs.github.com/en/rest/git)
- [PostgreSQL row security policies](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
- [Redis Streams consumer groups](https://redis.io/docs/latest/develop/data-types/streams/)
