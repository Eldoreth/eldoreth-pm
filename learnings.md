# Sprint Learnings — SAP Munich Interview Prep

## Sprint 1 — Lovable (May 6, 2026)

### Mini-app 1: Flowly (BPMN Generator)
- Cost: 2 credits
- Pattern learned: DSL + live graph rendering, D3.js
- Surprise: immediately understood a custom invented DSL syntax
- SAP relevance: DSL → BPMN is exactly the AI-augmented authoring Signavio team is building

### Mini-app 2: AI Use Case Wizard
- Pattern learned: multi-step flow, ICE scoring, markdown export
- Surprise: free text input → contextual use cases (not generic)
- Markdown export working end-to-end
- PoV potential: discovery call → structured output in 10 minutes

### Mini-app 3: Event Log Inspector
- Pattern learned: file upload, PapaParse, data dashboard
- Sample dataset with intentional quality issues: realistic
- 5 automated data quality checks with precise counts
- Interactive bar chart with hover tooltip
- REAL POTENTIAL: replaces 2 days of Excel work in a PoV

### Lovable — rules learned
- Prompt structure: action + explicit output + specific constraints
- Don't polish practice apps: volume of experience matters more
- "No API call / client-side only" avoids exposing API keys in public projects
- Screenshot input works well for replicating existing layouts
- Upgrade to Pro before any technical assignment — free credits run out fast
- Export to GitHub for any app worth keeping

### How I use Lovable (for interview)
Claude chat designs the architecture and writes optimized prompts.
Lovable executes and generates the UI.
I direct, evaluate output, and connect it to real business value.
That's the PM role in vibe coding — not typing, directing.

---

## Sprint 2 — Claude Code (May 7, 2026)

### Setup
- Claude Code installed globally via npm on Schenker (Ubuntu, Python 3.12)
- Accessed remotely via VS Code Remote SSH through Cloudflare Tunnel (ssh.eldoreth.com)
- Port forwarding via VS Code Ports tab for local browser access

### What was built
- parser.py: XES + CSV event log parser, pydantic models, 31 tests, 97% coverage
- discovery.py: process discovery (alpha/inductive/heuristics), variants, bottlenecks, 33 tests, 100% coverage
- router.py + main.py: FastAPI layer, 3 endpoints, Swagger UI at /docs, 33 tests, 100% coverage
- Total: 631 lines, 97 tests, 99% coverage

### Claude Code — rules learned
- CLAUDE.md is the persistent system prompt: create it first, always
- Prompt structure: action + explicit output + key:value constraints
- Option 2 "allow all edits this session" = true vibe coding mode
- It reads existing files before writing — never codes blind
- Auto-fixes bugs (pandas 3.x breaking change fixed autonomously)
- Auto-updates README.md per CLAUDE.md rule
- Saves project memory across sessions

### How I use Claude Code (for interview)
Claude chat (this conversation) designs architecture and writes structured prompts.
Claude Code executes on the codebase autonomously.
I review, approve, and connect output to business value.
Pattern: You (goal) → Claude chat (architecture + prompts) → Claude Code (execution) → working codebase.
This is PM-level AI orchestration, not just coding.

### Key insight
97 tests, 99% coverage, Swagger UI live — built in one evening.
The differentiator is not writing code, it's knowing what to ask and recognizing good output.

---

## Final Round Prep — TODO (Sprint 5, May 20)
- [ ] MCP pricing for Signavio: study Anthropic, Salesforce Einstein, SAP Joule pricing models
- [ ] Behavioral: unblock internal stalled situation (2 STAR stories ready)
- [ ] Behavioral: brilliant but socially destructive colleague
