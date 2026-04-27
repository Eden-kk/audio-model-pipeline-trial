# Audio Model Trial — Frontend

Vite 6 + React 19 + TypeScript 5 + Tailwind CSS 3 web app for the audio model playground.

## Local development

### Prerequisites

- Node ≥ 18
- [pnpm](https://pnpm.io/) ≥ 8 (`npm install -g pnpm`)

### Install

```bash
pnpm install
```

### Run (dev server)

```bash
pnpm dev
```

Opens at `http://localhost:5173`. Hot-reload enabled.

### Build

```bash
pnpm build
```

Output in `dist/`.

## Pointing at a non-localhost backend

By default the frontend talks to `http://localhost:8000`. To use a different backend:

```bash
VITE_API_URL=https://my-backend.example.com pnpm dev
```

Or create a `.env.local` file:

```
VITE_API_URL=https://my-backend.example.com
```

The variable is picked up at **build time** by Vite — rebuild after changing it in production.

## Pages

| Page | Status | Description |
|------|--------|-------------|
| Playground | Live | Mic record or file upload → pick ASR adapter → run → transcript |
| Pipelines | P1 stub | DAG pipeline composer (coming in P1) |
| Run | P1 stub | Live DAG execution view (coming in P1) |
| Corpus | P1 stub | Clip library with scenario tagging (coming in P1) |
| Settings | P1 stub | API keys + cost caps (coming in P1) |

## Playground usage (no backend)

The adapter dropdown shows "No backend connected" gracefully when the backend is
unreachable. You can still type an adapter ID manually and the Run button will
attempt the calls (which will fail with a descriptive error shown inline).

## Stack

- **Vite 6** — build tooling
- **React 19** — UI framework
- **TypeScript 5** — type safety
- **Tailwind CSS 3** — utility-first styles
- **tailwind-merge + clsx** — conditional class utilities
- **react-router-dom 7** — client-side routing
- Native `fetch` + `WebSocket` — no extra HTTP library
