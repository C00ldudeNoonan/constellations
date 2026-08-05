# Vendored third-party asset

`3d-force-graph.min.js` — the rendering library for the concept-cloud
visualization (issue #255). It is vendored (rather than loaded from a CDN) so a
generated artifact is a single self-contained file that renders offline in any
browser.

- **Package:** `3d-force-graph` (bundles three.js)
- **Version:** 1.80.0
- **Upstream:** https://github.com/vasturiano/3d-force-graph
- **License:** MIT © Vasco Asturiano (https://github.com/vasturiano/3d-force-graph/blob/master/LICENSE)

## Regenerating / upgrading

```bash
curl -sL "https://unpkg.com/3d-force-graph@<version>/dist/3d-force-graph.min.js" \
  -o 3d-force-graph.min.js
```

The artifact renderer inlines this file at render time; the generated HTML also
carries a CDN fallback loader for the case the vendored copy is stripped.
