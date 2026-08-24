## What changed

Describe the change and why it is needed.

## Verification

- [ ] `python -m compileall -q app workers tools tests`
- [ ] `ruff check app workers tools tests`
- [ ] `pytest -q`
- [ ] `node --check static/app.js && node --test tests/test_project_state.js`
- [ ] The interface was reviewed manually when applicable
- [ ] No models, user media, exports, logs, secrets or personal paths were committed
